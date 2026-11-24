"""Verify native streaming output against frozen predictions and measure scaling."""

from copy import deepcopy
import ctypes as ct
import os
import pickle
import subprocess
import tempfile

from ablation import ROOT, chain, h, np, old, prepare, sha, write
from scipy.interpolate import BSpline, PPoly

WINDOWS = (16, 32, 64, 128, 256, 512, 1024)
LENGTHS = (1024, 4096, 16384, 65536)
SCALE_N = 16384
NAMES = ("KCWA", "AR", "ARX")
REPEATS = 15
SOURCE = ROOT / "stream.c"
FLAGS = ("-O3", "-std=c11", "-ffp-contract=off", "-fPIC", "-shared")
F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def w_polynomial(contract):
    assert contract["family"] == "MONOTONE_I_SPLINE_CORRECTION"
    metadata = contract["basis"]
    edges = np.r_[metadata["train_min"], metadata["knots"], metadata["train_max"]]
    assert np.all(np.diff(edges)>0)
    lo, hi = edges[[0,-1]]
    augmented = np.r_[np.repeat(lo,4), edges[1:-1], np.repeat(hi,4)]
    coefficients = np.asarray(contract["coefficient"])
    columns = len(augmented)-4
    assert len(coefficients) == columns
    mid = (edges[1:]+edges[:-1])*.5
    poly = np.zeros((len(mid), 5))
    for j, weight in enumerate(coefficients):
        antiderivative = BSpline(augmented, np.eye(columns)[j], 3).antiderivative()
        pp = PPoly.from_spline(antiderivative)
        index = np.searchsorted(pp.x, mid, side="right")-1
        np.testing.assert_array_equal(pp.x[index], edges[:-1])
        piece = pp.c[:,index].T.copy()
        base = float(antiderivative(lo))
        piece[:,-1] -= base
        poly += piece * (contract["direction"]*weight/(antiderivative(hi)-base))
    return dict(edges=edges, poly=poly, mean=metadata["mean"], inv_scale=1./metadata["scale"])


def fit_parameters(entity, model, pipe):
    nodes = chain(model)
    k, c, w, a = (nodes[key] for key in "KCWA")
    cfg = dict(L=h.L, ref=k["reference_index"], K_active=k["routing_status"]=="ADMITTED",
               C_active=c["routing_status"]=="ADMITTED", W_active=w["routing_status"]=="ADMITTED", reps=[])
    raw_channels = old.sru.COLUMNS if entity == old.sru.ENTITY else h.COLUMNS
    cfg["width"] = len(raw_channels)
    cfg["joint"] = dict(mean=[0.,0.], weight=[0.,0.])
    for rep in k["representations"]:
        contract = rep["contract"]
        meta = contract["basis"]
        assert meta["kind"] == "continuous"
        joint = rep["channel"]=="joint_lift"
        if joint:
            pca = rep["joint_lift_pca"]
            assert raw_channels[1:3] == ("crystal_lift", "crucible_lift")
            cfg["joint"] = dict(mean=pca["mean"], weight=np.asarray(pca["pc1_loading"])/pca["scale"])
        cfg["reps"].append(dict(index=1 if joint else raw_channels.index(rep["channel"]), joint=joint,
            intervals=rep["intervals"], intercept=contract["intercept"], theta=np.asarray(contract["theta"]),
            mean=meta["mean"], inv_scale=1/meta["scale"], knots=meta["knots"], feature_mean=meta["feature_mean"]))
    if cfg["C_active"]:
        assert c["recipe"]=="ADDITIVE_COMPRESSED"
        z = c["contract"]
        cfg["C"] = dict(intercept=z["intercept"], mean=z["mean"], weight=np.asarray(z["coefficient"])/z["scale"])
    else:
        cfg["C"] = dict(intercept=0., mean=np.zeros(len(cfg["reps"])), weight=np.zeros(len(cfg["reps"])))
    if cfg["W_active"]:
        cfg["W"] = w_polynomial(w["contract"])
    else:
        cfg["W"] = dict(edges=[0.,1.], poly=np.zeros((1,5)), mean=0., inv_scale=1.)
    if a["routing_status"]=="ADMITTED":
        z = a["contract"]
        offsets = h.a_state.offsets(a["profile"], pipe.A_max)
        cfg["A"] = dict(offset=offsets, intercept=z["intercept"], mean=np.asarray(z["mean"])+a["residual_mean"],
                        weight=np.asarray(z["coefficient"])/z["scale"])
    else:
        cfg["A"] = dict(offset=[], intercept=0., mean=[], weight=[])
    with (old.baseline.ROOT/entity/"linear.pkl").open("rb") as handle:
        linear = pickle.load(handle)
    cfg["linear"] = []
    for name in NAMES[1:]:
        z = linear[name]
        cfg["linear"].append(dict(intercept=float(z["ridge"].intercept_), mean=z["scaler"].mean_,
             weight=z["ridge"].coef_/z["scaler"].scale_, delta=z["target"]=="delta"))
    return cfg


def header(cfg):
    L, reps, a, w, c = (cfg[k] for k in ("L","reps","A","W","C"))
    jt, linear = cfg["joint"], cfg["linear"]
    max_bins = max(len(r["intervals"]) for r in reps)
    max_width = max(r["theta"].shape[1] for r in reps)
    count = len(a["offset"])
    last = int(max(a["offset"], default=0))
    flags = dict(L=L, X_WIDTH=cfg["width"], STRIDE=cfg["width"]+1, DELAY=h.D,
                 K_CHANNELS=len(reps), K_REFERENCE=cfg["ref"], K_ACTIVE=int(cfg["K_active"]),
                 C_ACTIVE=int(cfg["C_active"]), W_ACTIVE=int(cfg["W_active"]),
                 W_SEGMENTS=len(w["edges"])-1, A_COUNT=count, A_RING=last+1, A_READY=last+h.D)
    rows = [f"#define {k} {v}" for k,v in flags.items()]

    def c_value(a):
        if a.ndim == 0:
            return str(int(a)) if a.dtype.kind in "biu" else float(a).hex()
        return "{" + ",".join(c_value(v) for v in a) + "}"

    def declare(name, values, kind="double"):
        a = np.asarray(values)
        dims = "".join(f"[{val}]" for val in a.shape)
        rows.append(f"static const {kind} {name}{dims} = {c_value(a)};")

    def pad(v, count):
        return np.pad(v, (0,count-len(v)))

    declare("joint_mean", jt["mean"])
    declare("joint_weight", jt["weight"])
    for name, values in dict(k_bins=[len(r["intervals"]) for r in reps],
         k_width=[r["theta"].shape[1] for r in reps], k_input=[r["index"] for r in reps],
         k_joint=[r["joint"] for r in reps], k_knot_count=[len(r["knots"]) for r in reps],
         k_near=[pad(np.asarray(r["intervals"])[:,0],max_bins) for r in reps],
         k_far=[pad(np.asarray(r["intervals"])[:,1],max_bins) for r in reps]).items():
        declare(name, values, "size_t")
    for key in ("intercept","mean","inv_scale"):
        declare("k_"+key, [r[key] for r in reps])
    max_feature = max(len(r["feature_mean"]) for r in reps)
    declare("k_knots", [pad(r["knots"],max_feature) for r in reps])
    declare("k_feature_mean", [pad(r["feature_mean"],max_feature) for r in reps])
    declare("k_theta", [np.pad(r["theta"], ((0,max_bins-r["theta"].shape[0]),(0,max_width-r["theta"].shape[1]))) for r in reps])
    for prefix, model in (("c", c), ("w", w), ("a", a)):
        for key, v in model.items():
            if key == "offset":
                declare("a_offset", v if count else [0], "size_t")
            else:
                declare(prefix+"_"+key, [0.] if prefix=="a" and key in ("mean","weight") and count==0 else v)
    declare("a_count", count, "size_t")
    dim = flags["STRIDE"]*L
    for key in ("mean","weight"):
        declare("linear_"+key, [pad(v[key],dim) for v in linear])
    declare("linear_intercept", [a["intercept"] for a in linear])
    declare("linear_delta", [a["delta"] for a in linear], "size_t")
    return "\n".join(rows)+"\n"


def build(cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"model.h").write_text(header(cfg), encoding="ascii")
    target = directory/"stream.so"
    subprocess.run(["gcc", *FLAGS, "-I", str(directory), str(SOURCE), "-o", str(target)], check=True)
    lib = ct.CDLL(str(target))
    lib.stream.argtypes = (ct.c_int, F64, ct.c_size_t, ct.c_size_t, F64)
    lib.stream.restype = ct.c_double
    return lib


def measure(lib, data, start, count, expected=None):
    out = np.empty(count)
    calls = [[] for _ in NAMES]
    error = {}
    for i,name in enumerate(NAMES):
        lib.stream(i, data, start, count, out)
        assert np.isfinite(out).all()
        if expected is not None:
            np.testing.assert_allclose(out, expected[name], atol=5e-10, rtol=1e-12)
            error[name] = float(np.max(np.abs(out-expected[name])))
    for repeat in range(REPEATS):
        for j in range(len(NAMES)):
            i = (repeat+j)%len(NAMES)
            calls[i].append(lib.stream(i, data, start, count, out))
    return {name: dict(median_ns_per_row=float(np.median(calls[i])),
                     P95_run_ns_per_row=float(np.quantile(calls[i],.95)),
                     median_total_ms=float(np.median(calls[i])*count/1e6),
                     replicate_ns_per_row=calls[i], replay_max_abs=error.get(name))
            for i,name in enumerate(NAMES)}


def synthetic(template, L, A, rng):
    cfg = deepcopy(template)
    cfg["L"] = L
    for r in cfg["reps"]:
        edges = np.rint(np.asarray([0, *[edge[1] for edge in r["intervals"]]])*L/h.L).astype(int)
        for i in range(1,len(edges)):
            edges[i] = max(edges[i],edges[i-1]+1)
        assert edges[-1] <= L
        r["intervals"] = list(zip(edges[:-1],edges[1:]))
    cfg["A"] = dict(offset=np.arange(A), intercept=0., mean=np.zeros(A), weight=rng.normal(0.,.005/np.sqrt(A), A))
    cfg["linear"] = [dict(intercept=0., mean=np.zeros(m*L), weight=rng.normal(0.,.01/np.sqrt(m*L),m*L), delta=True)
                     for m in (1,cfg["width"]+1)]
    return cfg


def counts(cfg):
    used = [r for i,r in enumerate(cfg["reps"]) if cfg["C_active"] or
            (i==cfg["ref"] and (cfg["K_active"] or cfg["W_active"]))]
    return dict(process_channels=len(used), K_bins=sum(len(r["intervals"]) for r in used),
                K_basis_products=sum(r["theta"].size for r in used),
                C_coefficients=len(cfg["C"]["weight"]) if cfg["C_active"] else 0,
                W_segments=len(cfg["W"]["edges"])-1 if cfg["W_active"] else 0,
                A_lags=len(cfg["A"]["offset"]))


def run():
    cores = os.sched_getaffinity(0)
    core = 12 if 12 in cores else min(cores)
    os.sched_setaffinity(0,{core})
    host = next(line.partition(":")[2].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    toolchain = subprocess.check_output(["gcc","--version"],text=True).splitlines()[0]
    actual, scaling = {}, []
    rng = np.random.default_rng(20261124)
    with tempfile.TemporaryDirectory(prefix="kcwa-native-", dir=ROOT) as tmp:
        root = h.Path(tmp)
        for entity in old.ENTITIES:
            geo, pipe, data, _, model = prepare(entity)
            cfg = fit_parameters(entity,model,pipe)
            start, stop = geo["development"]
            with np.load(old.path(entity,"predictions_private.npz")) as native_ref, \
                 np.load(old.ROOT/"baseline_replay"/entity/"linear_evaluation.npz") as linear:
                np.testing.assert_array_equal(native_ref["target_index"],linear["target_index"])
                expected = dict(KCWA=native_ref["v06_kcwa"], AR=linear["AR"], ARX=linear["ARX"])
            lib = build(cfg,root/entity)
            times = measure(lib,np.ascontiguousarray(data),start,stop-start,expected)
            actual[entity] = dict(N=stop-start,L=h.L,counts=counts(cfg),models=times)
            print("ACTUAL", entity, {k:round(v["median_ns_per_row"],1) for k,v in times.items()},flush=True)
            if entity == old.sru.ENTITY:
                template = cfg
        start = max(WINDOWS)+h.D
        n = start+max(LENGTHS)
        t = np.arange(n)
        x = .55+.07*np.sin(t[:,None]/(19+7*np.arange(template["width"]))) + rng.normal(0.,.02,(n,template["width"]))
        y = .3+.05*np.sin(t/31)+rng.normal(0.,.015,n)
        raw = np.ascontiguousarray(np.c_[y,x])
        for L in WINDOWS:
            for mode,A in (("fixed",16),("dense",L)):
                cfg = synthetic(template,L,A,rng)
                lib = build(cfg,root/f"L{L}_{mode}")
                for N in (LENGTHS if L==h.L else (SCALE_N,)):
                    times = measure(lib,raw,start,N)
                    scaling.append(dict(L=L,N=N,A_mode=mode,counts=counts(cfg),models=times))
                    print("SCALING",L,N,mode,{k:round(v["median_ns_per_row"],1) for k,v in times.items()},flush=True)
    write(ROOT/"COMPLEXITY.json",dict(hardware=host,core=core,compiler=toolchain,flags=FLAGS,precision="FP64",
          repeats=REPEATS,source_sha256=sha(SOURCE),time_scope="Native stream with maturity lookup and causal history indexed from input buffers. Includes model and output writes. Excludes preload, interpreter, I/O and compilation.",
          fixed_window_N=SCALE_N,fixed_N_window=128,actual=actual,scaling=scaling))


if __name__ == "__main__":
    run()
