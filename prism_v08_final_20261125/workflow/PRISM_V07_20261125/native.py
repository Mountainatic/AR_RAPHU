"""Compile and time the frozen v0.7 stream and archive comparators on real development rows."""

import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from v07 import CASES, D, ROOT, Engine, frozen, g, h, inputs, legacy, np, poisoned, rows, sha, write


DIAG = ROOT.parent/"PRISM_V06_KCWA_DIAGNOSTICS_20261124"
FLAGS = ("-O3","-std=c11","-ffp-contract=off","-fPIC","-shared")
REPEATS = 15
F64 = np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")


def c_value(value):
    a = np.asarray(value)
    if a.ndim:
        return "{"+",".join(c_value(v) for v in a)+"}"
    return str(int(a)) if a.dtype.kind in "ibu" else float(a).hex()


def c_row(*parts):
    return "{"+",".join(c_value(x) for x in parts)+"}"


def channel_map(entity,reps):
    names = legacy.sru.COLUMNS if entity==legacy.sru.ENTITY else h.COLUMNS
    out = []
    for rep in reps:
        name = rep["channel"]
        if name=="joint_lift":
            p = rep["joint_lift_pca"]
            weight = np.asarray(p["pc1_loading"])/p["scale"]
            out.append(c_row(2,3,*p["mean"],*weight))
        else:
            out.append(c_row(1+names.index(name),1,0.,0.,1.,0.))
    return len(names)+1,out


def view_config(source,view):
    c = view["contract"]
    basis = c["basis"]
    left,right = np.array(view["edges"][:-1]),np.array(view["edges"][1:])
    coef = np.array(c["theta"])
    width = coef.shape[1]
    low,high = np.array(view["support"])
    scale = np.maximum(high-low,1e-9*(1+np.maximum(np.abs(low),np.abs(high))))
    knots = np.asarray(basis["knots"])
    centre = basis["center"][:width]
    if width>1:
        last,ref = knots[-1],knots[-2]
        active = knots[:width-1]
        ks = np.r_[0.,active]
        inv = np.r_[0.,1./(last-active)]
        rs = 1./(last-ref)
    else:
        last,ref,rs = 0.,-1.,1.
        ks,inv = [0.],[1.]
    text = c_row(source,int(view["view"]=="delta"),len(left),width,
                 basis["mean"],1./basis["scale"],c["intercept"],last,ref,
                 centre,ks,inv,rs,left,right,1./(right-left),low,high,1./scale,coef)
    return text,dict(bins=len(left),basis_products=int(coef.size))


def w_config(pipe,model):
    if model is None:
        return dict(mean=0., inv_scale=1., edges=np.array([0.,1.]), poly=np.zeros((1,5)), error=0.)
    m = model["contract"]
    b = m["basis"]
    mean,scale = b["mean"],b["scale"]
    lo,hi = (np.asarray(model["limits"])-mean)/scale
    points = np.array([*b["knots"],b["train_min"],b["train_max"]])
    edges = np.r_[lo,np.unique(points[(points>lo)&(points<hi)]),hi]
    ratio = np.linspace(0.,1.,5)
    poly = []
    for a,c in zip(edges[:-1],edges[1:]):
        value = pipe.w.predict_w_correction(mean+scale*(a+(c-a)*ratio),m)
        poly.append(np.polynomial.polynomial.polyfit(ratio,value,4)[::-1])
    poly = np.array(poly)
    z = np.linspace(lo,hi,1001)
    i = np.clip(np.searchsorted(edges,z,side="right")-1,0,len(poly)-1)
    frac = (z-edges[i])/(edges[i+1]-edges[i])
    reconstructed = np.array([np.polyval(c,x) for c,x in zip(poly[i],frac)])
    expected = pipe.w.predict_w_correction(mean+scale*z,m)
    np.testing.assert_allclose(reconstructed,expected,atol=1e-11,rtol=0.)
    return dict(mean=mean,inv_scale=1./scale,edges=edges,poly=poly,
                error=float(np.max(np.abs(reconstructed-expected))))


def a_config(model):
    if model is None:
        return dict(offset=[0],nrecent=0,alpha=[0.],nema=0,shift=0,weight=[0.],bias=0.,family="bypass")
    family,z = model["spec"][0]
    if family=="ar":
        stride,span = z
        offsets = np.arange(0,span,stride)
        half,shift = [],0
    elif family=="mix":
        offsets = np.arange(z)
        half,shift = (4,8,16,32,64),z
    else:
        offsets = []
        half,shift = g.EMA[:z],0
    ct = model["contract"]
    w = np.asarray(ct["coefficient"])/ct["scale"]
    assert len(w)==len(offsets)+len(half) and len(w)<=16
    alpha = 1.-np.exp2(-1./np.array(half,dtype=float))
    return dict(offset=np.array(offsets,dtype=int) if len(offsets) else [0],nrecent=len(offsets),
                alpha=alpha if len(half) else [0.],nema=len(half),shift=shift,weight=w,
                bias=ct["intercept"]-float(w@ct["mean"]),family=family)


def header(entity,pipe,k,kfit,model):
    stride,chan = channel_map(entity,k["representations"])
    views,counts = [],[]
    ref = None
    for j,pair in enumerate(kfit):
        for v in pair:
            if j==k["reference_index"] and v["view"]=="level":
                ref = len(views)
            record,count = view_config(j,v)
            views.append(record)
            counts.append(count)
    assert ref is not None
    a,w = a_config(model["A"]),w_config(pipe,model["W"])
    c = model["C"] or dict(mean=np.zeros(len(views)),scale=np.ones(len(views)),coef=np.zeros(len(views)),intercept=0.)
    macros = dict(CHANNELS=len(chan),STRIDE=stride,VIEWS=len(views),REFERENCE=ref,
        K_ACTIVE=int(k["routing_status"] in {"ADMITTED","MODEL"}), C_ACTIVE=int(model["C"] is not None),
        W_ACTIVE=int(model["W"] is not None), A_ACTIVE=int(model["A"] is not None),W_SEGMENTS=len(w["poly"]),
        A_RECENT=a["nrecent"],A_EMA=a["nema"],A_EMA_SHIFT=a["shift"])
    text = [f"#define {k} {v}" for k,v in macros.items()]
    text += ["static const Channel channels[]={"+",\n".join(chan)+"};",
             "static const View views[]={"+",\n".join(views)+"};"]

    def declare(name,value,decl="double",suffix=""):
        shape = "[]" if np.asarray(value).ndim else ""
        text.append(f"static const {decl} {name}{shape}{suffix}={c_value(value)};")

    abs_coef = np.abs(c["coef"])
    for name,value in (("c_mean",c["mean"]),("c_inv_scale",1./np.asarray(c["scale"])),("c_coef",c["coef"]),
                       ("c_abs_coef",abs_coef),("c_sum_abs",float(abs_coef.sum())),("c_intercept",c["intercept"]),
                       ("w_mean",w["mean"]),("w_inv_scale",w["inv_scale"]),("w_edges",w["edges"]),
                       ("w_inv_span",1./np.diff(w["edges"])),("a_bias",a["bias"]),("a_alpha",a["alpha"]),
                       ("a_weight",a["weight"])):
        declare(name,value)
    declare("w_poly",w["poly"],suffix="[5]")
    declare("a_offsets",a["offset"],decl="size_t")
    return "\n".join(text)+"\n",dict(K_views=len(views),K_bins=sum(v["bins"] for v in counts),
                   K_basis_products=sum(v["basis_products"] for v in counts),W_active=bool(macros["W_ACTIVE"]),
                   A_family=a["family"],A_dimension=a["nrecent"]+a["nema"],W_poly_max_abs=w["error"])


def build(directory,content):
    directory.mkdir()
    (directory/"model.h").write_text(content,encoding="ascii")
    target = directory/"v07.so"
    subprocess.run(["gcc",*FLAGS,"-I",str(directory),str(ROOT/"stream.c"),"-o",str(target)],check=True)
    library = ctypes.CDLL(str(target))
    fn = library.replay
    fn.argtypes = (F64,ctypes.c_size_t,ctypes.c_size_t,F64)
    fn.restype = ctypes.c_double
    return fn


def archive_timing():
    sys.path.insert(0,str(DIAG))
    spec = importlib.util.spec_from_file_location("archive_native_timing",DIAG/"timing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark(entity,scratch,archive):
    saved = frozen(entity)
    geo,pipe,data,_,stages = inputs(entity,development=True)
    start,stop = geo["development"]
    train_start,train_stop = geo["calibration"]
    t = rows(start,stop)
    k = stages["K"]
    ks = saved["K"]
    dev = g.context(entity,pipe,data,k,ks,stop)
    eng = Engine(pipe,dev,train_start)
    main = saved["chain"]
    assert main["issue_from"]==start
    # This timing-only fit uses the same upstream and a calibration-only mixed A.
    _,fit_pipe,training,_,fit_stages = inputs(entity)
    fit_ctx = g.context(entity,fit_pipe,training,fit_stages["K"],ks,train_stop)
    fit_engine = Engine(fit_pipe,fit_ctx,train_start)
    c,w,mix = fit_engine.a_select(train_stop,"mix")
    assert h.json.dumps(c)==h.json.dumps(main["C"]) and h.json.dumps(w)==h.json.dumps(main["W"])
    assert mix is not None and mix["spec"][0][0]=="mix"
    extra = dict(main,A=mix)
    funcs,refs,config = {},{},{}
    for label,model in (("v07",main),("v07_forced_mixed_timing",extra)):
        text,config[label] = header(entity,pipe,k,ks,model)
        funcs[label] = build(scratch/(entity+"_"+label),text)
        refs[label] = eng.predict(t,model)["KCWA"]
    old_data = archive.prepare(entity)
    old_cfg = archive.fit_parameters(entity,old_data[4],old_data[1])
    old = archive.build(old_cfg,scratch/(entity+"_archive"))
    funcs["v06"] = lambda x,i,n,out:old.stream(0,x,i,n,out)
    funcs["ARX"] = lambda x,i,n,out:old.stream(2,x,i,n,out)
    src = archive.old.path(entity,"predictions_private.npz")
    lin = archive.old.ROOT/"baseline_replay"/entity/"linear_evaluation.npz"
    with np.load(src) as ref, np.load(lin) as baseline:
        np.testing.assert_array_equal(ref["target_index"],t)
        np.testing.assert_array_equal(baseline["target_index"],t)
        refs["v06"] = ref["v06_kcwa"].copy()
        refs["ARX"] = baseline["ARX"].copy()
    # Exercise native-generation W polynomial conversion against an admitted archived W.
    archived_w = stages["W"]
    if archived_w.get("routing_status")=="ADMITTED":
        ct = archived_w["contract"]
        b = ct["basis"]
        limits = b["mean"]+b["scale"]*np.array([b["train_min"],b["train_max"]])
        config["archived_W_polynomial_max_abs"] = w_config(pipe,dict(limits=limits,contract=ct))["error"]
    raw = np.ascontiguousarray(data)
    times = {key:[] for key in funcs}
    errors = {}
    out = np.empty(len(t))
    cut = int(t[len(t)//2])
    alt = np.ascontiguousarray(poisoned(raw,cut))
    for label,fn in funcs.items():
        fn(raw,start,len(t),out)
        np.testing.assert_allclose(out,refs[label],atol=5e-10,rtol=0.)
        errors[label]=float(np.max(np.abs(out-refs[label])))
        n = cut-start+1
        fn(alt,start,n,out)
        np.testing.assert_allclose(out[:n],refs[label][:n],atol=5e-10,rtol=0.)
    rng = np.random.default_rng(20261125)
    for _ in range(REPEATS):
        for label in rng.permutation(list(funcs)):
            times[label].append(funcs[label](raw,start,len(t),out))
    metrics = {key:dict(median_ns=float(np.median(v)),p95_ns=float(np.quantile(v,.95)),
                        runs_ns=v,replay_max_abs=errors[key]) for key,v in times.items()}
    print(entity,{k:round(v["median_ns"],2) for k,v in metrics.items()},flush=True)
    return dict(entity=entity,N=len(t),config=config,timing=metrics,causal_cut=cut)


def run():
    available = os.sched_getaffinity(0)
    core = 12 if 12 in available else min(available)
    os.sched_setaffinity(0,{core})
    archive = archive_timing()
    with TemporaryDirectory(prefix="prism07-") as tmp:
        cases = {e:benchmark(e,Path(tmp),archive) for e in CASES}
    hardware = next(line.split(":",1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    compiler = subprocess.check_output(["gcc","--version"],text=True).splitlines()[0]
    write(ROOT/"NATIVE_TIMING.json",dict(precision="FP64",cpu=hardware,core=core,compiler=compiler,flags=FLAGS,
          repeats=REPEATS,source_sha256={name:sha(ROOT/name) for name in ("stream.c","native.py")},
          model_lock=sha(ROOT/"ALL_FROZEN.json"),
          scope="same native real-development replay, after process-prefix preload, excludes compilation and I/O",
          results=cases))


if __name__=="__main__":
    run()
