"""Same-host scalar and AVX2/FMA development comparisons for PRISM and frozen ARX."""

import ctypes
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory

from v08 import CASES, ROOT, g, inputs, np, poisoned, predict, read, rows, sealed, sha, v, write

V071 = ROOT.parent/"PRISM_V071_REVIEW_20261125"
if not V071.exists():
    V071 = ROOT.parent/"prism_v071_review_20261125"
sys.path.insert(0,str(V071))
import native_fast as fast
import native as native


REPEATS = 30
SCHEDULE = ("PRISM08_scalar","ARX_scalar","PRISM08_SIMD","ARX_SIMD","PRISM071_scalar","PRISM071_SIMD")


def build_arx(where,model,flags):
    where.mkdir()
    scaler,ridge=model["scaler"],model["ridge"]
    weights=np.asarray(ridge.coef_)/scaler.scale_
    assert weights.shape==(768,)
    vector=np.r_[weights[:128,None].T,weights[128:].reshape(128,5).T]
    bias=float(ridge.intercept_-weights@scaler.mean_)
    header=f"#define WIDTH 6\n#define DELTA {int(model['target']=='delta')}\n"
    header+="static const double weights[6][128]="+native.c_value(vector)+";\n"
    header+="static const double bias="+native.c_value(bias)+";\n"
    (where/"arx_model.h").write_text(header,encoding="ascii")
    out=where/"arx.so"
    subprocess.run(["gcc",*flags,"-I",str(where),str(ROOT/"arx_mirror.c"),"-o",str(out)],check=True)
    fn=ctypes.CDLL(str(out)).replay
    fn.restype=ctypes.c_double
    fn.argtypes=(native.F64,ctypes.c_size_t,ctypes.c_size_t,native.F64)
    return fn


def v071_model(e):
    root=V071/"candidate"
    p=root/e/"MODEL.json"
    assert sha(p)==read(root/"ALL_FROZEN.json")["models"][e]
    obj=read(p)
    choice=obj["winner"]
    return v.frozen(e) if choice=="v07" else obj["arms"][choice]


def comparison(e,where):
    state=sealed(e)
    current=state["models"][state["chosen"]]
    prev=v071_model(e)
    geo,pipe,data,_,stage=inputs(e,development=True)
    start,stop=geo["development"]
    t=rows(start,stop)
    k=stage["K"]
    raw=np.ascontiguousarray(data)
    out=np.empty(len(t))
    calls={}; expected={}; counts={}
    for key,model in (("PRISM08",current),("PRISM071",prev)):
        scalar,_=native.header(e,pipe,k,model["K"],model["chain"])
        vector,counts[key]=fast.header(e,pipe,k,model["K"],model["chain"])
        calls[key+"_scalar"]=native.build(where/(e+"_"+key+"_scalar"),scalar)
        calls[key+"_SIMD"]=fast.build(where/(e+"_"+key+"_SIMD"),vector)
        expected[key]=predict(e,pipe,k,data,model,t)["KCWA"]
    matched=v.legacy.baseline
    target=matched.ROOT/e/"linear.pkl"
    with target.open("rb") as f:
        arx=pickle.load(f)["ARX"]
    calls["ARX_scalar"]=build_arx(where/(e+"_ARX_scalar"),arx,native.FLAGS)
    calls["ARX_SIMD"]=build_arx(where/(e+"_ARX_SIMD"),arx,fast.FLAGS)
    with np.load(matched.ROOT/e/"linear_evaluation.npz") as reference:
        np.testing.assert_array_equal(reference["target_index"],t)
        expected["ARX"]=reference["ARX"].copy()
    cut=int(t[len(t)//2]); changed=np.ascontiguousarray(poisoned(data,cut))
    max_error={}
    for label,fn in calls.items():
        p=expected[label.rsplit("_",1)[0]]
        fn(raw,start,len(t),out)
        np.testing.assert_allclose(out,p,atol=5e-10,rtol=0.)
        max_error[label]=float(np.max(abs(out-p)))
        n=cut-start+1
        fn(changed,start,n,out)
        np.testing.assert_allclose(out[:n],p[:n],atol=5e-10,rtol=0.)
    elapsed={name:[] for name in calls}
    rng=np.random.default_rng(81125)
    for _ in range(REPEATS):
        for name in rng.permutation(SCHEDULE):
            elapsed[name].append(calls[name](raw,start,len(t),out))
    measured={key:dict(median_ns=float(np.median(vs)),p95_ns=float(np.quantile(vs,.95)),
                      runs_ns=vs,max_abs_prediction_error=max_error[key]) for key,vs in elapsed.items()}
    print(e,{name:round(row["median_ns"],2) for name,row in measured.items()},flush=True)
    return dict(N=len(t),timing=measured,active_model_counts=counts,causal_cut=cut,
                frozen_linear_sha256=sha(target))


def main():
    cores=os.sched_getaffinity(0); core=12 if 12 in cores else min(cores)
    os.sched_setaffinity(0,{core})
    flags=set(Path("/proc/cpuinfo").read_text().split()); assert {"avx2","fma"}<=flags
    cpu=next(x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name"))
    compiler=subprocess.check_output(["gcc","--version"],text=True).splitlines()[0]
    with TemporaryDirectory(prefix="prism08-simd-") as work:
        result={e:comparison(e,Path(work)) for e in CASES}
    source=[ROOT/"simd.py",ROOT/"arx_mirror.c",Path(fast.__file__),fast.ROOT/"stream_fast.c",
            Path(native.__file__),native.ROOT/"stream.c"]
    write(ROOT/"SIMD.json",dict(cpu=cpu,core=core,compiler=compiler,repeats=REPEATS,precision="FP64",
            rows="same real development origins per dataset",warm_prefix_excluded=True,
            scalar_flags=native.FLAGS,simd_flags=fast.FLAGS,
            source_sha256={str(p):sha(p) for p in source},model_lock=sha(ROOT/"MODEL_LOCK.json"),results=result))


if __name__=="__main__":
    main()
