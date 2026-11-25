"""Same-run benchmark for a prediction-equivalent AVX2 streaming kernel."""

import ctypes
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from diagnose import CASES, ROOT, V07, g, inputs, np, poisoned, read, rows, sha, v, write
import native as n


FLAGS = (*n.FLAGS,"-mavx2","-mfma","-funroll-loops")
REPEATS = 30


def header(entity,pipe,k,fit,model):
    base,counts = n.header(entity,pipe,k,fit,model)
    records = []
    c = model["C"]
    ref = 2*k["reference_index"]
    c_active = c is not None
    weights = np.asarray(c["coef"] if c_active else np.zeros(2*len(fit)))
    scl = np.asarray(c["scale"] if c_active else np.ones(len(weights)))
    bias_active = c_active and c["intercept"]!=0.
    k_active = k["routing_status"] in {"MODEL","ADMITTED"}
    all_support = bias_active or (c_active and model["W"] is not None and not np.any(weights))
    roles = dict(emitting=0,support_only=0,discarded=0,nonzero_bins=0,linear_emitting=0)
    for j,m in enumerate(view for pair in fit for view in pair):
        theta = np.asarray(m["contract"]["theta"])
        left,right = np.array(m["edges"][:-1]),np.array(m["edges"][1:])
        exact = int(np.sum(right-left==1))
        assert exact in (4,8) and np.array_equal(right[:exact],np.arange(1,exact+1))
        enough = weights[j]!=0. or (j==ref and (k_active or (not c_active and model["W"] is not None)))
        use = enough or all_support
        nonzero = np.where(np.any(theta!=0.,axis=0))[0]
        width = max(1,int(nonzero[-1])+1) if len(nonzero) else 1
        width = width if enough else 0
        roles["emitting" if enough else ("support_only" if use else "discarded")]+=1
        roles["linear_emitting"]+=int(width==1)
        roles["nonzero_bins"]+=int((np.any(theta!=0.,axis=1)).sum()) if enough else 0
        low,high = np.asarray(m["support"])
        spread = np.maximum(high-low,1e-9*(1+np.maximum(abs(low),abs(high))))
        beta = np.zeros((len(left),4))
        beta[:,:theta.shape[1]]=theta
        near_beta = beta[:exact][::-1].T
        far_beta = beta/(right-left)[:,None]
        records.append(n.c_row(exact,width,int(use),low[:exact][::-1],high[:exact][::-1],
                               1./spread[:exact][::-1],near_beta,far_beta))
    total = float(np.abs(weights).sum())
    base += f"#define NEEDS_MAX_TRUST {int(all_support)}\n"
    base += "#define FOR_EACH_VIEW(F) "+" ".join(f"F({j})" for j in range(len(records)))+"\n"
    base += "static const Packed packed[]={"+",\n".join(records)+"};\n"
    base += "static const double c_scaled_coef[]="+n.c_value(weights/scl)+";\n"
    base += "static const double c_inv_sum_abs="+n.c_value(1./total if total else 0.)+";\n"
    return base,dict(counts,**roles,C_intercept_preserves_max_trust=bias_active)


def build(directory,text):
    directory.mkdir()
    (directory/"model.h").write_text(text,encoding="ascii")
    target = directory/"fast.so"
    subprocess.run(["gcc",*FLAGS,"-I",str(directory),str(ROOT/"stream_fast.c"),"-o",str(target)],check=True)
    fun = ctypes.CDLL(str(target)).replay
    fun.restype = ctypes.c_double
    fun.argtypes = (n.F64,ctypes.c_size_t,ctypes.c_size_t,n.F64)
    return fun


def compare(entity,tmp,archive):
    m = v.frozen(entity)
    geo,pipe,data,_,stages = inputs(entity,development=True)
    start,stop = geo["development"]
    t = rows(start,stop)
    k = stages["K"]
    raw = np.ascontiguousarray(data)
    sel = ROOT/"candidate"/entity/"MODEL.json"
    freeze = read(ROOT/"candidate"/"ALL_FROZEN.json")
    assert sha(sel)==freeze["models"][entity]
    choices = read(sel)
    choice = choices["winner"]
    original_header,_ = n.header(entity,pipe,k,m["K"],m["chain"])
    orig = n.build(tmp/(entity+"_scalar"),original_header)
    text,counts = header(entity,pipe,k,m["K"],m["chain"])
    opt = build(tmp/(entity+"_opt"),text)
    _,old_pipe,_,_,older = archive.prepare(entity)
    old_cfg = archive.fit_parameters(entity,older,old_pipe)
    old_native = archive.build(old_cfg,tmp/(entity+"_older"))
    calls = dict(v07_scalar=orig,v07_optimized=opt,
                 v06=lambda a,b,c,d:old_native.stream(0,a,b,c,d),
                 ARX=lambda a,b,c,d:old_native.stream(2,a,b,c,d))
    y = np.empty(len(t))
    expected = np.empty(len(t))
    context = g.context(entity,pipe,data,k,m["K"],stop)
    pred = v.Engine(pipe,context,geo["calibration"][0]).predict(t,m["chain"])["KCWA"]
    orig(raw,start,len(t),expected)
    opt(raw,start,len(t),y)
    np.testing.assert_allclose(y,pred,atol=5e-10,rtol=0)
    np.testing.assert_allclose(y,expected,atol=5e-10,rtol=0)
    maxerr = float(max(np.max(abs(y-expected)),np.max(abs(y-pred))))
    selected_proof = dict(max_abs=maxerr,counts=counts,aliased_to_v07=choice=="v07")
    extra = None
    if choice!="v07":
        extra = choices["arms"][choice]
        ctx = g.context(entity,pipe,data,k,extra["K"],stop)
        extra_pred = v.Engine(pipe,ctx,geo["calibration"][0]).predict(t,extra["chain"])["KCWA"]
        text,chosen_counts = header(entity,pipe,k,extra["K"],extra["chain"])
        extra_opt = build(tmp/(entity+"_candidate_opt"),text)
        raw_header,_ = n.header(entity,pipe,k,extra["K"],extra["chain"])
        extra_ref = n.build(tmp/(entity+"_candidate_scalar"),raw_header)
        calls["v071_optimized"] = extra_opt
        calls["v071_scalar"] = extra_ref
        extra_expected = np.empty(len(t))
        extra_ref(raw,start,len(t),extra_expected)
        extra_opt(raw,start,len(t),y)
        np.testing.assert_allclose(y,extra_pred,atol=5e-10,rtol=0)
        np.testing.assert_allclose(y,extra_expected,atol=5e-10,rtol=0)
        selected_proof = dict(max_abs=float(max(np.max(abs(y-extra_expected)),np.max(abs(y-extra_pred)))),
                              counts=chosen_counts,aliased_to_v07=False)
    cuts = [int(t[37]),int(t[len(t)//2])]
    for cut in cuts:
        evil = np.ascontiguousarray(poisoned(raw,cut))
        length = cut+1-start
        opt(evil,start,length,y)
        np.testing.assert_allclose(y[:length],expected[:length],atol=5e-10,rtol=0)
        if extra is not None:
            extra_opt(evil,start,length,y)
            np.testing.assert_allclose(y[:length],extra_expected[:length],atol=5e-10,rtol=0)
    values = {a:[] for a in calls}
    rng = np.random.default_rng(20261125)
    for _ in range(REPEATS):
        for label in rng.permutation(list(calls)):
            values[label].append(calls[label](raw,start,len(t),y))
    results = {name:dict(median_ns=float(np.median(v)),p95_ns=float(np.quantile(v,.95)),runs=v)
               for name,v in values.items()}
    if choice=="v07":
        results["v071_optimized"] = results["v07_optimized"]
        results["v071_scalar"] = results["v07_scalar"]
    print(entity,counts,{k:round(v["median_ns"],2) for k,v in results.items()},maxerr,flush=True)
    return dict(N=len(t),counts=counts,max_abs=maxerr,cuts=cuts,timing=results,
                selected=choice,selected_proof=selected_proof)


def run():
    cpus = os.sched_getaffinity(0)
    core = 12 if 12 in cpus else min(cpus)
    os.sched_setaffinity(0,{core})
    available = set(Path("/proc/cpuinfo").read_text().split())
    assert {"avx2","fma"}<=available
    base = n.archive_timing()
    with TemporaryDirectory() as work:
        out = {e:compare(e,Path(work),base) for e in CASES}
    cpu = next(x.split(":",1)[1].strip() for x in open("/proc/cpuinfo") if x.startswith("model name"))
    write(ROOT/"NATIVE_OPTIMIZED.json",dict(results=out,repeats=REPEATS,core=core,cpu=cpu,
      flags=FLAGS,reference_flags=n.FLAGS,precision="FP64",
      model_lock=sha(V07/"ALL_FROZEN.json"), candidate_lock=sha(ROOT/"candidate"/"ALL_FROZEN.json"),
      source={p:sha(ROOT/p) for p in ("stream_fast.c","native_fast.py")}))


if __name__=="__main__":
    run()
