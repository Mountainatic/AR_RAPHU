"""Same-core native replay timings and long-stream FP64 comparison."""

from pathlib import Path
from tempfile import TemporaryDirectory
import ctypes
import os
import pickle
import subprocess

import core as c


np=c.np
ROOT=c.ROOT/"E4"
FLAGS=c.p.native.fast.FLAGS
CORE=64
REPEATS=35
LENGTHS=(64,128,256,512,1024,2048)
LONG=10_000_000
WINDOW=8192
ROWS=2048


def hx(z):
    return float(z).hex()


def line(items):
    return "{"+",".join(items)+"}"


def unpack(q,r,kind):
    model,mean,sd=r["model"]
    scaler=r["scale"]
    w=model.coef_*sd/scaler.scale_
    b=float(model.intercept_*sd+mean-w@scaler.mean_)
    matrix=w[:q.L][None,:]
    if kind!="AR":
        matrix=np.r_[matrix,w[q.L:].reshape(q.L,len(q.channels)).T]
    return np.ascontiguousarray(matrix),b,r["recipe"]=="delta"


def linear(q,work,w,b,delta,sparse=False):
    work.mkdir()
    sizes=f"#define LENGTH {q.L}\n#define ANCHOR_DELAY {q.D}\n#define STRIDE {len(q.channels)+1}\n"
    constants=sizes+f"#define DELTA {int(delta)}\nstatic const double bias={hx(b)};\n"
    if sparse:
        used=np.argwhere(w!=0.)
        count=len(used)
        ids=[int(i*2*q.L+j) for i,j in used]
        values=[w[i,j] for i,j in used]
        active=w.any(1).astype(int)
        if not count:
            ids,values=[0],[0.]
        pad=max(8,(len(ids)+7)//8*8)-len(ids)
        # Pad from the first initialized address, even if the target channel is inactive.
        ids += [ids[0]]*pad
        values += [0.]*pad
        header=constants+f"#define NONZERO {count}\n#define PADDED {len(ids)}\n"
        header+="static const int active[STRIDE]="+line(map(str,active))+";\n"
        header+="static const long long index[PADDED]="+line(map(str,ids))+";\n"
        header+="static const double weight[PADDED]="+line(map(hx,values))+";\n"
        source,include="elastic_sparse.c","sparse_model.h"
    else:
        header=constants+f"#define WIDTH {len(w)}\n"
        header+="static const double weights[WIDTH][LENGTH]="+line(line(map(hx,row)) for row in w)+";\n"
        source,include="arx_mirror.c","arx_model.h"
    (work/include).write_text(header,encoding="ascii")
    shared=work/"linear.so"
    subprocess.run(["gcc",*FLAGS,"-I",str(work),str(c.ROOT/source),"-o",str(shared)],check=True)
    f=ctypes.CDLL(str(shared)).replay
    fp=c.p.native.legacy.F64
    f.argtypes=(fp,ctypes.c_size_t,ctypes.c_size_t,fp)
    f.restype=ctypes.c_double
    return f


def measure(arms,a,start,n):
    rng=np.random.default_rng(14165)
    trace={key:[] for key in arms}
    out=np.empty(n)
    predictions={}
    for name,f in arms.items():
        f(a,start,n,out)
        predictions[name]=out.copy()
    for _ in range(REPEATS):
        for key in rng.permutation(list(arms)):
            trace[key].append(arms[key](a,start,n,out))
    return {name:dict(median_ns=float(np.median(t)),p10_ns=float(np.quantile(t,.1)),
                p90_ns=float(np.quantile(t,.9)),repeats=len(t),individual_ns=t)
            for name,t in trace.items()},predictions


def matched(q,k,fit,chain,a,work,models):
    f,_,_,counts,_=c.compile_kernel(q,k,fit,chain,work/"prism")
    arms={"PRISM":f}
    details={}
    for method in ("AR","ARX","ElasticNet"):
        run=models[method][0]
        w,b,delta=unpack(q,run,method)
        details[method]=dict(terms=int(w.size),nonzero=int(np.count_nonzero(w)))
        arms[method]=linear(q,work/method,w,b,delta)
        if method=="ElasticNet":
            arms["ElasticNet_sparse"]=linear(q,work/"elastic_sparse",w,b,delta,sparse=True)
    start=q.train
    out,predictions=measure(arms,a,start,ROWS)
    native_error={}
    with np.load(c.SUITE/q.name/"linear/PREDICTIONS.npz") as m:
        assert m["origins"][0]==start
        for method in arms:
            if method=="PRISM":
                continue
            src="ElasticNet_0" if method.startswith("ElasticNet") else method
            wanted=m[src][:ROWS]
            native_error[method]=float(np.max(abs(predictions[method]-wanted)))
            np.testing.assert_allclose(predictions[method],wanted,rtol=0,atol=5e-10)
    with np.load(c.SUITE/q.name/"prism/PREDICTIONS.npz") as m:
        wanted=m["PRISM"][:ROWS]
        native_error["PRISM"]=float(np.max(abs(predictions["PRISM"]-wanted)))
        np.testing.assert_allclose(predictions["PRISM"],wanted,rtol=0,atol=5e-10)
    fastest=min(("ElasticNet","ElasticNet_sparse"),key=lambda method:out[method]["median_ns"])
    return dict(results=out,dense_or_sparse_fastest=fastest,parity=native_error,details=details,PRISM_counts=counts)


def scaling(q,k,fit,chain,a,work):
    start=c.REANCHOR-WINDOW//2
    data=np.ascontiguousarray(np.tile(a[:4096],((start+WINDOW)//4096+1,1)))
    rng=np.random.default_rng(20261125)
    w=rng.normal(0.,.01,size=(len(q.channels)+1,max(LENGTHS)))
    results=[]
    for L in LENGTHS:
        sub=work/f"L_{L}"
        sub.mkdir()
        protocol=c.replace(q,L=L)
        template=fit if q.L==L else c.scaled_surface(q,fit,L)
        f,_,_,counts,_=c.compile_kernel(protocol,k,template,chain,sub/"prism")
        g=linear(protocol,sub/"arx",w[:,:L],0.,False)
        info,_=measure(dict(PRISM=f,ARX=g),data,start,WINDOW)
        results.append(dict(L=L,results=info,counts=counts))
        print("E4",q.name,"scale",L,{m:v["median_ns"] for m,v in info.items()},flush=True)
    return results


def stable(q,k,fit,chain,a,work):
    start=q.train
    data=np.ascontiguousarray(a[:4096])
    outputs={}
    timing={}
    for period in (256,c.REANCHOR,0):
        _,f,*_=c.compile_kernel(q,k,fit,chain,work/f"anchor_{period}",period=period)
        pred=np.empty(LONG,dtype=np.float64)
        timing[str(period)]=f(data,len(data),start,LONG,pred)
        assert np.isfinite(pred).all()
        outputs[period]=pred
    reference=outputs[256]
    audit={}
    for period in (c.REANCHOR,0):
        differences=np.abs(outputs[period]-reference)
        idx=int(differences.argmax())
        audit[str(period)]=dict(max_abs=float(differences[idx]),at_origin=start+idx,
            RMS=float(np.sqrt((differences@differences)/len(differences))),
            segments_max=[float(v.max()) for v in np.array_split(differences,10)])
    assert audit[str(c.REANCHOR)]["max_abs"]<=5e-8
    return dict(rows=LONG,cyclic_block=len(data),reference_reanchor_steps=256,
        measured_reanchor_events=(start+LONG-1)//c.REANCHOR-(start-1)//c.REANCHOR,
        tolerance=5e-8,error=audit,ns=timing)


def benchmark():
    os.sched_setaffinity(0,{CORE})
    compiler=subprocess.check_output(["gcc","--version"],text=True).splitlines()[0]
    cpu=next(line.strip().split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name"))
    c.x.bind(ROOT/"REGISTRATION.json",dict(code_lock=c.verify_code(),source_sha256={
        name:c.sha(c.ROOT/name) for name in ("e4.py","arx_mirror.c","elastic_sparse.c")},clock="CLOCK_MONOTONIC_RAW",
        core=CORE,cpu=cpu,compiler=compiler,flags=FLAGS,paired_randomized_repeats=REPEATS,
        matched_rows=ROWS,scaling_rows=WINDOW,L=LENGTHS,long_rows=LONG,
        scaling="Fixed selected PRISM box count and extended far widths; synthetic fixed-seed dense ARX; one reanchor within the timed window",
        stability="Cyclic observed training block, compare to same kernel reanchored every 256 steps"))
    all_rows={}; paired=[]; lengths=[]; audits=[]
    for name in c.MAIN:
        q,k,fit,chain,_=c.load(name)
        a=c.data(q,full=True)
        model=pickle.loads((c.SUITE/name/"linear/MODELS.pkl").read_bytes())
        with TemporaryDirectory(prefix="prism-tim-bench-") as temp:
            work=Path(temp)
            exact=matched(q,k,fit,chain,a,work,model)
            extension=scaling(q,k,fit,chain,a,work)
            long=stable(q,k,fit,chain,a,work)
        all_rows[name]=dict(matched=exact,extension=extension,stability=long)
        for m,v in exact["results"].items():
            paired.append(dict(data=name,H=q.H,D=q.D,L=q.L,method=m,
                **{key:val for key,val in v.items() if key!="individual_ns"},
                max_abs_vs_python=exact["parity"][m]))
        for row in extension:
            for method,r in row["results"].items():
                lengths.append(dict(data=name,L=row["L"],method=method,
                    **{key:val for key,val in r.items() if key!="individual_ns"}))
        for period,errors in long["error"].items():
            audits.append(dict(data=name,steps=long["rows"],reanchor_period=period,
                reference_period=long["reference_reanchor_steps"],ns_per_row=long["ns"][period],
                **{k:v for k,v in errors.items() if k!="segments_max"}))
        print("E4",name,"10^7",long["error"],flush=True)
    c.csv_table(ROOT/"NATIVE_LATENCY.csv",paired)
    c.csv_table(ROOT/"WINDOW_SCALING.csv",lengths)
    c.csv_table(ROOT/"LONG_STABILITY.csv",audits)
    c.write(ROOT/"EVIDENCE.json",all_rows)


if __name__=="__main__":
    benchmark()
