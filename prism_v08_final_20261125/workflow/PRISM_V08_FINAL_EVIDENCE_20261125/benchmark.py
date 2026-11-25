"""Fixed-core end-to-end timing and controlled runtime-only L scaling."""

import argparse
import copy
import ctypes
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from tail_replay import CASES, EXPECTED, ROOT, TORCH, V08, elastic_heads, fast, inputs, lin_predict, linear_heads, lock, np, read, rows, sealed, sha, v08, write
from simd import native


REPEATS=30
LENGTHS=(64,128,256,512,1024,2048)
N_SINGLE=192
N_SCALE=8192


def compile_replay(d,header,code,header_name="model.h"):
    d.mkdir()
    (d/header_name).write_text(header,encoding="ascii")
    (d/"replay.c").write_text(code,encoding="utf-8")
    dll=d/"replay.so"
    subprocess.run(["gcc",*fast.FLAGS,"-I",str(d),str(d/"replay.c"),"-o",str(dll)],check=True)
    fn=ctypes.CDLL(str(dll)).replay
    fn.restype=ctypes.c_double
    fn.argtypes=(native.F64,ctypes.c_size_t,ctypes.c_size_t,native.F64)
    return fn


def linear_kernel(path,weights,bias,delta,length):
    weights=np.asarray(weights)
    assert weights.size in (length,6*length)
    stack=weights.reshape(1,length) if weights.size==length else +        np.r_[weights[:length,None].T,weights[length:].reshape(length,5).T]
    h=(f"#define WIDTH {len(stack)}\n#define STRIDE 6\n#define DELTA {int(delta)}\n#define LENGTH {length}\n"
        +"static const double weights[WIDTH][LENGTH]="+native.c_value(stack)+";\n"
        +"static const double bias="+native.c_value(float(bias))+";\n")
    code=(V08/"arx_mirror.c").read_text().replace("L=128","L=LENGTH").replace("*WIDTH","*STRIDE")
    return compile_replay(path,h,code,header_name="arx_model.h")


def linear_params(head):
    estimator,scaler=head["ridge"],head["scaler"]
    weights=np.asarray(estimator.coef_)/scaler.scale_
    return weights,float(estimator.intercept_-weights@scaler.mean_),head["target"]=="delta"


def elastic_params(head):
    estimator,center,scale=head["model"]
    scaler=head["scaler"]
    weights=np.asarray(estimator.coef_)*scale/scaler.scale_
    return weights,float(estimator.intercept_*scale+center-weights@scaler.mean_),head["recipe"]=="delta"


def sparse_kernel(where,weights,bias,delta):
    matrix=np.r_[weights[:128,None].T,weights[128:].reshape(128,5).T]
    channel,lag=np.nonzero(matrix)
    values=matrix[channel,lag]
    n=len(values); padded=max(8,8*((n+7)//8))
    indexes=np.pad(channel*256+lag,(0,padded-n),constant_values=int(channel[0]*256+lag[0]) if n else 0)
    mask=np.any(matrix,axis=1).astype(int)
    header=f"#define NONZERO {n}\n#define PADDED {padded}\n#define DELTA {int(delta)}\n"
    header+="static const int active[6]="+native.c_value(mask)+";\n"
    header+="static const int64_t index[PADDED]="+native.c_value(indexes)+";\n"
    header+="static const double weight[PADDED]="+native.c_value(np.pad(values,(0,padded-n)))+";\n"
    header+="static const double bias="+native.c_value(float(bias))+";\n"
    return compile_replay(where,header,(ROOT/"elastic_sparse.c").read_text(),header_name="sparse_model.h")


def measure(calls,data,start,count,expected=None):
    out=np.empty(count)
    values={name:[] for name in calls}
    for name,fn in calls.items():
        fn(data,start,count,out)
        assert np.isfinite(out).all()
        if expected is not None:
            np.testing.assert_allclose(out,expected[name],rtol=0,atol=5e-10)
    rng=np.random.default_rng(20261125)
    for _ in range(REPEATS):
        for label in rng.permutation(list(calls)):
            values[label].append(calls[label](data,start,count,out))
    return {name:dict(median_ns=float(np.median(seq)),p95_ns=float(np.quantile(seq,.95)),runs_ns=seq)
            for name,seq in values.items()}


def data_from_frozen(e):
    state=sealed(e)
    model=state["models"][state["chosen"]]
    geo,pipe,data,_,stage=inputs(e,development=True)
    k=stage["K"]
    return model,geo,pipe,np.ascontiguousarray(data),k


def pareto(e,reg,work,prior_torch=None):
    model,geo,pipe,data,k=data_from_frozen(e)
    start=geo["development"][0]
    t=rows(start,start+N_SINGLE)
    binding=reg["frozen_baselines"][e]
    heads=linear_heads(binding); elastic=elastic_heads(binding)
    linear=lin_predict(heads,elastic,data,t)
    h,cfg=fast.header(e,pipe,k,model["K"],model["chain"])
    calls={"PRISM_v08":fast.build(work/"prism",h)}
    expected={"PRISM_v08":v08.predict(e,pipe,k,data,model,t)["KCWA"]}
    for family in ("AR","ARX"):
        calls[family]=linear_kernel(work/family,*linear_params(heads[family]),128)
        expected[family]=linear[family][0]
    nonzero={}
    for i,estimator in enumerate(elastic):
        name=f"ElasticNet_{i}"
        params=elastic_params(estimator)
        nonzero[name]=int(np.count_nonzero(params[0]))
        calls[name]=linear_kernel(work/name,*params,128)
        expected[name]=linear["ElasticNet"][i]
        compact=f"ElasticNetSparse_{i}"
        calls[compact]=sparse_kernel(work/compact,*params)
        expected[compact]=expected[name]
    times=measure(calls,data,start,len(t),expected)
    if prior_torch is None:
        p=work/"timing_data.npz"; np.savez(p,data=data,issues=t)
        manifest=work/"neural.json"; write(manifest,binding["transformer"])
        dst=work/"torch_timing.json"
        subprocess.run([str(TORCH),str(ROOT/"neural_replay.py"),str(p),str(manifest),str(dst),
                        "--timing","--core",str(min(os.sched_getaffinity(0)))],check=True)
        prior_torch=read(dst)
    assert prior_torch["core"]==min(os.sched_getaffinity(0))
    return dict(N=len(t),rows=[int(t[0]),int(t[-1])+1],model=cfg,times=times,
                elastic_nonzero=nonzero,torch=prior_torch,core=min(os.sched_getaffinity(0)))


def expanded_windows(model,length):
    tree=copy.deepcopy(model["K"])
    for pair in tree:
        for view in pair:
            old=np.asarray(view["edges"])
            sizes=np.diff(old)
            near=int((sizes==1).sum())
            assert near in (4,8)
            finish=length-(4 if view["view"]=="delta" else 0)
            edges=np.r_[old[:near+1],near+np.rint((old[near+1:]-near)*(finish-near)/(old[-1]-near)).astype(int)]
            assert edges[-1]==finish and (np.diff(edges)>0).all() and len(old)==len(edges)
            view["edges"]=edges.tolist()
    return tree


def scaling(e,reg,work):
    model,geo,pipe,original,k=data_from_frozen(e)
    a,b=geo["development"]
    src=original[a:b]
    warm=max(LENGTHS)+4
    data=np.ascontiguousarray(np.tile(src,(int(np.ceil((warm+N_SCALE)/len(src))),1))[:warm+N_SCALE])
    rng=np.random.default_rng(20260401)
    max_own=rng.normal(size=max(LENGTHS))
    max_x=rng.normal(size=(max(LENGTHS),5))
    fast_source=(fast.ROOT/"stream_fast.c").read_text()
    results=[]
    for width in LENGTHS:
        tree=expanded_windows(model,width)
        header,counts=fast.header(e,pipe,k,tree,model["chain"])
        address=2*width
        code=fast_source.replace("PREFIX_RING=256",f"PREFIX_RING={address}")
        assert f"PREFIX_RING={address}" in code
        a0=compile_replay(work/f"PRISM_{width}",header,code)
        if width==128:
            weights,bias,delta=linear_params(linear_heads(reg["frozen_baselines"][e])["ARX"])
        else:
            weights=np.r_[max_own[:width],max_x[:width].ravel()]*.01/np.sqrt(width)
            bias,delta=0.,True
        b0=linear_kernel(work/f"ARX_{width}",weights,bias,delta,width)
        times=measure({"PRISM":a0,"ARX":b0},data,warm,N_SCALE)
        results.append(dict(L=width,N=N_SCALE,bins=counts["K_bins"],times=times,
              prism_prefix_bytes=counts["K_views"]*address*5*8,arx_ring_bytes=6*(2*width)*8))
        print("SCALE",e,width,{k:round(x["median_ns"],2) for k,x in times.items()},flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("pareto","scaling"))
    parser.add_argument("--reuse-torch",action="store_true")
    args=parser.parse_args()
    reg=lock()
    previous=read(V08/"SIMD.json")
    core=previous["core"]
    os.sched_setaffinity(0,{core})
    compiler=subprocess.check_output(["gcc","--version"],text=True).splitlines()[0]
    assert compiler==previous["compiler"]
    results={}
    prior_timing=None
    if args.reuse_torch:
        assert args.mode=="pareto"
        prior_timing=read(ROOT/"BENCH_PARETO.json")
        assert prior_timing["core"]==core and prior_timing["release"]==EXPECTED
    with TemporaryDirectory(prefix="prism08-bench-") as d:
        for e in CASES:
            work=Path(d)/e; work.mkdir()
            if args.mode=="pareto":
                existing=prior_timing["series"][e] if prior_timing else None
                if existing is not None:
                    assert existing["N"]==N_SINGLE
                results[e]=pareto(e,reg,work,existing["torch"] if existing else None)
                if existing is not None:
                    assert results[e]["rows"]==existing["rows"]
            else:
                results[e]=scaling(e,reg,work)
            print("BENCH_COMPLETE",args.mode,e,flush=True)
    write(ROOT/f"BENCH_{args.mode.upper()}.json",dict(release=EXPECTED,core=core,cpu=previous["cpu"],
          compiler=compiler,flags=fast.FLAGS,repeats=REPEATS,
          source_sha256={p.name:sha(p) for p in (Path(__file__),fast.ROOT/"stream_fast.c",V08/"arx_mirror.c",
                                                ROOT/"neural_replay.py",ROOT/"elastic_sparse.c")},
          timing="C monotonic raw streaming loop; Transformer CPU batch 1 with feature assembly and scalar output",
          series=results))


if __name__=="__main__":
    main()
