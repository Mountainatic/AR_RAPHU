"""Bound final PRISM replay and shared TIM experiment utilities."""

import copy
import csv
import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SUITE=BASE/"PRISM_INDUSTRIAL_PROTOCOLS_20261125"
sys.path.insert(0,str(SUITE))
import experiment as x
import prism as p

from experiment import METRICS, data, np, read, score, sha, write


MAIN=("CZ_Rod1","SRU_H2S")
RODS=("CZ_Rod1","CZ_Rod2")
HORIZONS=(8,16,32,48,64)
REANCHOR=65536
OLD_LOCK="75a00f985e842f48182ecb2fb7f64bc954b30c84d754735485276da15aff43a2"
OLD_RESULTS="bee2c1822116baaac27e4779c370e76f450c90fbf6373d362e16b404c3d026df"
LOCK=ROOT/"FINAL_CODE_LOCK.json"


def parent(h=None):
    if h in (8,32):
        return BASE/f"PRISM_CZ_CW_DRIFT_H{h}_20261125"
    if h in (48,64):
        return BASE/f"PRISM_CZ_CW_DRIFT_H{h}_D0_20261125"
    return SUITE


def archived(h=None):
    path=parent(h)
    lock=path/("CORE_FROZEN.json" if h in (48,64) else "ALL_FROZEN.json")
    record=read(lock)
    assert record["registration"]==sha(path/"REGISTRATION.json")
    assert all(sha(path/n)==digest for n,digest in record["files"].items())
    return sha(lock)


def model_sources():
    regular=read(SUITE/"REGISTRATION.json")["source_sha256"]
    source={SUITE/name:digest for name,digest in regular.items()}
    final=p.c.ROOT
    source.update({final/name:digest for name,digest in read(final/"REGISTRATION.json")["source_sha256"].items()})
    original=p.v08.ROOT
    receipt=read(original/"RELEASE_LOCK.json")
    source.update({Path(name):digest for name,digest in receipt["source"].items()})
    # The tight native header path is bound by the previously sealed SIMD receipt.
    num=read(original/"SIMD.json")["source_sha256"]
    source.update({Path(name):digest for name,digest in num.items()})
    for path,digest in source.items():
        assert sha(path)==digest, str(path)
    return {str(path.relative_to(BASE)):digest for path,digest in source.items()}


def freeze_code():
    assert archived()==OLD_LOCK and sha(SUITE/"RESULTS.json")==OLD_RESULTS
    value=dict(parent_protocol_lock=OLD_LOCK,original_results_sha256=OLD_RESULTS,
        production_kernel=dict(reanchor_period=REANCHOR,arithmetic="FP64 AVX2/FMA",
          baseline_method="Rebuild from causal raw history and difference warmup; feedback and alarm state preserved"),
        parent_models={str(h):archived(h) for h in HORIZONS},
        sources={"core.py":sha(ROOT/"core.py"),"stream.c":sha(ROOT/"stream.c")},
        inherited=model_sources(),
        interpretation="Final deterministic kernel freeze; underlying fitted PRISM, gate and timing protocol unchanged.")
    x.bind(LOCK,value)
    return sha(LOCK)


def verify_code():
    lock=read(LOCK)
    assert all(sha(ROOT/name)==digest for name,digest in lock["sources"].items())
    assert sha(SUITE/"ALL_FROZEN.json")==lock["parent_protocol_lock"]
    return sha(LOCK)


def load(name,h=None):
    q=x.TASKS[name]
    if h is not None:
        q=replace(q,H=h)
    origin=parent(h)
    record=read(origin/name/"prism/FROZEN.json")
    m=record["models"][record["chosen"]]
    return q,record["reference"],m["K"],m["chain"],record["chosen"]


def configure(q):
    p.s.D=p.V.D=q.mature
    p.s.GAP=q.mature-1
    p.V.L=q.L
    p.V.EMBARGO=q.embargo


def engine(q,a,k,fit,stop,cls=None):
    configure(q)
    chosen=p.c.Engine if cls is None else cls
    return chosen(p.PIPE,p.context(q,a,k,fit,stop),q.calibration[0])


def rows(q):
    t=q.test_origins()
    target=t+q.H
    masks=dict(development=target<q.dev,tail=t>=q.dev)
    if q.name in x.OVERLAP and q.name.startswith("CZ"):
        a,b=x.OVERLAP[q.name]
        masks["tail_nonoverlap_targets"]=masks["tail"]&((target<a)|(target>=b))
    return t,masks


def summarize(q,a,t,out,mask):
    y=a[t+q.H,0][mask]
    anchor=a[t-q.D,0][mask]
    return {name:summary([score(y,v[mask],anchor) for v in preds]) for name,preds in out.items()}


def summary(runs):
    return dict(runs=runs,mean={m:float(np.mean([x[m] for x in runs])) for m in METRICS},
         sd={m:float(np.std([x[m] for x in runs],ddof=1)) if len(runs)>1 else None for m in METRICS})


def csv_table(path, records):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def functions(dll,monitor=True):
    fp=p.native.legacy.F64
    ordinary=dll.replay
    ordinary.argtypes=(fp,ctypes.c_size_t,ctypes.c_size_t,fp)
    ordinary.restype=ctypes.c_double
    cyc=dll.replay_cycle
    cyc.argtypes=(fp,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,fp)
    cyc.restype=ctypes.c_double
    aud=dll.replay_audit if monitor else None
    if aud:
        aud.argtypes=(*ordinary.argtypes,fp)
        aud.restype=ctypes.c_double
    return ordinary,cyc,aud


def compile_kernel(q,k,fit,chain,directory,*,period=REANCHOR,monitor=True):
    directory.mkdir(exist_ok=True)
    with patch.object(p.v08.h,"COLUMNS",q.channels):
        cfg,counts=p.native.header(q.name,p.PIPE,k,fit,chain,monitor=monitor)
    (directory/"model.h").write_text(cfg,encoding="ascii")
    target=directory/"stream.so"
    ring=1<<q.L.bit_length()
    flags=(*p.native.fast.FLAGS,f"-DDELAY={q.mature}",f"-DANCHOR_DELAY={q.D}",
          f"-DHISTORY={q.L}",f"-DPREFIX_RING={ring}",f"-DREANCHOR_PERIOD={period}")
    subprocess.run(["gcc",*flags,"-I",str(directory),str(ROOT/"stream.c"),"-o",str(target)],check=True)
    return (*functions(ctypes.CDLL(str(target)),monitor),counts,flags)


def replay(q,a,k,fits,chain,t,monitor=True):
    with TemporaryDirectory(prefix="prism-tim-") as scratch:
        f,_,audit,*_=compile_kernel(q,k,fits,chain,Path(scratch),monitor=monitor)
        whole=np.arange(q.train,int(t[-1])+1)
        y=np.empty(len(whole))
        if monitor:
            details=np.empty((len(whole),4))
            audit(a,q.train,len(whole),y,details)
            return y[t-q.train],details[t-q.train]
        f(a,q.train,len(whole),y)
        return y[t-q.train],None


def cert_pair(q,a,k,fits,chain,t):
    view=engine(q,a,k,fits,int(t[-1])+1)
    py=p.c.predict(view,t,chain)
    c,diag=replay(q,a,k,fits,chain,t)
    np.testing.assert_allclose(c,py["pred"],rtol=0,atol=5e-10)
    np.testing.assert_allclose(diag,np.column_stack([py[x] for x in ("trust","mean","active","event")]),
                               rtol=0,atol=5e-9)
    return py, float(np.max(np.abs(c-py["pred"])))


def scaled_surface(q,fit,length):
    changed=copy.deepcopy(fit)
    for pair in changed:
        for view in pair:
            old=np.asarray(view["edges"])
            near=int(np.count_nonzero(np.diff(old)==1))
            # Identify the exact *leading* taps; an unusually short last bin
            # is not part of that group.
            near=next((i for i,n in enumerate(np.diff(old)) if n!=1),near)
            stop=length-(q.mature if view["view"]=="delta" else 0)
            edge=np.r_[old[:near+1],near+np.rint((old[near+1:]-near)*(stop-near)/(old[-1]-near)).astype(int)]
            assert edge[-1]==stop and np.all(np.diff(edge)>0)
            view["edges"]=edge.tolist()
    return changed


if __name__=="__main__":
    print(freeze_code(),flush=True)
