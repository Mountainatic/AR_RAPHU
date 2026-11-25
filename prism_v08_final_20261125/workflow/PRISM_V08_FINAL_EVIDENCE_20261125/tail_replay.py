"""One-time chronological tail replay using the explicit PRISM v0.8 release lock."""

import argparse
import csv
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parent
V08=ROOT.parent/"PRISM_V08_20261125"
sys.path.insert(0,str(V08))
import v08
from v08 import CASES, Engine, g, inputs, np, poisoned, read, rows, sealed, sha, write
from simd import fast
import joblib


EXPECTED="4403fe0c314b0d5f4fe773c2f4a34ef5d17914db030f09efb657d86eafbc1a53"
PRIVATE=ROOT/"private"
TORCH=Path("/root/miniconda3/bin/python")
D,L=4,128
METRICS=("RMSE","MAE","MSE","R2_y","R2_d","eta_pers_pct")
LAYERS=("K","C","W","A")


def digest(path,digest_value):
    if sha(Path(path))!=digest_value:
        raise ValueError(f"Frozen hash mismatch: {path}")


def lock():
    path=V08/"RELEASE_LOCK.json"
    digest(path,EXPECTED)
    data=read(path)
    for name,value in data["source"].items():
        digest(name,value)
    for name,value in data["development"].items():
        digest(name,value)
    digest(V08/"MODEL_LOCK.json",data["model_lock"])
    digest(V08/"REGISTRATION.json",data["formal_registration"])
    digest(V08/"SIMD.json",data["simd"])
    for name,value in read(V08/"SIMD.json")["source_sha256"].items():
        digest(name,value)
    for e in CASES:
        sealed(e)
        for cls in ("linear","elastic_net","transformer"):
            for name,value in data["frozen_baselines"][e][cls].items():
                digest(name,value)
    return data


def srutext(path):
    found=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields=line.split()
            if len(fields)!=7:
                continue
            try:
                q=[float(t) for t in fields]
            except ValueError:
                continue
            if not found and q==list(range(1,8)):
                continue
            found.append(q)
    return np.asarray(found,dtype=np.float64)[:,[5,0,1,2,3,4]]


def exclusive(path,obj):
    from json import dump
    with path.open("x",encoding="utf-8") as fd:
        dump(obj,fd,indent=2)
        fd.write("\n")
        fd.flush()
        os.fsync(fd.fileno())


def ingest(reg):
    """Materialize each locked source exactly once, preserving warm replay prefix."""
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    manifest=PRIVATE/"SOURCES.json"
    if manifest.exists():
        bound=read(manifest)
        assert bound["release"]==EXPECTED
        for e,d in bound["arrays"].items():
            digest(PRIVATE/f"{e}.npy",d)
        return
    exclusive(PRIVATE/"ACCESS_STARTED.json",dict(release=EXPECTED,entities=list(CASES),
              rows={e:reg["tail_metadata"][e]["rows"] for e in CASES}))
    import pandas as pd
    meta=reg["tail_metadata"]
    cfg=meta["SRU_H2S"]
    source=Path(cfg["path"])
    digest(source,cfg["archived_source_sha256"])
    arrays={"SRU_H2S":srutext(source)}
    cz=meta["CZ_Rod1"]
    source=Path(cz["path"])
    assert cz["path"]==meta["CZ_Rod2"]["path"]
    digest(source,cz["archived_source_sha256"])
    columns=["晶体直径","主加热功率","晶升速度","埚升速度","晶转速度","埚转速度"]
    with pd.ExcelFile(source,engine="openpyxl") as excel:
        assert excel.sheet_names==["Sheet1","Sheet2"]
        for i,e in enumerate(("CZ_Rod1","CZ_Rod2")):
            df=pd.read_excel(excel,sheet_name=i,usecols=columns)
            arrays[e]=df[columns].apply(pd.to_numeric,errors="raise").to_numpy(np.float64)
    receipts={}
    for e,a in arrays.items():
        lo,hi=meta[e]["rows"]
        _,_,prefix,_,_=inputs(e,development=True)
        assert a.shape==(hi,6) and np.isfinite(a).all() and len(prefix)==lo
        np.testing.assert_array_equal(a[:lo],prefix)
        p=PRIVATE/f"{e}.npy"
        np.save(p,a)
        receipts[e]=sha(p)
    exclusive(manifest,dict(release=EXPECTED,arrays=receipts,prefix_exact=True))


def features(data,t):
    j=t[:,None]-np.arange(L)
    return np.c_[data[j-D,0],data[j,1:].reshape(len(t),-1)]


def linear_heads(binding):
    return pickle.loads(Path(next(k for k in binding["linear"] if k.endswith(".pkl"))).read_bytes())


def elastic_heads(binding):
    paths=sorted(k for k in binding["elastic_net"] if k.endswith(".joblib"))
    assert len(paths)==5
    return [joblib.load(path) for path in paths]


def lin_predict(heads,elastic,data,t):
    anchor=data[t-D,0]
    out={"Persistence":[anchor]}
    x=features(data,t)
    for kind,head in heads.items():
        z=x[:,:L] if kind=="AR" else x
        p=head["ridge"].predict(head["scaler"].transform(z))
        out[kind]=[p+(anchor if head["target"]=="delta" else 0.)]
    out["ElasticNet"]=[]
    for seed,item in enumerate(elastic):
        assert item["seed"]==seed
        m,ym,ys=item["model"]
        p=np.asarray(m.predict(item["scaler"].transform(x)),dtype=np.float64)*ys+ym
        out["ElasticNet"].append(p+(anchor if item["recipe"]=="delta" else 0.))
    return out


def score(y,p,anchor):
    residual=y-p
    d=y-anchor
    mse=float(np.mean(residual**2))
    return dict(RMSE=float(np.sqrt(mse)),MAE=float(np.mean(np.abs(residual))),MSE=mse,
                R2_y=1.-mse/float(np.var(y)),R2_d=1.-mse/float(np.var(d)),
                eta_pers_pct=100.*(1.-mse/float(np.mean(d*d))))


def summarize(y,a,preds,mask):
    result={}
    for kind,draws in preds.items():
        values=[score(y[mask],p[mask],a[mask]) for p in draws]
        result[kind]=dict(runs=values,mean={m:float(np.mean([v[m] for v in values])) for m in METRICS},
             sd={m:float(np.std([v[m] for v in values],ddof=1)) if len(values)>1 else None for m in METRICS})
    return result


def neural_child(binding,data,t,cut,work):
    p=work/"features.npz"
    np.savez(p,data=data,issues=t,cut=np.int64(cut))
    manifest=work/"neural.json"
    write(manifest,binding["transformer"])
    dst=work/"neural_results.npz"
    subprocess.run([str(TORCH),str(ROOT/"neural_replay.py"),str(p),str(manifest),str(dst)],check=True)
    with np.load(dst) as f:
        predictions=[f[f"seed_{i}"].copy() for i in range(5)]
    return predictions,read(dst.with_suffix(".json"))


def evaluate(e,reg):
    path=ROOT/e/"TAIL.json"
    if path.exists():
        assert read(path)["release"]==EXPECTED
        return
    lo,hi=reg["tail_metadata"][e]["rows"]
    t=rows(lo,hi)
    cut=int(t[len(t)//2])
    data=np.load(PRIVATE/f"{e}.npy")
    evil=np.ascontiguousarray(poisoned(data,cut))
    short=t[t<=cut]
    y=data[t,0]
    a=data[t-D,0]
    geo,pipe,_,rebind,stages=inputs(e,development=True)
    k=stages["K"]
    state=sealed(e)
    model=state["models"][state["chosen"]]
    chain=model["chain"]
    issue=chain["issue_from"]
    assert issue==geo["development"][0] and issue<lo
    pipe.accessor=rebind(data)
    ctx=g.context(e,pipe,data,k,model["K"],hi)
    engine=Engine(pipe,ctx,geo["calibration"][0])
    component=engine.components(t,chain)
    pred=engine.predict(t,chain)
    forecast=pred["KCWA"]
    delta=component.sum(1)
    additive_max=float(np.max(np.abs(a+delta-forecast)))
    attr=component*(2.*(y-a)-delta)[:,None]
    telescope=(y-a)**2-(y-forecast)**2
    attribution_max=float(np.max(np.abs(attr.sum(1)-telescope)))
    np.testing.assert_allclose(a+delta,forecast,atol=5e-11,rtol=0)
    np.testing.assert_allclose(attr.sum(1),telescope,atol=5e-11,rtol=0)
    pipe.accessor=rebind(evil)
    altered=v08.predict(e,pipe,k,evil,model,short)
    python_error=max(float(np.max(abs(q-pred[m][:len(q)]))) for m,q in altered.items())
    assert python_error<5e-11
    pipe.accessor=rebind(data)
    binding=reg["frozen_baselines"][e]
    heads=linear_heads(binding); elastic=elastic_heads(binding)
    out=lin_predict(heads,elastic,data,t)
    biased=lin_predict(heads,elastic,evil,short)
    linear_error={}
    for name,draws in out.items():
        errors=[]
        for p,q in zip(draws,biased[name],strict=True):
            np.testing.assert_allclose(q,p[:len(q)],atol=5e-11,rtol=0)
            errors.append(float(np.max(abs(q-p[:len(q)]))))
        linear_error[name]=errors
    with TemporaryDirectory(prefix=e+"-",dir=PRIVATE) as d:
        work=Path(d)
        header,counts=fast.header(e,pipe,k,model["K"],chain)
        native=fast.build(work/"native",header)
        result=np.empty(hi-issue)
        native(np.ascontiguousarray(data),issue,len(result),result)
        np.testing.assert_allclose(result[lo-issue:],forecast,atol=5e-10,rtol=0)
        native_error=float(np.max(np.abs(result[lo-issue:]-forecast)))
        partial=np.empty(cut+1-issue)
        native(evil,issue,len(partial),partial)
        np.testing.assert_allclose(partial,result[:len(partial)],atol=5e-10,rtol=0)
        native_poison=float(np.max(np.abs(partial-result[:len(partial)])))
        out["Transformer_expanded"], neural_audit=neural_child(binding,data,t,cut,work)
    out["PRISM_v08"]=[forecast]
    historical=reg["tail_metadata"][e]["historical_target_scoring"]["prior_scored_range"]
    masks={"full_tail":np.full(len(t),True)}
    if e.startswith("CZ_"):
        masks["nonoverlap_origins"]=(t<historical[0])|(t>=historical[1])
        assert int((~masks["nonoverlap_origins"]).sum())==historical[1]-historical[0]
    activation=dict(zip(LAYERS,[bool(k["routing_status"] in {"ADMITTED","MODEL"}),
                                 chain["C"] is not None,chain["W"] is not None,chain["A"] is not None]))
    subsets={}
    for name,mask in masks.items():
        subsets[name]=dict(N=int(mask.sum()),scores=summarize(y,a,out,mask),
            mean_attribution={m:float(q) for m,q in zip(LAYERS,attr[mask].mean(0),strict=True)},
            stage_metrics={m:score(y[mask],q[mask],a[mask]) for m,q in pred.items()})
    np.savez_compressed(PRIVATE/f"{e}_predictions.npz",origins=t,layer_names=LAYERS,anchor=a,truth=y,
                        increments=component,attribution=attr,**{name:np.stack(p) for name,p in out.items()})
    write(path,dict(entity=e,release=EXPECTED,rows=[lo,hi],prior_scored_range=historical,
        selected_mode=state["chosen"],routing=activation,A_family=chain["gates"]["A"]["selected_family"],
        subsets=subsets,native_model=counts,checks=dict(continuous_issue=issue,additive_max=additive_max,
             attribution_max=attribution_max,native_max=native_error,cut=cut,native_poison_max=native_poison,
             python_poison_max=python_error,linear_poison_max=linear_error,Transformer_CPU=neural_audit)))
    print("SCORED",e,{name:table["scores"]["PRISM_v08"]["mean"]["RMSE"] for name,table in subsets.items()},flush=True)


def aggregate():
    all_results={e:read(ROOT/e/"TAIL.json") for e in CASES}
    write(ROOT/"TAIL_RESULTS.json",all_results)
    header=("entity","subset","N","model","seed",*METRICS)
    for mode in ("SUMMARY","SEEDS","SEED_SD"):
        with (ROOT/f"TAIL_{mode}.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(header)
            for e,obj in all_results.items():
                for name,tab in obj["subsets"].items():
                    for model,v in tab["scores"].items():
                        rows0=enumerate(v["runs"]) if mode=="SEEDS" else [(mode.lower(),v["mean" if mode=="SUMMARY" else "sd"])]
                        for seed,m in rows0:
                            writer.writerow((e,name,tab["N"],model,seed,*(m[key] for key in METRICS)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("verify","ingest","evaluate","aggregate"))
    parser.add_argument("entity",nargs="?",choices=CASES)
    a=parser.parse_args()
    reg=lock()
    if a.action=="verify":
        print("VERIFIED",EXPECTED)
    elif a.action=="ingest":
        ingest(reg)
    elif a.action=="evaluate":
        assert a.entity and read(PRIVATE/"SOURCES.json")["release"]==EXPECTED
        digest(PRIVATE/f"{a.entity}.npy",read(PRIVATE/"SOURCES.json")["arrays"][a.entity])
        evaluate(a.entity,reg)
    else:
        aggregate()


if __name__=="__main__":
    main()
