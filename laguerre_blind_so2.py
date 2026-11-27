"""Training-only SO2 adapter and one-shot evaluation for the frozen dual-view algorithm."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np
from openpyxl import load_workbook

from laguerre_dual import WINDOW, cuts, paired_dm, select
from laguerre_dual_views import DualModel, forecast, prepare, readout
from laguerre_screen import CONFIGS, SRU_SOURCE, Series, digest, lag, pulse, rmse, save_json
from laguerre_unified import apply_bias, pressure_metrics, summarize_gate


def rows_h2s_workbook(path,end):
    if digest(path)!=SRU_SOURCE:
        raise ValueError("Unexpected SRU workbook")
    w=load_workbook(path,read_only=True,data_only=True)
    try:
        sheet,=w.worksheets
        header=next(sheet.iter_rows(min_row=1,max_row=1,values_only=True))
        if header[5:]!=("y1","y2"):
            raise ValueError("Unexpected official target order")
        a=np.array([[v[6],*v[:5]] for v in sheet.iter_rows(
             min_row=2,max_row=end+1,max_col=7,values_only=True)],dtype=float)
    finally:
        w.close()
    if a.shape!=(end,6) or not np.isfinite(a).all():
        raise ValueError("Unexpected SO2 extent")
    return a


def verify_phase_one(receipt):
    f=json.loads(receipt.read_text())
    root=Path(__file__).resolve().parent
    if (f["status"] not in ("phase1_passed_frozen", "uniform_ecm_frozen")
            or any(digest(root/n)!=h for n,h in f["code"].items())):
        raise ValueError("Phase-one lock / source check failed")
    return f


def stage(workbook,authority,phase_one,out):
    verify_phase_one(phase_one)
    old=json.loads((authority/"FROZEN.json").read_text())
    cfg=CONFIGS["sru"][old["selected"]]
    if json.loads(json.dumps(asdict(cfg)))!=old["configs"][old["selected"]]:
        raise ValueError("Envelope does not match frozen recipe")
    out.mkdir(parents=True,exist_ok=False)
    train=rows_h2s_workbook(workbook,6048)
    np.savez_compressed(out/"TRAIN.npz",values=train)
    save_json(out/"REGISTRATION.json",dict(target="Fortuna SRU y2 SO2",source=SRU_SOURCE,
        protocol=[0,8,64],rows={"train":[0,6048],"development":[6048,8064],"tail":[8064,10081]},
        phase_one_sha=digest(phase_one),support_source=digest(authority/"FROZEN.json"),
        support=asdict(cfg),support_index=old["selected"],train_sha=digest(out/"TRAIN.npz"),
        adapter=digest(Path(__file__)),tail_read_count=0))


def fit(out):
    reg=json.loads((out/"REGISTRATION.json").read_text())
    if digest(out/"TRAIN.npz")!=reg["train_sha"] or digest(Path(__file__))!=reg["adapter"]:
        raise ValueError("SO2 train receipt changed")
    with np.load(out/"TRAIN.npz",allow_pickle=False) as p:
        a=p["values"]
    data=Series(a[:,1:],a[:,0],a[:,0],8)
    cfg=CONFIGS["sru"][reg["support_index"]]
    e,s,c,record=select(data,cfg,WINDOW)
    print("INNER",s,flush=True)
    gains=[];audit=[]
    for stop,t in cuts(data)[2]:
        endo,spec,curve,selection=select(data.through(stop),cfg,WINDOW)
        f=prepare(data.through(t[-1]+1),stop,spec["near"],endo,cfg)
        m,_=readout(f,spec["alpha"],spec["mix"],curve)
        loss=(f.target[t]-f.safe_hat[t])**2-(f.target[t]-forecast(f,m,t))**2
        gains.append(loss)
        audit.append(dict(outer=[stop,int(t[-1])+1],inner=selection["folds"],parameters=spec,mean_gain=float(loss.mean())))
    admission=summarize_gate(gains,8)
    f=prepare(data,len(data),s["near"],e,cfg)
    m,details=readout(f,s["alpha"],s["mix"],c)
    model=DualModel(f.safe,f.mapping,m,8,admission["admitted"])
    path=out/"MODEL.pkl"
    path.write_bytes(pickle.dumps(model,protocol=5))
    save_json(out/"FROZEN.json",dict(registration=digest(out/"REGISTRATION.json"),model=digest(path),
       inner=record,outer=audit,parameters=s,endo=e,curve=c,coefficients=details,exo=admission,
       endo_level_coef=float(f.safe.beta[-1]), training_issues=[int(f.rows[0]),int(f.rows[-1])+1]))
    print("FROZEN",s,admission,flush=True)


def holm(tests):
    value=0.
    order=sorted((n for n,q in tests.items() if q["p"] is not None),key=lambda n:tests[n]["p"])
    for i,n in enumerate(order):
        value=max(value,min(1.,tests[n]["p"]*(len(order)-i)))
        tests[n]["p_holm"]=value


def score(workbook,out,archive,phase_one):
    from laguerre_blind_baselines import predictors
    verify_phase_one(phase_one)
    lock=json.loads((out/"TEST_LOCK.json").read_text())
    for name,sha in lock["files"].items():
        if digest(out/name)!=sha:
            raise ValueError("Blind-test lock changed")
    output=out/"RESULTS.json"
    marker=out/"TAIL_OPENED.json"
    if marker.exists() or output.exists():
        raise FileExistsError("SO2 tail was already opened")
    models=predictors(archive,out)
    m=pickle.loads((out/"MODEL.pkl").read_bytes())
    save_json(marker,dict(test_lock=digest(out/"TEST_LOCK.json"),target="y2",reads=1))
    a=rows_h2s_workbook(workbook,10081)
    with np.load(out/"TRAIN.npz",allow_pickle=False) as q:
        np.testing.assert_array_equal(a[:6048],q["values"])
    s=Series(a[:,1:],a[:,0],a[:,0],8)
    ds=m.predict(s)
    stream=m.stream(5)
    actual=np.array([stream.step(x,h) for x,h in zip(s.x,lag(s.lab,8))])
    expected=np.column_stack([ds[k] for k in ("prediction","endo","trust")])
    dif=float(np.max(abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    t=np.arange(8064,10081)
    truth=s.truth[t]
    legs=models(a,t)
    clean={"PRISM":dict(rmse=rmse(truth,ds["prediction"][t]),mae=float(abs(truth-ds["prediction"][t]).mean()))}
    dm={}
    loss=(truth-ds["prediction"][t])**2
    for name,pred in legs.items():
        e=pred-truth
        clean[name]=dict(rmse=float(np.sqrt(np.mean(e**2,axis=1)).mean()),
                         ensemble_rmse=rmse(truth,pred.mean(axis=0)),mae=float(abs(e).mean()))
        dm[name]=paired_dm(np.mean(e**2,axis=0)-loss,8)
    holm(dm)
    stress={}
    win=t[:512]
    scale=s.x[:6048].std(axis=0)
    for kind,trial in (("oscillation",pulse(s,8064,scale)),("bias",apply_bias(s,8064,scale))):
        pred=m.predict(trial)
        panel={"PRISM":pressure_metrics(pred["prediction"][win],ds["prediction"][win],s.truth[win],
                                        ds["endo"][win],True,pred["trust"][win])}
        # Sensor-invariant Persistence and AR are retained as explicit controls.
        for name,sev in models(np.column_stack((trial.lab,trial.x)),win).items():
            panel[name]=pressure_metrics(sev.mean(axis=0),legs[name][:,:512].mean(axis=0),
                                         s.truth[win],ds["endo"][win],False)
        stress[kind]=panel
    save_json(output,dict(target="SO2 y2",test_lock=digest(out/"TEST_LOCK.json"),clean=clean,dm=dm,
         stress=stress,n=len(t),parity=dif,mean_rho=float(ds["trust"][t].mean()),
         p10_rho=float(np.quantile(ds["trust"][t], .1))))
    print("SO2",clean,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("stage")
    for name in ("workbook","authority","phase_one","out"):
        q.add_argument(name,type=Path)
    q=sub.add_parser("fit");q.add_argument("out",type=Path)
    q=sub.add_parser("score")
    for name in ("workbook","out","archive","phase_one"):
        q.add_argument(name,type=Path)
    a=p.parse_args()
    if a.action=="stage":
        stage(a.workbook,a.authority,a.phase_one,a.out)
    elif a.action=="score":
        score(a.workbook,a.out,a.archive,a.phase_one)
    else:
        fit(a.out)

