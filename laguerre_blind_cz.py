"""Private, local direct-horizon transfer of the locked dual-view estimator."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import importlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import laguerre_dual as algorithm
from laguerre_anchored import Profiled, Standardized, X_BANKS
from laguerre_blind_so2 import holm, verify_phase_one
from laguerre_dual import paired_dm
from laguerre_dual_views import Fold, forecast, prepare_safe, readout
from laguerre_ecm import Dynamics, ExogenousMap
from laguerre_joint import LaguerreARX
from laguerre_screen import CONFIGS, digest, rmse, save_json
from laguerre_unified import pressure_metrics, summarize_gate


PROTOCOL = {
    "Rod1": (12061,16082,20103,(18184,19369)),
    "Rod2": (12376,16501,20627,(18631,19882)),
}
COLUMNS = ("晶体直径","主加热功率","晶升速度","埚升速度","晶转速度","埚转速度")
HORIZONS = (8,16,32,48,64)
WINDOW = 128
FIRST = 192
LGB_GRID = tuple(product(("level","delta"),((7,15),(15,30),(31,60)),(150,400),(.1,10.)))
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class Direct:
    x: np.ndarray
    lab: np.ndarray
    truth: np.ndarray
    delay: int
    anchor_delay = 0

    @classmethod
    def from_values(cls, value, horizon):
        y = value[:,0]
        return cls(value[:,1:],y,np.r_[y[horizon:],np.full(horizon,np.nan)],horizon)

    def __len__(self):
        return len(self.lab)

    def through(self, stop):
        return Direct(self.x[:stop],self.lab[:stop],self.truth[:stop],self.delay)

    def anchor(self):
        return self.lab

    def with_x(self,x):
        return Direct(x,self.lab,self.truth,self.delay)


def direct_rows(horizon,stop):
    return np.arange(FIRST,stop-horizon)


def prepare(data,stop,near,endo,config):
    # The anchor is current y at D=0. The response is H steps ahead.
    rows=direct_rows(data.delay,stop)
    safe,z,y0=prepare_safe(data,rows,endo)
    mapping=ExogenousMap.fit(data.x,rows,Dynamics(near,X_BANKS),config.padding,config.width)
    d=mapping.design(data.x)
    mask=np.ones(d.raw.shape[1:3],dtype=bool)
    x=d.raw[rows,:,:,0].reshape(len(rows),-1)
    p=Profiled(Standardized(np.column_stack((z[rows],x)),(data.truth-y0)[rows]),z.shape[1])
    # With D=0 there is no difference transfer. All exact level taps survive.
    return Fold(mapping,d,rows,z,safe,y0,mask,d.raw[rows].mean(axis=0),data.truth,p,near,near)


def select(data,config):
    with patch.object(algorithm,"train_rows",direct_rows),patch.object(algorithm,"prepare",prepare):
        return algorithm.select(data,config,WINDOW)


def model_fit(values,horizon,config,out):
    data=Direct.from_values(values,horizon)
    endo,spec,curve,search=select(data,config)
    t=direct_rows(horizon,len(data))
    edges=np.linspace(t[int(.774*len(t))],t[-1]+1,5,dtype=int)
    gains=[]; outer=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        lo,hi=int(lo),int(hi)
        test=np.arange(lo,hi)
        e,s,c,r=select(data.through(lo),config)
        f=prepare(data.through(hi),lo,s["near"],e,config)
        j,_=readout(f,s["alpha"],s["mix"],c)
        truth=f.target[test]
        err=(truth-f.safe_hat[test])**2-(truth-forecast(f,j,test))**2
        gains.append(err)
        outer.append(dict(test=[lo,hi],inner=r["folds"],parameters=s))
    gate=summarize_gate(gains,horizon)
    f=prepare(data,len(data),spec["near"],endo,config)
    j,audit=readout(f,spec["alpha"],spec["mix"],curve)
    model=LaguerreARX(f.safe,f.mapping,j,gate["admitted"])
    out.mkdir(exist_ok=False)
    path=out/"MODEL.pkl"
    path.write_bytes(pickle.dumps(model,protocol=5))
    save_json(out/"FROZEN.json",dict(horizon=horizon,model=digest(path),parameters=spec,
          curvature=curve,safe=endo,fit=audit,search=search,outer=outer,exogenous=gate,
          endo_level_coef=float(f.safe.beta[-1]), training_issues=[int(f.rows[0]),int(f.rows[-1])+1]))
    print("CZ H",horizon,spec,"HAC",gate["t_stat"],gate["positive_blocks"],gate["admitted"],flush=True)


def legacy(protocol):
    sys.path.insert(0,str(protocol))
    return tuple(importlib.import_module(n) for n in ("experiment","linear"))


def head(config,seed):
    import lightgbm as lgb
    _,(leaves,minimum),rounds,penalty=config
    return lgb.LGBMRegressor(num_leaves=leaves,min_child_samples=minimum,n_estimators=rounds,
        reg_lambda=penalty,learning_rate=.04,subsample=.85,subsample_freq=1,colsample_bytree=.75,
        n_jobs=1,deterministic=True,force_col_wise=True,random_state=seed,verbosity=-1)


def fit_baselines(rod,out,protocol,transformer, values=None):
    x,lin=legacy(protocol)
    name="CZ_"+rod
    q=x.TASKS[name]
    if values is None:
        values=train_values(out)
    dst=out/"Baselines"
    dst.mkdir(exist_ok=False)
    lin.ROOT=dst
    lin.data=lambda _:values
    save_json(dst/"REGISTRATION.json",dict(geometry=asdict(q),linear_source=digest(Path(lin.__file__)),
        adapter=digest(Path(__file__)),fit_scope="training prefix only",lightgbm_grid=LGB_GRID))
    lin.train(q)
    folds=[(tr,ev,lin.design(q,values,tr,"ARX"),lin.design(q,values,ev,"ARX")) for tr,ev in x.folds(q)]
    def loss(h):
        mse=n=0
        for tr,ev,d,p in folds:
            y,_=q.target(values,tr,h[0]); target,_=q.target(values,ev,h[0])
            fitted=head(h,20261125).fit(d,y)
            e=fitted.booster_.predict(p,num_threads=1)-target
            mse+=e@e
            n+=len(e)
        return float(mse/n)
    with ThreadPoolExecutor(max_workers=3) as pool:
        risk=list(pool.map(loss,LGB_GRID))
    pick=LGB_GRID[int(np.argmin(risk))]
    origins=q.train_origins()
    m=lin.design(q,values,origins,"ARX")
    y,_=q.target(values,origins,pick[0])
    weights={}
    for seed in SEEDS:
        fit=head(pick,seed).fit(m,y)
        a=dst/f"lgb_{seed}.txt"
        fit.booster_.save_model(str(a))
        weights[a.name]=digest(a)
    t=json.loads((transformer/"FROZEN.json").read_text())
    if t["last_training_target"]!=int(origins[-1])+q.H:
        raise ValueError("Transformer clock differs")
    for file,h in t["files"].items():
        if digest(transformer/file)!=h:
            raise ValueError("Transformer weight receipt changed")
    p=dst/name/"linear/MODELS.pkl"
    save_json(dst/"FROZEN.json",dict(linear=digest(p),recipe=pick,search=risk,lightgbm=weights,
          transformer_files=t["files"],transformer_manifest=digest(transformer/"FROZEN.json")))
    print("CZ BASELINES",rod,pick,flush=True)


def train_values(out):
    f=json.loads((out/"REGISTRATION.json").read_text())
    path=out/"TRAIN.npz"
    if f["adapter"]!=digest(Path(__file__)) or digest(path)!=f["train_file"]:
        raise ValueError("Training receipt changed")
    with np.load(path,allow_pickle=False) as p:
        return p["values"]


def stage(rod,prefix,out,phase_one,authority):
    verify_phase_one(phase_one)
    train,dev,total,overlap=PROTOCOL[rod]
    old=json.loads((authority/"FROZEN.json").read_text())
    config=CONFIGS["sru"][old["selected"]]
    out.mkdir(parents=True,exist_ok=False)
    path=prefix/"train.npz"
    with np.load(path,allow_pickle=False) as a:
        data=a["values"]
    if data.shape!=(train,6):
        raise ValueError("Unexpected native CZ training extent")
    np.savez_compressed(out/"TRAIN.npz",values=data)
    save_json(out/"REGISTRATION.json",dict(rod=rod,protocol=[16,0,128],horizons=HORIZONS,
       starts=[0,train,dev,total],overlap=overlap,exact_history_start=FIRST,
       d0="Full exogenous level transfers; the endogenous increment uses span H.",
       phase_one=digest(phase_one),adapter=digest(Path(__file__)),source_train=digest(path),
       train_file=digest(out/"TRAIN.npz"),performance_read="historical tail; frozen before this replay",
       envelope_source=digest(authority/"FROZEN.json"),envelope_index=old["selected"],envelope=asdict(config)))


def full_rod(workbook,rod,train,prefix):
    name={"Rod1":"Sheet1","Rod2":"Sheet2"}[rod]
    a=pd.read_excel(workbook,sheet_name=name,usecols=list(COLUMNS))[list(COLUMNS)].to_numpy(dtype=float)
    n,d,total,_=PROTOCOL[rod]
    if a.shape!=(total,6) or not np.isfinite(a).all():
        raise ValueError("CZ workbook does not match expected geometry")
    with np.load(prefix/"development.npz",allow_pickle=False) as p:
        dev=p["values"]
    np.testing.assert_array_equal(a[:n],train)
    np.testing.assert_array_equal(a[n:d],dev)
    return a


def predictors(rod,out,protocol,transformer,arch):
    import lightgbm as lgb
    import torch
    x,lin=legacy(protocol)
    name="CZ_"+rod
    q=x.TASKS[name]
    d=out/"Baselines"
    f=json.loads((d/"FROZEN.json").read_text())
    p=d/name/"linear/MODELS.pkl"
    if digest(p)!=f["linear"]:
        raise ValueError("Linear baseline changed")
    linear=pickle.loads(p.read_bytes())
    boosters=[]
    for path,digest_ in f["lightgbm"].items():
        if digest(d/path)!=digest_:
            raise ValueError("LightGBM changed")
        boosters.append(lgb.Booster(model_file=str(d/path)))
    if digest(transformer/"FROZEN.json")!=f["transformer_manifest"]:
        raise ValueError("Transformer fit manifest changed")
    spec=importlib.util.spec_from_file_location("cz_registered_transformer",arch)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    models=[]
    for name,digest_ in f["transformer_files"].items():
        path=transformer/name
        if digest(path)!=digest_:
            raise ValueError("Archived neural file changed")
        state=torch.load(path,map_location="cpu",weights_only=False)
        net=module.build("Transformer",6,state["config"][0],q.L)
        net.load_state_dict(state["state"])
        models.append((net.eval(),state))
    def run(a,t):
        anchor=a[t,0]
        all_={"Persistence":anchor[None,:]}
        for k,arms in linear.items():
            z=lin.design(q,a,t,k)
            all_[k]=np.array([m.predict(s["scale"].transform(z))*sd+center+
                     (anchor if s["recipe"]=="delta" else 0.)
                     for s in arms for m,center,sd in [s["model"]]])
        z=lin.design(q,a,t,"ARX")
        all_["LightGBM"]=np.array([fitted.predict(z,num_threads=1)+(anchor if f["recipe"][0]=="delta" else 0.)
                                  for fitted in boosters])
        ix=t[:,None]-np.arange(q.L-1,-1,-1)
        raw=np.concatenate((a[ix,1:],a[ix,0,None]),axis=2).astype(np.float32)
        readings=[]
        for net,state in models:
            center,scale,mean,sd=state["stats"]
            batch=torch.from_numpy((raw-center)/scale)
            with torch.inference_mode():
                y=torch.cat([net(z) for z in batch.split(64)]).numpy().astype(float)
            readings.append(y*sd+mean+(anchor if state["config"][2]=="delta" else 0.))
        all_["Transformer"]=np.array(readings)
        return all_
    return run


def score(rod,out,prefix,workbook,protocol,transformer,arch,public):
    d=out/"Baselines"
    manifest=out/"TEST_LOCK.json"
    if manifest.exists():
        raise FileExistsError("CZ replay already registered")
    models={}
    for horizon in HORIZONS:
        p=out/f"H{horizon}"
        f=json.loads((p/"FROZEN.json").read_text())
        if digest(p/"MODEL.pkl")!=f["model"]:
            raise ValueError("Frozen CZ model changed")
        models[horizon]=pickle.loads((p/"MODEL.pkl").read_bytes()),f
    baseline=predictors(rod,out,protocol,transformer,arch)
    save_json(manifest,dict(registration=digest(out/"REGISTRATION.json"),baseline=digest(d/"FROZEN.json"),
        architecture=digest(arch),horizons={str(k):digest(out/f"H{k}/FROZEN.json") for k in HORIZONS}))
    a=full_rod(workbook,rod,train_values(out),prefix)
    _,dev,total,(overlap_start,overlap_stop)=PROTOCOL[rod]
    subset=lambda t:(t<overlap_start)|(t>=overlap_stop)
    common=np.arange(dev+max(HORIZONS),total)
    common=common[subset(common)]
    model_rows=[]; result={}
    for h,(model,f) in models.items():
        series=Direct.from_values(a,h)
        result[h]=model,model.predict(series)
        p=result[h][1]
        rows=common-h
        stream=model.stream()
        online=np.array([stream.step(tx,y) for tx,y in zip(series.x,series.lab)])
        compare=np.column_stack([p[k] for k in ("prediction","endo","trust")])
        np.testing.assert_allclose(online,compare,atol=2e-9,rtol=0)
        model_rows.append(dict(h=h,n=len(rows),rmse=rmse(a[common,0],p["prediction"][rows]),
           safe_rmse=rmse(a[common,0],p["endo"][rows]),max_stream_abs=float(np.max(abs(online-compare))),
           exo=f["exogenous"],parameters=f["parameters"]))
    model,p=result[16]
    t=np.arange(dev,total-16)
    t=t[subset(t+16)]
    truth=a[t+16,0]
    base=baseline(a,t)
    candidates={"PRISM":p["prediction"][t][None,:],**base}
    metrics={}
    for name,y in candidates.items():
        e=y-truth
        metrics[name]=dict(rmse=float(np.sqrt(np.mean(e**2,axis=1)).mean()),mae=float(abs(e).mean()),
                           ensemble_rmse=rmse(truth,y.mean(axis=0)),seeds=len(y))
    # Preserve time gaps in HAC after excluding the overlapping raw target range.
    dm={}
    pieces=np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>1)+1)
    loss=(candidates["PRISM"][0]-truth)**2
    for name,y in base.items():
        delta=np.mean((y-truth)**2,axis=0)-loss
        residual=delta-delta.mean()
        lag_=max(15,int(4*(len(delta)/100)**(2/9)))
        totalvar=residual@residual
        for k in range(1,lag_+1):
            totalvar+=2*(1-k/(lag_+1))*sum(residual[v][k:]@residual[v][:-k] for v in pieces)
        from scipy.stats import norm
        sd=np.sqrt(max(totalvar,0.))/len(delta)
        stat=float(delta.mean()/sd) if sd else None
        dm[name]=dict(n=len(delta),lag=lag_,loss_gain=float(delta.mean()),t=stat,
                      p=float(2*norm.sf(abs(stat))) if stat is not None else None)
    holm(dm)
    stress={}
    if rod=="Rod2":
        win=t[:512]; targets=a[win+16,0]; y0=p["endo"][win]
        scale=a[:PROTOCOL[rod][0],1:].std(axis=0)
        rng=np.random.default_rng(20261125)
        signs=rng.choice((-1,1),256)
        for key,perturb in (("oscillation",np.c_[signs,-signs].ravel()),("bias",np.ones(512))):
            moved=a.copy()
            moved[win,1:]+=4*scale*perturb[:,None]
            projections=baseline(moved,win)
            pred=model.predict(Direct.from_values(moved,16))
            stress[key]={"PRISM":pressure_metrics(pred["prediction"][win],p["prediction"][win],targets,
                                                     y0,True,pred["trust"][win])}
            for key2,y in projections.items():
                stress[key][key2]=pressure_metrics(y.mean(axis=0),base[key2][:,:512].mean(axis=0),targets,y0,False)
    save_json(public,dict(rod=rod,source=digest(workbook),lock=digest(manifest),n=len(t),
       common_targets={"first":int(common[0]),"stop":total,"n":len(common),
                       "overlap_removed":list(PROTOCOL[rod][3])},
       horizons=model_rows,h16=metrics,paired_dm=dm,stress=stress))
    print("CZ SCORED",rod,metrics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    for k,names in (("stage",("prefix","out","phase_one","authority")),
                    ("fit",("out",)),("baseline",("out","protocol","transformer")),
                    ("score",("out","prefix","workbook","protocol","transformer","arch","public"))):
        z=sub.add_parser(k);z.add_argument("rod",choices=PROTOCOL)
        for n in names:z.add_argument(n,type=Path)
        if k=="fit": z.add_argument("horizon",type=int,choices=HORIZONS)
    a=p.parse_args()
    if a.action=="stage":
        stage(a.rod,a.prefix,a.out,a.phase_one,a.authority)
    elif a.action=="fit":
        m=json.loads((a.out/"REGISTRATION.json").read_text())
        model_fit(train_values(a.out),a.horizon,CONFIGS["sru"][m["envelope_index"]],a.out/f"H{a.horizon}")
    elif a.action=="baseline":
        fit_baselines(a.rod,a.out,a.protocol,a.transformer)
    else:
        score(a.rod,a.out,a.prefix,a.workbook,a.protocol,a.transformer,a.arch,a.public)

