"""Complete the frozen Stream 11 benchmark on its registered issue and label clocks."""

import argparse
import json
from pathlib import Path
import pickle
import sys
from itertools import product

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


LENGTH = 64
SEEDS = tuple(range(5))
RIDGE = (.01, .1, 1., 10., 100.)
ENET = tuple(product((.001, .003, .01, .03), (.2, .8)))
LGB = tuple(product((15, 31), (20, 50), (150, 300)))
TF = tuple(product((32, 64), (.0003, .001)))
EPOCHS = 12


def design(s, rows):
    anchor = s.anchor()[rows]
    # The archive uses the latest assay and its previous three differences.
    ar = np.column_stack((anchor, s.lab[rows[:, None]-s.delay-np.arange(1, 4)]-anchor[:, None]))
    edges = np.arange(0, LENGTH+1, 4)
    cumulative = np.vstack((np.zeros(s.x.shape[1]), s.x.cumsum(axis=0)))
    ago = rows[:, None]+1-edges
    patches = (cumulative[ago[:, :-1]]-cumulative[ago[:, 1:]])/4.
    return ar, np.column_stack((ar, s.x[rows], patches.reshape(len(rows), -1))), patches


def tokens(s, rows):
    ar, _, patches = design(s, rows)
    x = np.concatenate((patches[:, ::-1], s.x[rows, None, :]), axis=1)
    return np.concatenate((x, np.broadcast_to(ar[:, None, :], (*x.shape[:2], ar.shape[1]))),
                          axis=2).astype(np.float32)


def fit_linear(x, y, factory, scale_target=False):
    scale = StandardScaler().fit(x)
    location = float(y.mean()) if scale_target else 0.
    spread = float(y.std()) if scale_target else 1.
    model = factory().fit(scale.transform(x), (y-location)/spread)
    return scale, model, location, spread


def linear_predict(arm, x):
    scale, model, location, spread = arm
    return model.predict(scale.transform(x))*spread+location


def linear_kkt(arm, x, y):
    scale, model, location, spread = arm
    z = scale.transform(x)
    e = model.predict(z)-(y-location)/spread
    if isinstance(model, Ridge):
        residual = z.T@e/len(e) + model.alpha*model.coef_/len(e)
    else:
        b = model.coef_
        gradient = z.T@e/len(e) + model.alpha*(1-model.l1_ratio)*b
        radius = model.alpha*model.l1_ratio
        residual = np.where(b != 0., gradient + radius*np.sign(b), np.maximum(0., abs(gradient)-radius))
    return dict(slope_inf=float(np.max(abs(residual))), intercept_abs=float(abs(e.mean())))


def lgbm(spec, seed):
    from lightgbm import LGBMRegressor
    leaves, minimum, rounds = spec
    return LGBMRegressor(num_leaves=leaves, min_child_samples=minimum, n_estimators=rounds,
                         learning_rate=.04, reg_lambda=1., subsample=.85, subsample_freq=1,
                         colsample_bytree=.75, n_jobs=1, deterministic=True, force_col_wise=True,
                         random_state=seed, verbosity=-1)


def select(grid, folds, fit, predict, x, y):
    scores = []
    count = sum(len(valid) for _, valid in folds)
    for spec in grid:
        sse = 0.
        for rows, valid in folds:
            model = fit(x[rows], y[rows], spec)
            e = predict(model, x[valid])-y[valid]
            sse += e@e
        scores.append(float(sse/count))
    return grid[int(np.argmin(scores))], scores


def cpu(s, rows, folds, out):
    y = s.truth-s.anchor()
    selected = y[rows]
    ar, arx, _ = design(s, np.arange(LENGTH+s.delay, len(s)))
    offset = LENGTH+s.delay
    delta_rows = rows-offset
    dev = [(a-offset,b-offset) for a,b in folds]
    y = y[offset:]
    ridge = lambda x,y,a: fit_linear(x, y, lambda: Ridge(alpha=a))
    def enet(x, y, spec):
        alpha, mix = spec
        return fit_linear(x,y,lambda: ElasticNet(alpha=alpha,l1_ratio=mix,max_iter=12000,tol=5e-7),True)
    ar_pick, ar_grid = select(RIDGE,dev,ridge,linear_predict,ar,y)
    en_pick, en_grid = select(ENET,dev,enet,linear_predict,arx,y)
    lg_pick, lg_grid = select(LGB,dev,lambda x,y,s:lgbm(s,0).fit(x,y),
                              lambda m,x:m.booster_.predict(x,num_threads=1),arx,y)
    arms = dict(AR=ridge(ar[delta_rows],selected,ar_pick),
                ARX=ridge(arx[delta_rows],selected,10.),
                ElasticNet=enet(arx[delta_rows],selected,en_pick))
    for seed in SEEDS:
        lgbm(lg_pick,seed).fit(arx[delta_rows],selected).booster_.save_model(str(out/f"LGB_{seed}.txt"))
    (out/"linear.pkl").write_bytes(pickle.dumps(arms,protocol=5))
    audit={name:linear_kkt(model,(ar if name=="AR" else arx)[delta_rows],selected)
           for name,model in arms.items()}
    if max(max(v.values()) for v in audit.values()) > 1e-5:
        raise RuntimeError(f"Linear baseline KKT audit failed: {audit}")
    return dict(AR=ar_pick,ARX=10.,ElasticNet=en_pick,LightGBM=lg_pick,AR_grid=ar_grid,
                ENet_grid=en_grid,LightGBM_grid=lg_grid,kkt=audit)


def network(width, channels, count):
    import torch
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=nn.Linear(channels,width)
            pos=torch.arange(count).float()[:,None]
            scale=torch.exp(torch.arange(0,width,2).float()*(-np.log(10000.)/width))
            pe=torch.zeros(count,width)
            pe[:,0::2]=torch.sin(pos*scale)
            pe[:,1::2]=torch.cos(pos*scale)
            self.register_buffer("position",pe)
            layer=nn.TransformerEncoderLayer(width,4,4*width,.1,activation="gelu",batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.head=nn.Linear(width,1)

        def forward(self,x):
            return self.head(self.encoder(self.embed(x)+self.position)[:,-1]).squeeze(-1)
    return Head()


def fit_tf(x,y,rows,spec,seed,epochs,device,validation=None):
    import torch
    torch.manual_seed(seed)
    generator=torch.Generator(device=device).manual_seed(seed)
    raw=x[rows]
    mean=raw.mean(axis=(0,1))
    scale=np.maximum(raw.std(axis=(0,1)),1e-7)
    loc,sd=float(y[rows].mean()),float(y[rows].std())
    feature=torch.as_tensor((raw-mean)/scale,device=device)
    response=torch.as_tensor((y[rows]-loc)/sd,dtype=torch.float32,device=device)
    net=network(spec[0],x.shape[2],x.shape[1]).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=spec[1])
    trace=[]
    if validation is not None:
        v=torch.as_tensor((x[validation]-mean)/scale,device=device)
        target=torch.as_tensor(y[validation],device=device)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(rows),device=device,generator=generator).split(128):
            opt.zero_grad(set_to_none=True)
            loss=(net(feature[ix])-response[ix]).square().mean()
            loss.backward()
            opt.step()
        if validation is not None:
            net.eval()
            with torch.inference_mode():
                p=torch.cat([net(block) for block in v.split(128)])*sd+loc
                trace.append(float((p-target).square().sum()))
    return dict(state={k:v.cpu() for k,v in net.state_dict().items()}, mean=mean, scale=scale, loc=loc,sd=sd,
                spec=spec),trace


def transformer(s,rows,folds,out):
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    y=s.truth-s.anchor()
    offset=LENGTH+s.delay
    x=tokens(s,np.arange(offset,len(s)))
    y=y[offset:]
    folds=[(a-offset,b-offset) for a,b in folds]
    count=sum(len(b) for _,b in folds)
    risk=[]
    for spec in TF:
        trace=np.sum([fit_tf(x,y,a,spec,0,EPOCHS,device,b)[1] for a,b in folds],axis=0)/count
        risk.append(trace.tolist())
        print(s.delay,spec,min(trace),flush=True)
    i,j=np.unravel_index(np.argmin(risk),(len(TF),EPOCHS))
    for seed in SEEDS:
        arm,_=fit_tf(x,y,rows-offset,TF[i],seed,j+1,device)
        torch.save(arm,out/f"TF_{seed}.pt")
    return dict(spec=TF[i],epochs=int(j+1),device=device,risks=risk)


def predict_bank(s,rows,out):
    import torch
    from lightgbm import Booster
    anchor=s.anchor()[rows]
    ar,x,_=design(s,rows)
    weights=pickle.loads((out/"linear.pkl").read_bytes())
    pred={"Persistence":anchor[None,:],
          **{k:(anchor+linear_predict(m,ar if k=="AR" else x))[None,:] for k,m in weights.items()},
          "LightGBM": np.array([anchor+Booster(model_file=str(out/f"LGB_{i}.txt")).predict(x,num_threads=1)
                                 for i in SEEDS])}
    tokens_=tokens(s,rows)
    neural=[]
    torch.set_num_threads(2)
    for i in SEEDS:
        a=torch.load(out/f"TF_{i}.pt",map_location="cpu",weights_only=False)
        model=network(a["spec"][0],tokens_.shape[2],tokens_.shape[1]).eval()
        model.load_state_dict(a["state"])
        with torch.inference_mode():
            z=torch.from_numpy((tokens_-a["mean"])/a["scale"])
            p=torch.cat([model(h) for h in z.split(128)]).numpy()
        neural.append(anchor+a["loc"]+a["sd"]*p.astype(float))
    pred["Transformer"]=np.array(neural)
    return pred


def score(data,run,out,source_lock,d):
    from laguerre_bj import TEP_ARX
    from laguerre_blind_so2 import holm
    from laguerre_dual import paired_dm
    from laguerre_screen import append,digest,pulse,read_tep,save_json
    from laguerre_unified import apply_bias,pressure_metrics
    records={kind:json.loads((out/kind/"FIT.json").read_text()) for kind in ("cpu","transformer")}
    for kind,r in records.items():
        assert r["train"]==digest(data/"train.npz") and r["version"]==digest(source_lock)
        assert all(digest(out/kind/f)==h for f,h in r["weights"].items())
    state=json.loads((run/"FROZEN.json").read_text())
    assert digest(run/"MODEL.pkl")==state["model"]
    reference=json.loads((run/"RESULTS.json").read_text())
    assert digest(run/"FROZEN.json")==reference["frozen"]
    save_json(out/"TEST_LOCK.json",dict(model=state["model"],version=digest(source_lock),
              fits={k:digest(out/k/"FIT.json") for k in records}))
    train,dev,tail=(read_tep(data,k,d) for k in ("train","development","tail"))
    full=append(append(train,dev),tail)
    start=len(train)+len(dev)
    ids=np.arange(start,len(full))
    scale=train.x.std(axis=0)
    model=pickle.loads((run/"MODEL.pkl").read_bytes())
    trials=dict(clean=full,oscillation=pulse(full,start,scale),bias=apply_bias(full,start,scale))
    results={}
    clean={}
    # Merge the disjoint CPU and Transformer artifacts without modifying their receipts.
    bank=out/"bank"
    bank.mkdir(exist_ok=True)
    for kind,rec in records.items():
        for file in rec["weights"]:
            (bank/file).symlink_to((out/kind/file).resolve())
    for label,series in trials.items():
        rows=ids if label=="clean" else ids[:512]
        actual=full.truth[rows]
        base=predict_bank(series,rows,bank)
        pr=model.predict(series)
        base["PRISM"]=pr["prediction"][rows][None,:]
        if label=="clean":
            clean=base
        dm={}
        metrics={}
        for name,seeds in base.items():
            loss=(seeds-actual)**2
            forecast=seeds.mean(axis=0)
            m=pressure_metrics(forecast,clean[name][:,:len(rows)].mean(axis=0),actual,pr["endo"][rows],
                               name=="PRISM",pr["trust"][rows] if name=="PRISM" else None)
            m["mean_seed_rmse"]=float(np.sqrt(loss.mean(axis=1)).mean())
            m["mean_seed_mae"]=float(np.abs(seeds-actual).mean())
            metrics[name]=m
            if name!="PRISM":
                dm[name]=paired_dm(loss.mean(axis=0)-(base["PRISM"][0]-actual)**2,d)
        holm(dm)
        results[label]=dict(metrics=metrics,dm=dm,issue_rows=[int(rows[0]),int(rows[-1])+1])
    np.testing.assert_allclose(results["clean"]["metrics"]["ARX"]["rmse"],TEP_ARX[d],rtol=0,atol=2e-12)
    np.testing.assert_allclose(results["clean"]["metrics"]["PRISM"]["rmse"],reference["tail_rmse"],rtol=0,atol=2e-14)
    save_json(out/"RESULTS.json",dict(delay=d,tail=digest(data/"tail.npz"),lock=digest(out/"TEST_LOCK.json"),
              tests=results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("cpu","transformer","score"))
    for name in ("source_lock","data","run","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("delay",type=int,choices=(1,2))
    a=parser.parse_args()
    sys.path.insert(0,str(a.source_lock.parent))
    from laguerre_blind_so2 import verify_phase_one
    from laguerre_dual import WINDOW, late_folds
    from laguerre_screen import digest,read_tep,save_json,train_rows
    f=verify_phase_one(a.source_lock)
    assert f["commit"]=="7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c"
    data=read_tep(a.data,"train",a.delay)
    fit=json.loads((a.run/"FROZEN.json").read_text())
    assert fit["data"]["train.npz"]==digest(a.data/"train.npz")
    if a.action=="score":
        score(a.data,a.run,a.out,a.source_lock,a.delay)
    else:
        dest=a.out/a.action
        dest.mkdir(parents=True,exist_ok=False)
        rows=train_rows(a.delay,len(data))
        folds=[(train_rows(a.delay,s),t) for s,t in late_folds(data,WINDOW)]
        metadata=(cpu if a.action=="cpu" else transformer)(data,rows,folds,dest)
        save_json(dest/"FIT.json",dict(train=digest(a.data/"train.npz"),version=digest(a.source_lock),
                  script=digest(Path(__file__)), rows=[int(rows[0]),int(rows[-1])+1],
                  folds=[([int(s[0]),int(s[-1])+1],[int(t[0]),int(t[-1])+1]) for s,t in folds],
                  fit=metadata,weights={p.name:digest(p) for p in sorted(dest.iterdir())}))
