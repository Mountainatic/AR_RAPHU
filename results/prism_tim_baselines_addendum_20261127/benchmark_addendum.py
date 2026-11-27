"""Seal and score three additional controls for the frozen Laguerre ECM."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
from scipy.special import log_ndtr
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import StandardScaler


CASES = ("TEP_D1", "TEP_D2", "H2S", "SO2", "Rod1", "Rod2")
PLS_COMPONENTS = (2, 4, 8, 12, 16, 24)
GRU_WIDTH = (24, 48)
GRU_EPOCHS = 8
GRU_LR = .001
GRU_BATCH = 256
OSC_SEED = 20261125
NEW_MODELS = ("ElasticNet_clipped", "DPLS", "GRU")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_pickle(path):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = pickle.loads(path.read_bytes())
    versions = sorted({str(n.message) for n in notices
                       if issubclass(n.category, InconsistentVersionWarning)})
    return model, versions


class Protocol:
    def __init__(self, a):
        import laguerre_blind_cz as cz
        from laguerre_blind_so2 import verify_phase_one
        from laguerre_screen import Series, digest, read_sru_prefix, read_tep, train_rows
        self.a = a
        self.lock = verify_phase_one(a.lock)
        assert self.lock["commit"] == "7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c"
        assert all(digest(a.source/f) == v for f,v in self.lock["code"].items())
        self.tep = a.task.startswith("TEP")
        self.cz = a.task in cz.PROTOCOL
        self.mode_dir = a.model/"H16" if self.cz else a.model
        self.model_fit = read(self.mode_dir/"FROZEN.json")
        self.model_digest = digest(self.mode_dir/"FROZEN.json")
        if self.cz:
            reg = read(a.model/"REGISTRATION.json")
            test = read(a.model/"TEST_LOCK.json")
            assert reg["phase_one"] == digest(a.lock)
            assert test["horizons"]["16"] == self.model_digest
            assert reg["source_train"] == digest(a.data/"train.npz")
            assert test["baseline"] == digest(a.model/"Baselines/FROZEN.json")
            self.train = cz.Direct.from_values(cz.train_values(a.model),16)
            self.horizon, self.gap, self.window, self.history = 16, 16, 128, 128
            self.train_indices = lambda stop: cz.direct_rows(16,stop)
            self.first = cz.PROTOCOL[a.task][1]
        elif a.task == "SO2":
            reg = read(a.model/"REGISTRATION.json")
            test = read(a.model/"TEST_LOCK.json")
            assert reg["phase_one_sha"] == digest(a.lock)
            assert test["files"]["FROZEN.json"] == self.model_digest
            assert test["files"]["TRAIN.npz"] == digest(a.model/"TRAIN.npz")
            with np.load(a.model/"TRAIN.npz",allow_pickle=False) as p:
                v=p["values"]
            self.train=Series(v[:,1:],v[:,0],v[:,0],8)
            self.first=reg["rows"]["tail"][0]
            self.horizon, self.gap, self.window, self.history = 0, 8, 64, 64
            self.train_indices=lambda stop:train_rows(8,stop)
        else:
            previous=read(a.model/"RESULTS.json")
            assert previous["frozen"] == self.model_digest
            assert all(self.lock["code"][n] == v for n,v in self.model_fit["sources"].items())
            assert self.model_fit["data"]["train.npz"] == digest(a.data/"train.npz")
            d=int(a.task[-1]) if self.tep else 8
            self.train=read_tep(a.data,"train",d) if self.tep else read_sru_prefix(a.data,"train")
            self.first=8000 if self.tep else 8064
            self.horizon, self.gap, self.window, self.history = 0, d, 64, 64
            self.train_indices=lambda stop:train_rows(d,stop)
        self.rows=self.train_indices(len(self.train))
        assert [int(self.rows[0]),int(self.rows[-1])+1] == self.model_fit["training_issues"]
        self.folds=[]
        for lo,hi in ((.70,.85),(.85,1.)):
            val=self.rows[int(lo*len(self.rows)):int(hi*len(self.rows))]
            tr=self.train_indices(int(val[0])-self.window)
            if tr[-1]+self.gap >= val[0]-self.window or val[-1]+self.gap >= len(self.train):
                raise ValueError("Fold label crosses its maturity embargo")
            self.folds.append((tr,val))
        self.seed=tuple(range(3 if self.cz else 5))
        self.fit_first=int(self.rows[0])

    def verify_source(self):
        from laguerre_screen import digest
        if self.tep:
            if digest(self.a.tep_code/"tep_baselines.py") != self.a.tep_helper_digest:
                raise ValueError("TEP helper changed")

    def linear(self,s,rows):
        if self.tep:
            from tep_baselines import design
            return design(s,rows)[1]
        ix=rows[:,None]-np.arange(self.history)
        return np.column_stack((s.anchor()[ix],s.x[ix].reshape(len(rows),-1)))

    def sequence(self,s,rows):
        if self.tep:
            from tep_baselines import tokens
            return tokens(s,rows)
        ix=rows[:,None]-np.arange(self.history-1,-1,-1)
        return np.concatenate((s.x[ix],s.anchor()[ix,None]),axis=2).astype(np.float32)

    def all_data(self):
        from laguerre_blind_cz import Direct, PROTOCOL, full_rod
        from laguerre_blind_so2 import rows_h2s_workbook
        from laguerre_control import load_prefix
        from laguerre_screen import Series, append, digest, read_tep, receipt
        from laguerre_unified import full_h2s
        a=self.a
        if self.cz:
            v=full_rod(a.workbook,a.task,np.column_stack((self.train.lab,self.train.x)),a.data)
            s=Direct.from_values(v,16)
            _,first,total,(lo,hi)=PROTOCOL[a.task]
            issue=np.arange(first,total-16)
            idx=issue[(issue+16<lo)|(issue+16>=hi)]
        elif a.task=="SO2":
            v=rows_h2s_workbook(a.workbook,10081)
            np.testing.assert_array_equal(v[:len(self.train)],np.c_[self.train.lab,self.train.x])
            s=Series(v[:,1:],v[:,0],v[:,0],8)
            idx=np.arange(self.first,len(s))
        else:
            task="tep" if self.tep else "sru"
            previous=read(a.model/"RESULTS.json")
            assert self.model_fit["data"] == receipt(a.data,task,("train","development"))
            train,dev=load_prefix(task,a.data,self.gap)
            assert self.first == len(train)+len(dev)
            if self.tep:
                assert digest(a.data/"tail.npz") == previous["source"]
                s=append(append(train,dev),read_tep(a.data,"tail",self.gap))
            else:
                assert digest(a.workbook) == previous["source"]
                s=full_h2s(train,dev,a.workbook)
            idx=np.arange(self.first,len(s))
        assert np.array_equal(idx[:512],np.arange(self.first,self.first+512))
        return s,idx


def pls_fit(x,y,n):
    scale=StandardScaler().fit(x)
    model=PLSRegression(n_components=n,scale=False,max_iter=500).fit(scale.transform(x),y)
    if len(model.n_iter_) != n:
        raise ValueError("Requested PLS rank did not fit")
    return scale,model


def pls_predict(arm,x):
    return arm[1].predict(arm[0].transform(x)).reshape(-1)


def train_pls(q,out):
    ids=q.rows
    x=q.linear(q.train,ids)
    y=(q.train.truth-q.train.anchor())[ids]
    sse=np.zeros(len(PLS_COMPONENTS))
    count=0
    for prefix,valid in q.folds:
        tr,ev=prefix-ids[0],valid-ids[0]
        count+=len(ev)
        for i,n in enumerate(PLS_COMPONENTS):
            error=pls_predict(pls_fit(x[tr],y[tr],n),x[ev])-y[ev]
            sse[i]+=error@error
    choice=PLS_COMPONENTS[int(np.argmin(sse))]
    (out/"DPLS.pkl").write_bytes(pickle.dumps(pls_fit(x,y,choice),protocol=5))
    return {"components":choice,"fold_pooled_mse":(sse/count).tolist(),"grid":PLS_COMPONENTS}


def gru(width,channels):
    from torch import nn
    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence=nn.GRU(channels,width,batch_first=True)
            self.output=nn.Linear(width,1)

        def forward(self,x):
            _,end=self.sequence(x)
            return self.output(end[-1]).squeeze(-1)
    return Readout()


def fit_gru(x,y,rows,width,epochs,seed,device,valid=None):
    import torch
    torch.manual_seed(seed)
    rng=torch.Generator(device=device).manual_seed(seed)
    features=x[rows]
    mean=features.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    std=np.maximum(features.std(axis=(0,1),dtype=np.float64),1e-6).astype(np.float32)
    ym,ys=float(y[rows].mean()),float(y[rows].std())
    v=torch.from_numpy((features-mean)/std).to(device)
    target=torch.as_tensor((y[rows]-ym)/ys,dtype=torch.float32,device=device)
    m=gru(width,x.shape[2]).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=GRU_LR)
    errors=[]
    if valid is not None:
        xv=torch.from_numpy((x[valid]-mean)/std).to(device)
        yv=torch.as_tensor(y[valid],device=device)
    for _ in range(epochs):
        m.train()
        for subset in torch.randperm(len(rows),device=device,generator=rng).split(GRU_BATCH):
            opt.zero_grad(set_to_none=True)
            (m(v[subset])-target[subset]).square().mean().backward()
            opt.step()
        if valid is not None:
            m.eval()
            with torch.inference_mode():
                e=torch.cat([m(h) for h in xv.split(GRU_BATCH)])*ys+ym-yv
                errors.append(float(e.square().sum()))
    return dict(width=width,mean=mean,std=std,ym=ym,ys=ys,
                state={k:v.cpu() for k,v in m.state_dict().items()}),errors


def train_gru(q,out):
    import torch
    torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    ids=q.rows
    x=q.sequence(q.train,ids)
    y=(q.train.truth-q.train.anchor())[ids]
    val_count=sum(len(v) for _,v in q.folds)
    grid=[]
    for width in GRU_WIDTH:
        trace=np.sum([fit_gru(x,y,a-ids[0],width,GRU_EPOCHS,0,device,b-ids[0])[1]
                      for a,b in q.folds],axis=0)/val_count
        grid.append(trace.tolist())
        print(q.a.task,width,"best_mse",float(trace.min()),flush=True)
    i,j=np.unravel_index(np.argmin(grid),(len(GRU_WIDTH),GRU_EPOCHS))
    epochs=int(j)+1
    for seed in q.seed:
        arm,_=fit_gru(x,y,np.arange(len(y)),GRU_WIDTH[i],epochs,seed,device)
        torch.save(arm,out/f"GRU_{seed}.pt")
    return dict(width=GRU_WIDTH[i],epochs=epochs,seed_list=q.seed,device=device,fold_pooled_mse=grid,
                width_grid=GRU_WIDTH,lr=GRU_LR,max_epochs=GRU_EPOCHS,batch_size=GRU_BATCH)


def load_gru(out,seeds):
    import torch
    torch.set_num_threads(2)
    frozen=[]
    for i in seeds:
        arm=torch.load(out/f"GRU_{i}.pt",weights_only=False,map_location="cpu")
        m=gru(arm["width"],len(arm["mean"])).eval()
        m.load_state_dict(arm["state"])
        frozen.append((m,arm))
    def run(x):
        pred=[]
        for m,a in frozen:
            features=torch.from_numpy((x-a["mean"])/a["std"])
            with torch.inference_mode():
                p=torch.cat([m(v) for v in features.split(GRU_BATCH)]).numpy()
            pred.append(a["ym"]+a["ys"]*p.astype(float))
        return np.array(pred)
    return run


def feature_clip(q,s,lo,hi):
    clipped=np.clip(s.x,lo,hi)
    if q.cz:
        return type(s).from_values(np.column_stack((s.lab,clipped)),16)
    return s.with_x(clipped)


def prior_linear_path(q):
    a=q.a
    if q.tep:
        return a.existing/"cpu/linear.pkl"
    if q.cz:
        return a.model/"Baselines"/f"CZ_{a.task}/linear/MODELS.pkl"
    if a.task=="SO2":
        return a.model/"classical/SRU_H2S/linear/MODELS.pkl"
    return a.paired/"SRU_H2S/linear/MODELS.pkl"


def replay_original(q):
    from laguerre_screen import digest
    a=q.a
    path=prior_linear_path(q)
    if q.tep:
        from tep_baselines import linear_predict,predict_bank
        lock=read(a.existing/"TEST_LOCK.json")
        assert lock["version"]==digest(a.lock)
        cpu=a.existing/"cpu/FIT.json"
        tf=a.existing/"transformer/FIT.json"
        assert digest(cpu)==lock["fits"]["cpu"] and digest(tf)==lock["fits"]["transformer"]
        for fit in (cpu,tf):
            assert all(digest(fit.parent/p)==v for p,v in read(fit)["weights"].items())
            for p,v in read(fit)["weights"].items():
                assert digest(a.existing/"bank"/p)==v
        linear,versions=load_pickle(path)
        def elast(s,t):
            return (s.anchor()[t]+linear_predict(linear["ElasticNet"],q.linear(s,t)))[None,:]
        return lambda s,t:predict_bank(s,t,a.existing/"bank"),elast,versions
    if q.cz:
        from laguerre_blind_cz import predictors
        assert digest(path)==read(a.model/"Baselines/FROZEN.json")["linear"]
        assert digest(a.arch)==read(a.model/"TEST_LOCK.json")["architecture"]
        run=predictors(a.task,a.model,a.protocol,a.existing,a.arch)
    elif a.task=="SO2":
        from laguerre_blind_baselines import predictors
        lock=read(a.model/"TEST_LOCK.json")
        for file,h in lock["files"].items():
            assert digest(a.model/file)==h
        run=predictors(a.archive,a.model)
    else:
        from laguerre_pressure import predictors
        run,_=predictors(a.archive,a.paired,a.existing)
        assert digest(path)==read(a.paired/"LOCK.json")["baselines"]
    linear,versions=load_pickle(path)
    def lin(name,s,t):
        anchor=s.anchor()[t]
        x=s.anchor()[t[:,None]-np.arange(q.history)] if name=="AR" else q.linear(s,t)
        values=[]
        for arm in linear[name]:
            m,mu,sd=arm["model"]
            z=m.predict(arm["scale"].transform(x))*sd+mu
            values.append(z+(anchor if arm["recipe"]=="delta" else 0.))
        return np.array(values)
    def all_(s,t):
        bank=run(np.c_[s.lab,s.x],t)
        if a.task=="H2S":
            bank.update(Persistence=s.anchor()[t][None,:],AR=lin("AR",s,t))
        return bank
    return all_,lambda s,t:lin("ElasticNet",s,t),versions


def fault(q,s):
    scale=4.*q.train.x.std(axis=0)
    sign=np.random.default_rng(OSC_SEED).choice((-1,1),256)
    matrices={"clean":s}
    for label,change in (("oscillation",np.c_[sign,-sign].ravel()),("bias",np.ones(512))):
        x=s.x.copy()
        x[q.first:q.first+512]+=scale*change[:,None]
        matrices[label]=type(s).from_values(np.c_[s.lab,x],16) if q.cz else s.with_x(x)
    return matrices


def dm(gain,rows,delay):
    n=len(gain)
    lag=min(n-1,max(delay-1,int(4*(n/100)**(2/9))))
    z=gain-gain.mean()
    spans=np.split(z,np.flatnonzero(np.diff(rows)>1)+1)
    estimate=z@z
    for k in range(1,lag+1):
        estimate+=2*(1-k/(lag+1))*sum(b[k:]@b[:-k] for b in spans)
    se=float(np.sqrt(max(0.,estimate))/n)
    t=float(gain.mean()/se) if se else None
    return dict(n=n,bartlett_lag=lag,mean_loss_gain=float(gain.mean()),t=t,
                log_p=float(np.log(2)+log_ndtr(-abs(t))) if t is not None else None)


def holm(tests):
    records=sorted((v["log_p"],name) for name,v in tests.items() if v["log_p"] is not None)
    previous=-np.inf
    for i,(p,name) in enumerate(records):
        previous=max(previous,min(0.,p+np.log(len(records)-i)))
        tests[name]["p_holm"]=float(np.exp(previous))
        tests[name]["log10_p_holm"]=float(previous/np.log(10))


def score(q):
    from laguerre_screen import digest,save_json
    from laguerre_unified import pressure_metrics
    a=q.a
    registration=read(a.out/"REGISTRATION.json")
    assert registration["archive_model"]==q.model_digest and registration["source_lock"]==digest(a.lock)
    assert registration["script"]==digest(Path(__file__))
    fit=read(a.out/"FIT.json")
    assert fit["registration"]==digest(a.out/"REGISTRATION.json")
    for name,h in fit["weights"].items():
        assert digest(a.out/name)==h
    assert digest(q.mode_dir/"MODEL.pkl")==q.model_fit["model"]
    assert not (a.out/"RESULTS.json").exists()
    limits=registration["clipping"]
    lo,hi=np.array(limits["lower"]),np.array(limits["upper"])
    np.testing.assert_array_equal([lo,hi],np.quantile(q.train.x,[.01,.99],axis=0))
    original,elastic,versions=replay_original(q)
    save_json(a.out/"TEST_LOCK.json",dict(registration=digest(a.out/"REGISTRATION.json"),
              fit=digest(a.out/"FIT.json"),archive_elastic=digest(prior_linear_path(q))))
    series,rows=q.all_data()
    model=pickle.loads((q.mode_dir/"MODEL.pkl").read_bytes())
    pls=pickle.loads((a.out/"DPLS.pkl").read_bytes())
    nn=load_gru(a.out,q.seed)
    clean,panels={},{}
    for label,s in fault(q,series).items():
        ts=rows if label=="clean" else rows[:512]
        forecast=model.predict(s)
        observed=series.truth[ts]
        anchor=s.anchor()[ts]
        bank=original(s,ts)
        np.testing.assert_allclose(elastic(s,ts),bank["ElasticNet"],rtol=0.,atol=1e-12)
        bank.update({"PRISM":forecast["prediction"][ts][None,:],
                    "ElasticNet_clipped":elastic(feature_clip(q,s,lo,hi),ts),
                    "DPLS":(anchor+pls_predict(pls,q.linear(s,ts)))[None,:],
                    "GRU":anchor+nn(q.sequence(s,ts))})
        assert len(bank)==10
        if label=="clean":
            clean=bank
        stats,tests={},{}
        for name,vs in bank.items():
            protected=name=="PRISM"
            z=vs.mean(axis=0)
            ref=clean[name][:,:len(ts)].mean(axis=0)
            stats[name]=pressure_metrics(z,ref,observed,forecast["endo"][ts],protected,
                                        forecast["trust"][ts] if protected else None)
            errors=vs-observed
            stats[name].update(mean_seed_rmse=float(np.sqrt(np.mean(errors**2,axis=1)).mean()),
                               mean_seed_mae=float(abs(errors).mean()),seeds=len(vs))
            if not protected:
                gain=np.mean(errors**2,axis=0)-(bank["PRISM"][0]-observed)**2
                tests[name]=dm(gain,ts,q.gap)
        holm(tests)
        panels[label]=dict(metrics=stats,dm=tests,count=len(ts),first_issue=int(ts[0]),
                           stop_issue=int(ts[-1]+1),issue_sha256=issue_digest(ts))
    check_original(q,panels)
    save_json(a.out/"RESULTS.json",dict(case=a.task,commit=q.lock["commit"],
              score_lock=digest(a.out/"TEST_LOCK.json"),baseline_version_warnings=versions,panels=panels))
    print(a.task,"closed",flush=True)


def issue_digest(rows):
    from hashlib import sha256
    return sha256(rows.astype("<i8").tobytes()).hexdigest()


def check_original(q,panels):
    a=q.a
    if q.cz:
        archive=read(a.previous)
        expected={k:v["rmse"] for k,v in archive["h16"].items()}
        records=panels["clean"]["metrics"]
        field="mean_seed_rmse"
    elif q.tep:
        archive=read(a.existing/"RESULTS.json")
        for case,panel in panels.items():
            for name,previous in archive["tests"][case]["metrics"].items():
                np.testing.assert_allclose(panel["metrics"][name]["rmse"],previous["rmse"],rtol=0.,atol=2e-9)
        return
    elif a.task=="SO2":
        archive=read(a.model/"RESULTS.json")
        expected={k:v["rmse"] for k,v in archive["clean"].items()}
        records=panels["clean"]["metrics"]
        field="mean_seed_rmse"
    else:
        archive=read(a.model/"RESULTS.json")
        expected={k:v["mean_seed_rmse"] for k,v in archive["pressure"]["clean_dm"].items()}
        expected["PRISM"]=archive["tail_rmse"]
        records=panels["clean"]["metrics"]
        field="mean_seed_rmse"
    for model,value in expected.items():
        np.testing.assert_allclose(records[model][field],value,rtol=0.,atol=2e-9)
    stress=archive.get("stress",archive.get("pressure",{}).get("stress",{}))
    for case,saved in stress.items():
        for model,row in saved.items():
            name="PRISM" if model=="DualARX" else model
            np.testing.assert_allclose(panels[case]["metrics"][name]["rmse"],row["rmse"],rtol=0.,atol=2e-9)


def fit(q):
    from laguerre_screen import digest,save_json
    a=q.a
    a.out.mkdir(parents=True,exist_ok=False)
    lower,upper=np.quantile(q.train.x,[.01,.99],axis=0)
    save_json(a.out/"REGISTRATION.json",dict(case=a.task,script=digest(Path(__file__)),
              source_lock=digest(a.lock),archive_model=q.model_digest,
              training_issues=[int(q.rows[0]),int(q.rows[-1]+1)],
              folds=[([int(t[0]),int(t[-1]+1)],[int(v[0]),int(v[-1]+1)]) for t,v in q.folds],
              label_span=q.gap,tail_issue=q.first,
              clipping=dict(quantile=[.01,.99],lower=lower.tolist(),upper=upper.tolist(),
                            raw_sensors_only=True,all_issued_input_history=True)))
    dpls=train_pls(q,a.out)
    print(a.task,"DPLS",dpls["components"],flush=True)
    neural=train_gru(q,a.out)
    weights={p.name:digest(p) for p in sorted(a.out.iterdir()) if p.suffix in (".pt",".pkl")}
    save_json(a.out/"FIT.json",dict(registration=digest(a.out/"REGISTRATION.json"),
              dpls=dpls,gru=neural,weights=weights))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit","score"))
    p.add_argument("task",choices=CASES)
    for n in ("source","lock","model","data","out"):
        p.add_argument(n,type=Path)
    for n in ("workbook","existing","archive","paired","protocol","arch","previous","tep-code"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--tep-helper-digest")
    a=p.parse_args()
    sys.path.insert(0,str(a.source))
    if a.tep_code:
        sys.path.insert(1,str(a.tep_code))
    protocol=Protocol(a)
    protocol.verify_source()
    {"fit":fit,"score":score}[a.action](protocol)
