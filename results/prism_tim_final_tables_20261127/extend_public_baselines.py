"""Extend public DPLS/GRU selection using training folds and seal before replay."""

import argparse
import copy
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import pinv
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


PLS_GRID=(2,4,8,12,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320)
PLS_CAPS=(96,160,256,320)
GRU_WIDTHS=(24,48,64,96,128,160,192,256,384)
INITIAL_WIDTHS=4
EPOCH_CAP=96
PATIENCE=9
MIN_EPOCH=12


def path_losses(x,y,fit,valid,candidates):
    scale=StandardScaler().fit(x[fit])
    a,b=scale.transform(x[fit]),scale.transform(x[valid])
    model=PLSRegression(n_components=candidates[-1],scale=False,max_iter=500).fit(a,y[fit])
    if len(model.n_iter_)!=candidates[-1]:
        raise RuntimeError("DPLS path ended before its requested rank")
    xc,yc=a.mean(axis=0),float(y[fit].mean())
    scores=[]
    for rank in candidates:
        w=model.x_weights_[:,:rank]
        p=model.x_loadings_[:,:rank]
        beta=w@pinv(p.T@w)@model.y_loadings_[0,:rank]
        estimate=(b-xc)@beta+yc
        if rank==candidates[-1]:
            np.testing.assert_allclose(estimate,model.predict(b).ravel(),rtol=0.,atol=1e-10)
        elif rank==24:
            direct=PLSRegression(n_components=rank,scale=False,max_iter=500).fit(a,y[fit])
            np.testing.assert_allclose(estimate,direct.predict(b).ravel(),rtol=0.,atol=1e-10)
        e=estimate-y[valid]
        scores.append(float(e@e))
    return np.array(scores)


def dpls(q,dst,old):
    first=q.rows[0]
    design=q.linear(q.train,q.rows)
    target=(q.train.truth-q.train.anchor())[q.rows]
    folds=[(a-first,b-first) for a,b in q.folds]
    used=[]
    for cap in PLS_CAPS:
        grid=[rank for rank in PLS_GRID if rank<=cap]
        loss=np.sum([path_losses(design,target,a,b,grid) for a,b in folds],axis=0)/sum(len(b) for a,b in folds)
        peak=int(np.argmin(loss))
        used.append(cap)
        print(q.a.task,"DPLS",cap,grid[peak],float(loss[peak]),flush=True)
        if peak+2<len(grid):
            break
    scale=StandardScaler().fit(design)
    arm=PLSRegression(n_components=grid[peak],scale=False,max_iter=500).fit(scale.transform(design),target)
    (dst/"DPLS.pkl").write_bytes(pickle.dumps((scale,arm),protocol=5))
    return dict(components=grid[peak],tried=grid,expanded_caps=used,pooled_mse=loss.tolist(),
                prior=old["dpls"]["components"],upper_boundary=peak==len(grid)-1,
                ranks_above=len(grid)-peak-1,gain_vs_prior_rank=float(loss[grid.index(24)]-loss[peak]))


class NeuralFold:
    def __init__(self,torch,baseline,x,y,fit,valid,width,seed,device):
        self.torch=torch
        self.batch=baseline.GRU_BATCH
        self.device=device
        features=x[fit]
        self.center=features.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
        self.scale=np.maximum(features.std(axis=(0,1),dtype=np.float64),1e-6).astype(np.float32)
        self.ymean,self.yscale=float(y[fit].mean()),float(y[fit].std())
        self.x=torch.from_numpy((features-self.center)/self.scale).to(device)
        self.y=torch.as_tensor((y[fit]-self.ymean)/self.yscale,dtype=torch.float32,device=device)
        self.v=torch.from_numpy((x[valid]-self.center)/self.scale).to(device)
        self.actual=torch.as_tensor(y[valid],device=device)
        torch.manual_seed(seed)
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.model=baseline.gru(width,x.shape[2]).to(device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=baseline.GRU_LR)

    def epoch(self):
        torch=self.torch
        self.model.train()
        perm=torch.randperm(len(self.y),device=self.device,generator=self.generator)
        for rows in perm.split(self.batch):
            self.opt.zero_grad(set_to_none=True)
            (self.model(self.x[rows])-self.y[rows]).square().mean().backward()
            self.opt.step()
        self.model.eval()
        with torch.inference_mode():
            error=torch.cat([self.model(v) for v in self.v.split(self.batch)])*self.yscale+self.ymean-self.actual
        return float(error.square().sum())


def recurrent(q,dst,old,baseline):
    import torch
    torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    ids=q.rows
    seq=q.sequence(q.train,ids)
    target=(q.train.truth-q.train.anchor())[ids]
    fold_pairs=[(t-ids[0],v-ids[0]) for t,v in q.folds]
    validation=sum(len(v) for _,v in fold_pairs)
    traces={}
    contenders=list(GRU_WIDTHS[:INITIAL_WIDTHS])
    for width in contenders:
        members=[NeuralFold(torch,baseline,seq,target,t,v,width,0,device) for t,v in fold_pairs]
        trace=[]
        priority=float("inf")
        last=0
        for epoch in range(1,EPOCH_CAP+1):
            current=sum(m.epoch() for m in members)/validation
            trace.append(current)
            if current < priority*(1.-1e-4):
                priority=current
                last=epoch
            if epoch>=MIN_EPOCH and epoch-last>=PATIENCE:
                break
        del members
        traces[width]=trace
        best=min(traces,key=lambda w:min(traces[w]))
        print(q.a.task,"GRU",width,"best epoch",int(np.argmin(trace))+1,
              "trained",len(trace),"mse",min(trace),flush=True)
        if width==contenders[-1] and best==width and len(contenders)<len(GRU_WIDTHS):
            contenders.append(GRU_WIDTHS[len(contenders)])
    width=min(traces,key=lambda w:min(traces[w]))
    epoch=int(np.argmin(traces[width]))+1
    for seed in q.seed:
        model,_=baseline.fit_gru(seq,target,np.arange(len(target)),width,epoch,seed,device)
        torch.save(model,dst/f"GRU_{seed}.pt")
    return dict(width=width,epochs=epoch,epochs_evaluated=len(traces[width]),grid=contenders,traces=traces,
                patience=PATIENCE,epoch_cap=EPOCH_CAP,min_epoch=MIN_EPOCH,device=device,
                width_upper_boundary=width==max(contenders),stop_at_epoch_cap=len(traces[width])==EPOCH_CAP,
                optimum_at_last_epoch=epoch==len(traces[width]),seeds=list(q.seed),
                previous_width=old["gru"]["width"],previous_epochs=old["gru"]["epochs"])


def fit(a,q,baseline):
    from laguerre_screen import digest,save_json
    a.out.mkdir(parents=True,exist_ok=False)
    record=dict(case=a.task,commit=q.lock["commit"],study=digest(Path(__file__)),
                source_lock=digest(a.lock),frozen_fit=q.model_digest,
                parent_result=digest(a.archived/"RESULTS.json"),parent_fit=digest(a.archived/"FIT.json"),
                baseline_script=digest(Path(baseline.__file__)),training_issues=[int(q.rows[0]),int(q.rows[-1])+1])
    save_json(a.out/"REGISTRATION.json",record)
    old=baseline.read(a.archived/"FIT.json")
    if a.action=="fit-dpls":
        decision=dpls(q,a.out,old)
    else:
        decision=recurrent(q,a.out,old,baseline)
    model_files={p.name:digest(p) for p in a.out.iterdir() if p.suffix in (".pkl",".pt")}
    save_json(a.out/"FIT.json",dict(registration=digest(a.out/"REGISTRATION.json"),baseline=a.action[4:],
              fit=decision,weights=model_files))


def revised(a,q,baseline):
    from laguerre_screen import digest,save_json
    from laguerre_unified import pressure_metrics
    old=baseline.read(a.archived/"RESULTS.json")
    verified={}
    for part in ("dpls","gru"):
        base=a.out/part
        reg=baseline.read(base/"REGISTRATION.json")
        fit=baseline.read(base/"FIT.json")
        assert stable_registration(a,q,baseline,reg)
        assert fit["registration"]==digest(base/"REGISTRATION.json")
        assert all(digest(base/n)==v for n,v in fit["weights"].items())
        verified[part]=digest(base/"FIT.json")
    assert digest(q.mode_dir/"MODEL.pkl")==q.model_fit["model"]
    assert not (a.out/"RESULTS.json").exists()
    save_json(a.out/"TEST_LOCK.json",dict(fits=verified,source_lock=digest(a.lock),
                                        old_result=digest(a.archived/"RESULTS.json"),model=q.model_fit["model"]))
    full,rows=q.all_data()
    y=full.truth[rows]
    model=pickle.loads((q.mode_dir/"MODEL.pkl").read_bytes())
    dpls_arm=pickle.loads((a.out/"dpls/DPLS.pkl").read_bytes())
    gru_runner=baseline.load_gru(a.out/"gru",q.seed)
    panels=copy.deepcopy(old["panels"])
    clean={}
    for name,s in baseline.fault(q,full).items():
        panel=panels[name]
        t=rows if name=="clean" else rows[:512]
        assert panel["issue_sha256"]==baseline.issue_digest(t)
        arc=model.predict(s)
        truth=full.truth[t]
        pr=arc["prediction"][t]
        np.testing.assert_allclose(np.sqrt(np.mean((pr-truth)**2)),panel["metrics"]["PRISM"]["rmse"],rtol=0.,atol=2e-12)
        np.testing.assert_allclose(np.mean(abs(pr-truth)),panel["metrics"]["PRISM"]["mae"],rtol=0.,atol=2e-12)
        anchor=s.anchor()[t]
        predictions={
            "DPLS":(anchor+baseline.pls_predict(dpls_arm,q.linear(s,t)))[None,:],
            "GRU":anchor+gru_runner(q.sequence(s,t)),
        }
        if name=="clean":
            clean=predictions
        original_loss=(pr-truth)**2
        for label,values in predictions.items():
            error=values-truth
            metrics=pressure_metrics(values.mean(axis=0),clean[label][:,:len(t)].mean(axis=0),truth,
                                     arc["endo"][t],False)
            metrics.update(mean_seed_rmse=float(np.sqrt(np.mean(error**2,axis=1)).mean()),
                           mean_seed_mae=float(np.mean(abs(error))),seeds=len(values))
            panel["metrics"][label]=metrics
            panel["dm"][label]=baseline.dm(np.mean(error**2,axis=0)-original_loss,t,q.gap)
        assert len(panel["dm"])==9
        baseline.holm(panel["dm"])
    save_json(a.out/"RESULTS.json",dict(case=a.task,commit=q.lock["commit"],panels=panels,
             tail_variance=float(y.var()),clean_count=len(y),frozen_replay="validated against archived RMSE and MAE",
             test_lock=digest(a.out/"TEST_LOCK.json")))


def stable_registration(a,q,baseline,reg):
    from laguerre_screen import digest
    return (reg["source_lock"]==digest(a.lock) and reg["study"]==digest(Path(__file__))
            and reg["baseline_script"]==digest(Path(baseline.__file__))
            and reg["parent_result"]==digest(a.archived/"RESULTS.json")
            and reg["parent_fit"]==digest(a.archived/"FIT.json") and reg["frozen_fit"]==q.model_digest)


def cz_variance(a,q,baseline):
    from laguerre_screen import digest,save_json
    archive=baseline.read(a.archived/"RESULTS.json")
    s,rows=q.all_data()
    assert archive["panels"]["clean"]["issue_sha256"]==baseline.issue_digest(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    save_json(a.out/"RESULTS.json",dict(case=a.task,commit=q.lock["commit"],panels=archive["panels"],
              tail_variance=float(s.truth[rows].var()),clean_count=len(rows),
              preserved_result=digest(a.archived/"RESULTS.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit-dpls","fit-gru","score","cz-variance"))
    p.add_argument("task",choices=("TEP_D1","TEP_D2","H2S","SO2","Rod1","Rod2"))
    for arg in ("source","lock","model","data","out","archived","baseline_code"):
        p.add_argument(arg,type=Path)
    for arg in ("workbook","tep-code"):
        p.add_argument("--"+arg,type=Path)
    p.add_argument("--tep-helper-digest")
    a=p.parse_args()
    sys.path[:0]=[str(a.source),str(a.baseline_code)]
    if a.tep_code:
        sys.path.insert(2,str(a.tep_code))
    import benchmark_addendum as baseline
    q=baseline.Protocol(a)
    q.verify_source()
    if a.action.startswith("fit-"):
        if q.cz:
            raise ValueError("CZ baseline fits remain frozen")
        fit(a,q,baseline)
    elif a.action=="score":
        revised(a,q,baseline)
    else:
        if not q.cz:
            raise ValueError("Variance-only replay is for frozen CZ arms")
        cz_variance(a,q,baseline)
