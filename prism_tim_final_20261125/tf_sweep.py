"""Train-only expanded Transformer resweep for SRU H2S."""

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from multiprocessing import get_context
from statistics import median
from pathlib import Path

import torch

import core as c


Q=c.x.TASKS["SRU_H2S"]
ROOT=c.ROOT/"E1"/"Transformer"
ARCH=c.BASE/"PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025"/"baseline_models.py"
GRID=tuple(product((32,64,128),(.0001,.0003,.001),("level","delta")))
SEEDS=(0,1,2,3,4)
SCREEN=3
REFINE=8


def register():
    c.x.bind(ROOT/"REGISTRATION.json",dict(code_lock=c.verify_code(),source_sha256=c.sha(Path(__file__)),
        architecture_sha256=c.sha(ARCH),grid=GRID,screen_epochs=SCREEN,refine_epochs=REFINE,
        refine_candidates=3,folds="70–85% and 85–100%, causal history embargo",
        seeds=SEEDS,selection_seed=20261023,cpu="FP32 deterministic",batch=256,
        protocol=dict(H=Q.H,D=Q.D,L=Q.L)))


def init():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


@lru_cache(maxsize=1)
def model_class():
    assert c.sha(ARCH)==c.read(ROOT/"REGISTRATION.json")["architecture_sha256"]
    spec=importlib.util.spec_from_file_location("sru_architecture",ARCH)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build


def features(a,t):
    idx=t[:,None]-c.np.arange(Q.L-1,-1,-1)
    assert idx.min()>=Q.D
    return c.np.concatenate((a[idx,1:],a[idx-Q.D,0,None]),axis=2).astype(c.np.float32)


@lru_cache(maxsize=6)
def partition(fold,recipe):
    a=c.data(Q)
    tr,ev=list(c.x.folds(Q))[fold] if fold is not None else (Q.train_origins(),None)
    def tensor(t,stats=None):
        X=features(a,t)
        y,anchor=Q.target(a,t,recipe)
        if stats is None:
            stats=(X.mean((0,1),dtype=c.np.float64).astype(c.np.float32),
                c.np.maximum(X.std((0,1),dtype=c.np.float64),1e-6).astype(c.np.float32),
                float(y.mean()),max(float(y.std()),1e-6))
        return torch.from_numpy((X-stats[0])/stats[1]),torch.from_numpy(((y-stats[2])/stats[3]).astype(c.np.float32)),anchor,stats
    train=tensor(tr)
    test=tensor(ev,train[-1]) if ev is not None else None
    return train,test


def fit(cfg,seed,fold,epochs):
    width,lr,recipe=cfg
    torch.manual_seed(seed)
    c.np.random.seed(seed)
    model=model_class()("Transformer",1+len(Q.channels),width,Q.L)
    (X,Y,_,stats),other=partition(fold,recipe)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    trace=[]
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(X)).split(256):
            opt.zero_grad(set_to_none=True)
            loss=(model(X[batch])-Y[batch]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if other is not None:
            z,y,_,_=other
            model.eval()
            with torch.inference_mode():
                pred=torch.cat([model(v) for v in z.split(128)])
            trace.append(float((pred-y).double().square().mean())*(stats[3]**2))
    if other is not None:
        ix=int(c.np.argmin(trace))
        return dict(MSE=trace[ix],epoch=ix+1,N=len(other[1]),trace=trace)
    return dict(weight=model.state_dict(),stats=stats,config=cfg,seed=seed,epochs=epochs)


def select_fold(job):
    stage,idx,cfg,fold,n=job
    output=ROOT/"selection"/f"{stage}_{idx}_{fold}.json"
    if output.exists():
        row=c.read(output)
        assert row["config"]==list(cfg)
    else:
        row=dict(config=cfg,**fit(cfg,20261023,fold,n))
        c.write(output,row)
    print(stage,idx,fold,row["MSE"],flush=True)
    return idx,fold,row


def select(pool,grid,stage,n):
    result=[[] for _ in grid]
    jobs=[(stage,i,cfg,j,n) for i,cfg in enumerate(grid) for j in range(2)]
    for index,fold,r in pool.map(select_fold,jobs):
        result[index].append((fold,r))
    scores=[]
    for i,vals in enumerate(result):
        folds=[r for _,r in sorted(vals)]
        scores.append(dict(config=list(grid[i]),index=i,folds=folds,
          MSE=float(c.np.average([v["MSE"] for v in folds],weights=[v["N"] for v in folds]))))
    c.write(ROOT/(stage.upper()+".json"),scores)
    return sorted(scores,key=lambda r:(r["MSE"],r["index"]))


def refit(job):
    cfg,n,seed=job
    path=ROOT/f"seed_{seed}.pt"
    tmp=path.with_suffix(".tmp")
    torch.save(fit(cfg,seed,None,n),tmp)
    tmp.replace(path)
    print("FINAL",seed,n,flush=True)
    return path.name,c.sha(path)


def train():
    register()
    with ProcessPoolExecutor(max_workers=12,mp_context=get_context("spawn"),initializer=init) as pool:
        winners=select(pool,GRID,"screen",SCREEN)
        chosen=select(pool,[tuple(r["config"]) for r in winners[:3]],"refine",REFINE)[0]
        n=round(median(r["epoch"] for r in chosen["folds"]))
        print("SELECTED",chosen["config"],n,flush=True)
        files=dict(pool.map(refit,[(chosen["config"],n,z) for z in SEEDS]))
    c.x.bind(ROOT/"FROZEN.json",dict(registration=c.sha(ROOT/"REGISTRATION.json"),selected=chosen,
        epochs=n,files=files,torch=torch.__version__,last_fit_label=int(Q.train_origins()[-1])))


def evaluate():
    register()
    init()
    locked=c.read(ROOT/"FROZEN.json")
    assert locked["registration"]==c.sha(ROOT/"REGISTRATION.json")
    assert all(c.sha(ROOT/p)==h for p,h in locked["files"].items())
    a=c.data(Q,full=True)
    t=Q.test_origins()
    cut=(Q.dev+Q.total)//2
    pre=t[t<=cut]
    x,y=features(a,t),features(Q.poison(a,cut),pre)
    c.np.testing.assert_array_equal(x[:len(pre)],y)
    scores,checks={},{}
    for seed in SEEDS:
        model=torch.load(ROOT/f"seed_{seed}.pt",map_location="cpu",weights_only=False)
        width,_,recipe=model["config"]
        net=model_class()("Transformer",1+len(Q.channels),width,Q.L).eval()
        net.load_state_dict(model["weight"])
        stats=model["stats"]
        def forward(z,t):
            with torch.inference_mode():
                b=torch.from_numpy((z-stats[0])/stats[1])
                pred=torch.cat([net(x) for x in b.split(64)]).numpy().astype(c.np.float64)
            return pred*stats[3]+stats[2]+(a[t-Q.D,0] if recipe=="delta" else 0.)
        p=forward(x,t)
        d=forward(y,pre)
        checks[str(seed)]=float(c.np.max(abs(p[:len(pre)]-d)))
        assert checks[str(seed)]<5e-7
        scores[f"seed_{seed}"]=p
    c.np.savez(ROOT/"PREDICTIONS.npz",origins=t,**scores)
    c.write(ROOT/"CAUSAL.json",dict(cut=int(cut),max_abs=checks))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("train","evaluate"))
    {"train":train,"evaluate":evaluate}[cli.parse_args().action]()
