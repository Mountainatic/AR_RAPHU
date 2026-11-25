"""Matched tail baselines and autocorrelation-aware paired tests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import sys

from scipy import stats

import core as c
from linear import design

sys.path.insert(0,str(c.ROOT/"vendor"))
import lightgbm as lgb


ROOT=c.ROOT/"E1"
GRID=tuple(product(("level","delta"),((7,15),(15,30),(31,60)),(150,400),(.1,10.)))
SEEDS=tuple(range(5))


def register():
    c.x.bind(ROOT/"REGISTRATION.json",dict(code_lock=c.verify_code(),runner=c.sha(Path(__file__)),
        entities=c.MAIN,models=("Persistence","AR","ARX","ElasticNet","LightGBM","Transformer","PRISM"),
        lightgbm=dict(version=lgb.__version__,space=GRID,seeds=SEEDS,
            fixed=dict(learning_rate=.04,subsample=.85,colsample_bytree=.75),folds="Existing two causal expanding folds"),
        DM=dict(loss="Baseline seed-average squared loss minus frozen PRISM squared loss",
            HAC="Bartlett: max(H+D-1, floor(4*(N/100)^(2/9)))",
            reference="Two-sided asymptotic standard normal",multiple_testing="Holm within each dataset")))


def head(config,seed):
    _,(leaves,minimum),rounds,penalty=config
    return lgb.LGBMRegressor(num_leaves=leaves,min_child_samples=minimum,n_estimators=rounds,
        reg_lambda=penalty,learning_rate=.04,subsample=.85,subsample_freq=1,colsample_bytree=.75,
        n_jobs=1,deterministic=True,force_col_wise=True,random_state=seed,verbosity=-1)


def fit(name):
    register()
    q=c.x.TASKS[name]
    a=c.data(q)
    dst=ROOT/name/"LightGBM"
    dst.mkdir(parents=True,exist_ok=True)
    folds=[(tr,ev,design(q,a,tr,"ARX"),design(q,a,ev,"ARX")) for tr,ev in c.x.folds(q)]

    def trial(config):
        runs=[]
        for tr,ev,X,Z in folds:
            y,_=q.target(a,tr,config[0])
            reference,_=q.target(a,ev,config[0])
            model=head(config,20261125).fit(X,y)
            residual=model.booster_.predict(Z,num_threads=1)-reference
            runs.append((float(residual@residual),len(residual)))
        return dict(config=config,MSE=sum(v for v,_ in runs)/sum(n for _,n in runs))

    with ThreadPoolExecutor(max_workers=4) as pool:
        risks=list(pool.map(trial,GRID))
    selected=min(risks,key=lambda row:row["MSE"])
    config=selected["config"]
    t=q.train_origins()
    X=design(q,a,t,"ARX")
    y,_=q.target(a,t,config[0])
    files={}
    for seed in SEEDS:
        model=head(config,seed).fit(X,y)
        path=dst/f"seed_{seed}.txt"
        model.booster_.save_model(str(path))
        files[path.name]=c.sha(path)
    c.x.bind(dst/"FROZEN.json",dict(registration=c.sha(ROOT/"REGISTRATION.json"),selected=selected,
        search=risks,files=files,last_fit_target=int(t[-1])+q.H))
    print(name,selected,flush=True)


def lgb_predict(q,a,t):
    dst=ROOT/q.name/"LightGBM"
    lock=c.read(dst/"FROZEN.json")
    assert lock["registration"]==c.sha(ROOT/"REGISTRATION.json")
    cut=(q.dev+q.total-q.H)//2
    pre=t[t<=cut]
    X=design(q,a,t,"ARX")
    poisoned=q.poison(a,cut)
    c.np.testing.assert_array_equal(X[:len(pre)],design(q,poisoned,pre,"ARX"))
    delta=lock["selected"]["config"][0]=="delta"
    result=[]
    for file,digest in lock["files"].items():
        model_path=dst/file
        assert c.sha(model_path)==digest
        model=lgb.Booster(model_file=str(model_path))
        pred=model.predict(X,num_threads=1)+(a[t-q.D,0] if delta else 0.)
        trial=model.predict(X[:len(pre)],num_threads=1)+(poisoned[pre-q.D,0] if delta else 0.)
        c.np.testing.assert_array_equal(pred[:len(pre)],trial)
        result.append(pred)
    return result


def dm(loss,lag):
    n=len(loss)
    estimate=float(loss.mean())
    u=loss-estimate
    hac=float(u@u/n)
    for k in range(1,lag+1):
        hac+=2*(1-k/(lag+1))*float(u[k:]@u[:-k]/n)
    se=float(c.np.sqrt(max(hac,0.)/n))
    stat=estimate/se if se else None
    return dict(n=n,lag=lag,mean_baseline_minus_PRISM_MSE=estimate,standard_error=se,
                DM=stat,p=float(2*stats.norm.sf(abs(stat))) if stat is not None else None)


def holm(rows):
    last=0.
    valid=sorted((r for r in rows if r["p"] is not None),key=lambda x:x["p"])
    for i,row in enumerate(valid):
        last=max(last,min(1.,row["p"]*(len(valid)-i)))
        row["p_Holm"]=last


def evaluation(name):
    q,k,fit,chain,_=c.load(name)
    a=c.data(q,full=True)
    t,masks=c.rows(q)
    py,numerical=c.cert_pair(q,a,k,fit,chain,t)
    dst=c.SUITE/name
    with c.np.load(dst/"linear/PREDICTIONS.npz") as linear:
        c.np.testing.assert_array_equal(linear["origins"],t)
        results={m:[linear[m]] for m in ("Persistence","AR","ARX")}
        results["ElasticNet"]=[linear[f"ElasticNet_{s}"] for s in SEEDS]
    results["LightGBM"]=lgb_predict(q,a,t)
    neural=dst/"neural/PREDICTIONS.npz" if q.H else ROOT/"Transformer/PREDICTIONS.npz"
    with c.np.load(neural) as archive:
        c.np.testing.assert_array_equal(archive["origins"],t)
        prefix="Transformer" if q.H else "seed"
        results["Transformer"]=[archive[f"{prefix}_{s}"] for s in SEEDS]
    results["PRISM"]=[py["pred"]]
    tail=masks["tail"]
    truth=a[t[tail]+q.H,0]
    baseline=(results["PRISM"][0][tail]-truth)**2
    lag=max(q.mature-1,int(4*(len(truth)/100)**(2/9)))
    comparisons=[]
    for method,pred in results.items():
        if method=="PRISM":
            continue
        u=c.np.mean([(z[tail]-truth)**2 for z in pred],axis=0)-baseline
        comparisons.append(dict(data=name,baseline=method,**dm(u,lag),
                     sensitivity=dm(u,q.L-1),p_Holm=None))
    holm(comparisons)
    return dict(data=name,protocol=[q.H,q.D,q.L],python_C_max_abs=numerical,
        scores={key:dict(n=int(mask.sum()),models=c.summarize(q,a,t,results,mask))
                for key,mask in masks.items()},comparisons=comparisons,
        test_target_ranges={key:[int(t[mask][0])+q.H,int(t[mask][-1])+q.H+1] for key,mask in masks.items()})


def evaluate():
    register()
    entries=[evaluation(name) for name in c.MAIN]
    scores=[dict(data=e["data"],split=split,N=s["n"],method=m,repeats=len(v["runs"]),
        **v["mean"],**{key+"_sd":std for key,std in v["sd"].items()})
       for e in entries for split,s in e["scores"].items() for m,v in s["models"].items()]
    c.csv_table(ROOT/"METRICS.csv",scores)
    c.csv_table(ROOT/"DM.csv",[{k:v for k,v in r.items() if k!="sensitivity"}|
            {"sensitivity_lag":r["sensitivity"]["lag"],"sensitivity_p":r["sensitivity"]["p"]}
          for e in entries for r in e["comparisons"]])
    c.write(ROOT/"EVIDENCE.json",entries)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("fit","evaluate"))
    args=cli.parse_args()
    if args.action=="fit":
        for name in c.MAIN:
            fit(name)
    else:
        evaluate()
