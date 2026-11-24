"""Two chronological training folds and five refits of six classical regressors."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import subprocess
import sys
import time
import warnings

from common import (CLASSICAL, ENTITIES, ROOT, SEEDS, design, development, matched, np, poison,
                    read, registration, run_result, sha, target, train_rows, write)
import joblib
from sklearn import __version__ as sklearn_version
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

DEST = ROOT / "classical"
GRID = {
    "PCR": [dict(n_components=k) for k in (8,24,64,128)],
    "PLS": [dict(n_components=k) for k in (4,12,32)],
    "ElasticNet": [dict(alpha=a,l1_ratio=r) for a,r in product((.003,.03,.3),(.2,.8))],
    "RBF_SVR": [dict(gamma_scale=g,C=c) for g,c in ((.25,3.),(.5,10.),(1.,3.),(1.,10.),(2.,3.))],
    "RandomForest": [dict(max_features=f,min_samples_leaf=m,max_depth=d) for f,m,d in
                     (("sqrt",3,16),("sqrt",10,None),(.12,5,14))],
    "HistGradientBoosting": [dict(max_leaf_nodes=k,max_features=f,learning_rate=l2) for k,f,l2 in
                             ((15,.5,.08),(31,.5,.08),(31,1.,.08),(63,.5,.05))],
}
TREES = ("RandomForest","HistGradientBoosting")
RANDOM = TREES
ENET_GRAM = False


def folder(entity,model):
    return DEST/entity/model


def register():
    value=dict(common=registration(),grid=GRID,source=sha(ROOT/"classical.py"),common_source=sha(ROOT/"common.py"),
               sklearn=sklearn_version, target_modes=["delta","level"], selection_seed=20261124,
               selection="Pooled squared error over two chronological, embargoed training folds.",
               repeated="Five full refits per model. Random seeds for the two tree ensembles; deterministic algorithms retain the same fit inputs.",
               RF_n_estimators=128, HGB_max_iter=160, HGB_early_stopping=False, SVR_epsilon=.05,
               elastic_net_gram_entities=["CZ_Rod1","CZ_Rod2"])
    p=DEST/"REGISTRATION.json"
    if p.exists():
        assert read(p)==matched.json.loads(matched.json.dumps(value))
    else:
        write(p,value)


def bound():
    p=read(DEST/"REGISTRATION.json")
    assert p["source"]==sha(ROOT/"classical.py") and p["common_source"]==sha(ROOT/"common.py")
    assert p["common"]==matched.json.loads(matched.json.dumps(registration()))
    assert p["grid"]==GRID and p["sklearn"]==sklearn_version
    return p


def estimator(model,spec,seed):
    if model=="PCR":
        return make_pipeline(PCA(**spec,svd_solver="covariance_eigh"),LinearRegression())
    if model=="PLS":
        return PLSRegression(**spec,scale=False,max_iter=2000)
    if model=="ElasticNet":
        return ElasticNet(**spec,max_iter=15000,tol=1e-4,precompute=ENET_GRAM)
    if model=="RBF_SVR":
        return SVR(kernel="precomputed",C=spec["C"],epsilon=.05,cache_size=1024)
    if model=="RandomForest":
        return RandomForestRegressor(**spec,n_estimators=128,n_jobs=2,random_state=seed)
    if model=="HistGradientBoosting":
        return HistGradientBoostingRegressor(**spec,max_iter=160,early_stopping=False,min_samples_leaf=20,
                                             l2_regularization=1.,random_state=seed)
    raise ValueError(model)


def fit_model(model,spec,seed,x,y):
    y_mean=float(y.mean())
    y_scale=max(float(y.std()),1e-10)
    fitted=estimator(model,spec,seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        fitted.fit(x,(y-y_mean)/y_scale)
    return fitted,y_mean,y_scale


def infer(model,x,x_train=None,gamma=None):
    est,y_mean,y_scale=model
    if x_train is not None:
        keep=est.support_
        prediction=rbf_kernel(x,x_train[keep],gamma=gamma)@est.dual_coef_.ravel()+est.intercept_[0]
    else:
        prediction=est.predict(x).reshape(-1)
    return np.asarray(prediction,dtype=np.float64)*y_scale+y_mean


def scaled_design(model,a,fit,eval_rows=None):
    x=design(a,fit)
    sc=None if model in TREES else StandardScaler().fit(x)
    z=None if eval_rows is None else design(a,eval_rows)
    if sc is not None:
        x=sc.transform(x)
        if z is not None:
            z=sc.transform(z)
    return x,z,sc


def select(entity,model,a,t,seed):
    ledger={(i,recipe):[] for i,recipe in product(range(len(GRID[model])),("delta","level"))}
    for fold,(fi,ev) in enumerate(matched.folds(t)):
        x,z,_=scaled_design(model,a,fi,ev)
        last_gamma=None
        for i,spec in enumerate(GRID[model]):
            is_svr=model=="RBF_SVR"
            gamma=spec.get("gamma_scale",0.)/x.shape[1]
            if is_svr and gamma!=last_gamma:
                gram=rbf_kernel(x,gamma=gamma)
                last_gamma=gamma
            for recipe in ("delta","level"):
                y,_=target(a,fi,recipe)
                _,anchor=target(a,ev,recipe)
                trace=dict(fold=fold,last_fit=int(fi[-1]),first_score=int(ev[0]),N=len(ev))
                try:
                    fitted=fit_model(model,spec,seed,gram if is_svr else x,y)
                except ConvergenceWarning as problem:
                    ledger[i,recipe].append(dict(**trace,status="DID_NOT_CONVERGE",reason=str(problem)))
                    continue
                p=infer(fitted,z,x if is_svr else None,gamma)+anchor
                residual=a[ev,0]-p
                assert np.isfinite(residual).all()
                ledger[i,recipe].append(dict(**trace,status="COMPLETE",SSE=float(residual@residual)))
        print("CV",entity,model,fold+1,flush=True)
    result=[]
    for (i,recipe),folds in ledger.items():
        mse=(sum(f["SSE"] for f in folds)/sum(f["N"] for f in folds)
             if all(f["status"]=="COMPLETE" for f in folds) else None)
        result.append(dict(grid_index=i,spec=GRID[model][i],recipe=recipe,pooled_MSE=mse,folds=folds))
    best=min((r for r in result if r["pooled_MSE"] is not None),key=lambda r:r["pooled_MSE"])
    return best,result


def train(entity,model):
    reg=bound()
    start=time.monotonic()
    a,t=train_rows(entity)
    dest=folder(entity,model)
    receipt=dest/"FROZEN.json"
    if receipt.exists():
        verify(entity,model)
        return
    ledger=dest/"SELECTION.json"
    if ledger.exists():
        selection=read(ledger)
        assert selection["registration"]==sha(DEST/"REGISTRATION.json")
        best=selection["selected"]
    else:
        best,scores=select(entity,model,a,t,reg["selection_seed"])
        write(ledger,dict(selected=best,candidates=scores,registration=sha(DEST/"REGISTRATION.json")))
    z,_,scaler=scaled_design(model,a,t)
    svr=model=="RBF_SVR"
    gamma=best["spec"].get("gamma_scale",0.)/z.shape[1]
    x=rbf_kernel(z,gamma=gamma) if svr else z
    y,_=target(a,t,best["recipe"])
    files={}
    for seed in SEEDS:
        i=time.monotonic()
        fitted=fit_model(model,best["spec"],seed,x,y)
        name=f"seed_{seed}.joblib"
        path=dest/name
        tmp=path.with_suffix(".tmp")
        joblib.dump(dict(model=fitted,scaler=scaler,gamma=gamma,seed=seed,recipe=best["recipe"],spec=best["spec"]),
                    tmp,compress=3)
        tmp.replace(path)
        files[name]=sha(path)
        print("REFIT",entity,model,seed,round(time.monotonic()-i,1),flush=True)
    write(receipt,dict(selected=best,selection_sha256=sha(ledger),fit_rows=len(t),fit_last=int(t[-1]),files=files,
                       elapsed_seconds=time.monotonic()-start))


def verify(entity,model):
    path=folder(entity,model)
    receipt=read(path/"FROZEN.json")
    assert set(receipt["files"])=={f"seed_{seed}.joblib" for seed in SEEDS}
    assert sha(path/"SELECTION.json")==receipt["selection_sha256"]
    assert all(sha(path/name)==digest for name,digest in receipt["files"].items())
    return receipt


def evaluate(entity,model):
    bound()
    seal=read(DEST/"ALL_FROZEN.json")
    dest=folder(entity,model)
    assert sha(dest/"FROZEN.json")==seal["receipts"][str(dest.relative_to(DEST))]
    assert sha(DEST/"REGISTRATION.json")==seal["registration"]
    record=verify(entity,model)
    a,t=development(entity)
    past,train=train_rows(entity)
    spoiled,prefix=poison(a,t)
    x,z=design(a,t),design(spoiled,prefix)
    np.testing.assert_array_equal(x[:len(prefix)],z)
    sv=design(past,train) if model=="RBF_SVR" else None
    predictions=[]
    for seed in SEEDS:
        obj=joblib.load(dest/f"seed_{seed}.joblib")
        assert obj["seed"]==seed and obj["recipe"]==record["selected"]["recipe"] and obj["spec"]==record["selected"]["spec"]
        sc=obj["scaler"]
        p=infer(obj["model"],sc.transform(x) if sc is not None else x,
                sc.transform(sv) if sv is not None else None,obj["gamma"])+target(a,t,obj["recipe"])[1]
        rz=infer(obj["model"],sc.transform(z) if sc is not None else z,
                 sc.transform(sv) if sv is not None else None,obj["gamma"])+target(spoiled,prefix,obj["recipe"])[1]
        np.testing.assert_allclose(rz,p[:len(prefix)],rtol=1e-11,atol=1e-10)
        if predictions and model not in RANDOM:
            np.testing.assert_array_equal(p,predictions[0])
        predictions.append(p)
    result=run_result(entity,model,t,predictions,dict(kind="classical", selected=record["selected"],
            deterministic=model not in RANDOM, causal_checks=len(SEEDS)))
    write(dest/"RESULT.json",result)
    print("SCORED_CLASSICAL",entity,model,flush=True)


def launch(action,entity,model):
    log=folder(entity,model)/f"{action}.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as handle:
        subprocess.run([sys.executable,"-B","-u",__file__,action,entity,model],
                       stdout=handle,stderr=subprocess.STDOUT,check=True)
    print("DONE",action,entity,model,flush=True)


def main():
    register()
    assert set(GRID)==set(CLASSICAL)
    names=("RandomForest","HistGradientBoosting","RBF_SVR","ElasticNet","PLS","PCR")
    queue=[(e,m) for m in names for e in ("CZ_Rod2","CZ_Rod1","SRU_H2S")]
    with ThreadPoolExecutor(6) as pool:
        for action in ("train","evaluate"):
            tasks=[pool.submit(launch,action,*args) for args in queue]
            for task in tasks:
                task.result()
            if action=="train":
                receipts={}
                for e,m in queue:
                    verify(e,m)
                    dest=folder(e,m)
                    receipts[str(dest.relative_to(DEST))]=sha(dest/"FROZEN.json")
                write(DEST/"ALL_FROZEN.json",dict(registration=sha(DEST/"REGISTRATION.json"),receipts=receipts))
    write(DEST/"COMPLETE.json",dict(result_count=len(queue),freeze=sha(DEST/"ALL_FROZEN.json")))


if __name__=="__main__":
    if len(sys.argv)==1:
        main()
    else:
        action,e,m=sys.argv[1:]
        ENET_GRAM = m=="ElasticNet" and e!="SRU_H2S"
        with threadpool_limits(limits=2 if m in TREES else 1):
            {"train":train,"evaluate":evaluate}[action](e,m)
