"""Cross lag resolution with reference-vs-joint channels under identical ridge/EN grids."""

import itertools
import pickle
from pathlib import Path
import sys
import time
import warnings

from common import ROOT, CASES, matched, h, np, inputs, poisoned, score, read, write, sha, registration, bound
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

SCRIPT = Path(__file__).resolve()
DEST = ROOT/"resolution"
WINDOWS = {
    "coarse": (0,8,24,60,128),
    "near_raw": (*range(9),24,60,128),
    "dense": tuple(range(129)),
}
SCOPES = ("reference", "joint")
METHODS = ("ridge","elastic_net")
RECIPES = ("level","delta")
RIDGE = np.array([.01,.1,1.,10.,100.,1000.,10000.,30000.,100000.,300000.,1e6,3e6,1e7,1e8,1e9])
ENET = tuple(itertools.product((.003,.03,.3),(.2,.8)))


def register():
    registration(SCRIPT,dict(windows=WINDOWS, channel_scopes=SCOPES, methods=METHODS,
        recipes=RECIPES, ridge_alphas=RIDGE.tolist(), elastic_net_grid=ENET,
        reference="Frozen original K reference coordinate from the disjoint earlier training block; CZ joint lift uses its frozen PCA.",
        label_history="The 128 individual mature y lags are shared in all arms.",
        folds="Original two embargoed training folds. Same objectives and grid within each method.",
        elastic_net="Exact Gram path for all entities, same tolerances as the preceding benchmark."))


def process(data, reference, scope):
    if scope == "joint":
        return data[:,1:]
    names = legacy_columns(reference["channel"])
    if reference["channel"] == "joint_lift":
        pca=reference["joint_lift_pca"]
        return (((data[:,2:4]-pca["mean"])/pca["scale"])@np.asarray(pca["pc1_loading"]))[:,None]
    return data[:,1+names.index(reference["channel"])][:,None]


def legacy_columns(name):
    return tuple(f"u{i}" for i in range(1,6)) if name.startswith("u") else h.COLUMNS


def design(data,t,reference,scope,name):
    lags=t[:,None]-np.arange(h.L)
    assert lags.min()>=h.D
    own=data[lags-h.D,0]
    x=process(data,reference,scope)[lags]
    edges=WINDOWS[name]
    if name != "dense":
        x=np.stack([x[:,a:b].mean(1) for a,b in zip(edges[:-1],edges[1:],strict=True)],axis=1)
    return np.c_[own,x.reshape(len(own),-1)]


def target(data,t,recipe):
    anchor=data[t-h.D,0] if recipe=="delta" else np.zeros(len(t))
    return data[t,0]-anchor,anchor


def gram_paths(x):
    gram=x.T@x
    eigen, v=np.linalg.eigh((gram+gram.T)*.5)
    return np.maximum(eigen,0.),v,np.ascontiguousarray(gram)


def ridge_solutions(x,y,eigen,v):
    return v@((v.T@(x.T@(y-y.mean())))[:,None]/(eigen[:,None]+RIDGE[None,:]))


def enet(x,y,gram,alpha,l1):
    center=float(y.mean()); scale=max(float(y.std()),1e-10)
    e=ElasticNet(alpha=alpha,l1_ratio=l1,fit_intercept=False,precompute=gram,max_iter=15000,tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        e.fit(x,(y-center)/scale)
    return e.coef_*scale,center


def tune(data,origins,reference,scope,name):
    errors={k:[] for k in itertools.product(METHODS,RECIPES,range(len(RIDGE)))}
    failed=set()
    for fit,ev in matched.folds(origins):
        x=design(data,fit,reference,scope,name)
        sc=StandardScaler().fit(x)
        x=sc.transform(x); z=sc.transform(design(data,ev,reference,scope,name))
        eig,v,gram=gram_paths(x)
        for recipe in RECIPES:
            y,_=target(data,fit,recipe)
            truth,_=target(data,ev,recipe)
            rr=truth[:,None]-(z@ridge_solutions(x,y,eig,v)+y.mean())
            for i,value in enumerate((rr*rr).sum(0)):
                errors["ridge",recipe,i].append(float(value))
            for i,(alpha,l1) in enumerate(ENET):
                key=("elastic_net",recipe,i)
                try:
                    beta,intercept=enet(x,y,gram,alpha,l1)
                except ConvergenceWarning:
                    failed.add(key)
                    continue
                e=truth-(z@beta+intercept)
                errors[key].append(float(e@e))
    best={m:min(((sum(err),recipe,i) for (method,recipe,i),err in errors.items()
                  if method==m and len(err)==2 and (m,recipe,i) not in failed)) for m in METHODS}
    return best,len(failed)


def reference_k(stages):
    k=stages["K"]
    return next(r for r in k["representations"] if r["channel"]==k["reference_channel"])


def fit(entity):
    bound(SCRIPT)
    start=time.monotonic()
    _,_,data,_,stages=inputs(entity)
    t=matched.issues(entity)
    ref=reference_k(stages)
    models={}
    decisions={}
    for scope,name in itertools.product(SCOPES,WINDOWS):
        best,failed=tune(data,t,ref,scope,name)
        x=design(data,t,ref,scope,name)
        sc=StandardScaler().fit(x); x=sc.transform(x)
        eig,v,gram=gram_paths(x)
        for method,(risk,recipe,i) in best.items():
            y,_=target(data,t,recipe)
            if method=="ridge":
                beta=ridge_solutions(x,y,eig,v)[:,i]; intercept=float(y.mean())
                spec=dict(alpha=float(RIDGE[i]))
            else:
                beta,intercept=enet(x,y,gram,*ENET[i]); spec=dict(alpha=ENET[i][0],l1_ratio=ENET[i][1])
            arm=f"{scope}_{name}_{method}"
            models[arm]=(sc,beta,intercept,scope,name,recipe)
            decisions[arm]=dict(**spec,recipe=recipe,fold_pooled_SSE=risk,features=x.shape[1],nonconverged_candidates=failed)
        print("FIT",entity,scope,name,flush=True)
    target_path=DEST/entity/"models.pkl"
    target_path.parent.mkdir(exist_ok=True,parents=True)
    tmp=target_path.with_suffix(".tmp"); tmp.write_bytes(pickle.dumps(models,protocol=5)); tmp.replace(target_path)
    write(target_path.with_name("FIT.json"),dict(entity=entity,sha256=sha(target_path), decisions=decisions,
           fit_rows=[int(t[0]),int(t[-1])+1], K_reference=ref["channel"], elapsed_seconds=time.monotonic()-start))


def freeze():
    bound(SCRIPT)
    result={}
    for e in CASES:
        p=DEST/e/"models.pkl"
        assert sha(p)==read(p.with_name("FIT.json"))["sha256"]
        result[e]=sha(p)
    write(DEST/"ALL_FROZEN.json",dict(models=result))


def evaluate(entity):
    bound(SCRIPT)
    p=DEST/entity/"models.pkl"
    assert sha(p)==read(DEST/"ALL_FROZEN.json")["models"][entity]
    models=pickle.loads(p.read_bytes())
    _,_,data,_,stages=inputs(entity,development=True)
    t=matched.issues(entity,evaluate=True); ref=reference_k(stages)
    y=data[t,0]
    cut=73; dirty=poisoned(data,int(t[cut])); prefix=t[:cut+1]
    results={}; predictions={}
    for arm,(sc,beta,intercept,scope,name,recipe) in models.items():
        x=design(data,t,ref,scope,name)
        z=design(dirty,prefix,ref,scope,name)
        np.testing.assert_array_equal(x[:len(z)],z)
        pred=sc.transform(x)@beta+intercept+target(data,t,recipe)[1]
        predictions[arm]=pred
        results[arm]=score(y,pred)
    with np.load(matched.ROOT/entity/"linear_evaluation.npz") as ctrl:
        np.testing.assert_array_equal(ctrl["target_index"],t)
        np.testing.assert_allclose(ctrl["ARX"],predictions["joint_dense_ridge"],rtol=0.,atol=5e-10)
    write(DEST/entity/"RESULT.json",dict(entity=entity,N=len(t),models=results,fit=read(p.with_name("FIT.json")),
          causality="prefix feature poison passed for every arm", dense_joint_ridge_matches_published_ARX=True))
    print("SCORED_RESOLUTION",entity,{k:round(v["RMSE"],6) for k,v in results.items()},flush=True)


if __name__ == "__main__":
    action,*args=sys.argv[1:]
    {"register":register,"fit":fit,"freeze":freeze,"evaluate":evaluate}[action](*args)
