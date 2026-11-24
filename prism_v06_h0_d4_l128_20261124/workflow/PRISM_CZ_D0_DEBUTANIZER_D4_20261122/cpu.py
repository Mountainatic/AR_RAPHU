"""Direct CZ QC32 training and linear baselines."""
import importlib
import pickle
import sys
import time
from pathlib import Path

from protocol import (CASE, CODE_ROOT, D, DRIVER, FEEDBACK, FIRST_ISSUE, GEOMETRY, H, L, NAMES, ROOT, RODS,
                      folds, issues, np, read, sha, values, write)
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PRIMARY = "QC32_RLS"
RECIPES = ("level", "delta")
WARM = FEEDBACK+32
ALPHAS = (.01,.1,1.,10.,100.,1000.,10000.,30000.,100000.,300000.,1e6,3e6,1e7,1e8,1e9)
def frame(rod, a, start, stop):
    o=np.arange(start,stop)
    return pd.DataFrame(dict(entity_id=rod,origin=o,target_end=o+H,y=a[o+H,0],anchor=a[o-D,0]))


def model_runtime(recipe):
    paths=[ROOT/"runtime"/f"{recipe}_CZ", DRIVER/"runtimes"/f"{recipe}_CZ", DRIVER]
    sys.path[:0]=[str(x) for x in paths]
    m=importlib.import_module("model")
    m.L=m.E_HISTORY=L
    m.configure(D)
    if L != 128:
        edges = [0, 1, *(2**k for k in range(1, L.bit_length()))]
        if CASE == "DEBUT":
            edges = [0, 1, 4, 16]
        intervals = tuple(zip(edges[:-1], edges[1:]))
        m.e_intervals = lambda length: intervals
    configs = m.configs

    def with_process(variant, family):
        return [c for c in configs(variant, family)
                if c["kind"] in ("LINEAR", "URYSOHN") and c.get("own") and c.get("memory")]

    m.configs = with_process
    return m


def streams(m, data, v, upper):
    q=m.predict_kcw(data,v,upper)
    return m.source(v,q,"one_parent"),q


def bind():
    reg=read(ROOT/"CPU_REGISTRATION.json")
    assert all(sha(Path(path)) == h for path,h in reg["code_sha256"].items())


def pair(m, case, data, upper, a, start, stop, policy):
    context=frame(case,a,start-WARM,start)
    active=frame(case,a,start,stop)
    hist,_=streams(m,data,context,upper)
    f,q=streams(m,data,active,upper)
    if policy is None or not policy["admitted"]:
        return q.copy(),q,0
    pred,log=m.mature_a.predict_spec(f,policy["model"],policy["selected"],history=hist,update=True)
    assert (log["latest_parameter_label"] <= f.latest_available_target_index).all()
    assert (log["latest_feature_label"] <= f.latest_available_target_index).all()
    return q+pred,q,log["updates"]


def feature_selector(m):
    a = m.mature_a
    selected = [spec for spec in a.SPECS_CONDITIONAL if spec["family"] == PRIMARY and spec["p"] == 1]
    assert selected
    for key in ("SPECS_RLS", "SPECS_BIAS", "SPECS_Q", "SPECS_DYNAMIC"):
        setattr(a, key, [])
    a.SPECS_CONDITIONAL = selected
    a.all_designs = lambda f: {(PRIMARY, 1): a.dynamic_features(f, 1, PRIMARY)}
    return a


def short_record_policy(a, f):
    """One forward gate using a fixed QC32 policy from the CZ experiment."""
    spec = dict(family=PRIMARY, p=1, window=0, forget=.995, gain=1.)
    design = a.all_designs(f)
    ei = a.split(f,.65,1.)
    fi = a.earlier(f,ei)
    initial = a.initialize(f,fi,spec,design)
    fit_last = initial["fit_last_target_end"]
    first_score = int(f.iloc[ei].origin.min())
    assert fit_last < first_score-a.L-a.PMAX-a.DELAY
    eval_frame = f.iloc[ei]
    estimate,log = a.predict_spec(eval_frame,initial,spec,history=f.iloc[:ei[0]].tail(WARM))
    assert (log["latest_parameter_label"]<=eval_frame.latest_available_target_index).all()
    error = eval_frame.residual.to_numpy()
    parent = float(np.mean(error**2))
    child = float(np.mean((error-estimate)**2))
    final = a.initialize(f,np.arange(len(f)),spec,design)
    return dict(selected=spec,model=final,admitted=child<parent,parent_risk=parent,child_risk=child,
                outer=[dict(role="single chronological training-only gate, fixed QC32 parameters",
                            rows=len(ei),fit_rows=initial["fit_rows"],latest_fit_target=fit_last,
                            first_evaluation_origin=first_score)])


def save_pickle(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj,protocol=5))
    tmp.replace(path)


def frozen_path(rod, recipe):
    return ROOT / rod / "prism" / "full" / f"{recipe}.pkl"


def train_parent(rod, recipe):
    bind()
    dest = frozen_path(rod, recipe)
    receipt = dest.with_suffix(".json")
    if receipt.exists():
        assert sha(dest) == read(receipt)["sha256"]
        return
    started = time.monotonic()
    m = model_runtime(recipe)
    a = values(rod)
    data = {rod:a}
    first,train_end,*_ = GEOMETRY[rod]
    if CASE == "DEBUT":
        first = 825
    t = frame(rod,a,FIRST_ISSUE,first-H)
    cal = frame(rod,a,first+L+2*D,train_end-H-max(1,D)+1)
    assert t.target_end.max() < cal.origin.min()-(L+D-1)
    assert cal.target_end.max() == train_end-max(1,D)
    up = m.fit_kcw(data,t,"full",allow_calibration=False)
    f,_ = streams(m,data,cal,up)
    selector = feature_selector(m)
    policy = selector.fit(f) if CASE=="CZ" else short_record_policy(selector,f)
    risk = policy["child_risk"] if policy["admitted"] else policy["parent_risk"]
    if risk is None:
        raise RuntimeError("Outer calibration unsupported")
    save_pickle(dest,dict(rod=rod,recipe=recipe,upper=up,policy=policy,last_parent_label=int(t.target_end.max())))
    write(receipt,dict(sha256=sha(dest),recipe=recipe,risk=risk,upper=up["parent"]["config"],
                       link=up["parent"]["coordinate"],policy=policy["selected"],admitted=policy["admitted"],
                       training_seconds=time.monotonic()-started))
    print("FROZEN",rod,recipe,risk,flush=True)


def chosen(rod):
    paths = [frozen_path(rod,r) for r in RECIPES]
    dest = min(paths,key=lambda p:read(p.with_suffix(".json"))["risk"])
    assert sha(dest)==read(dest.with_suffix(".json"))["sha256"]
    return dest


def linear_design(a,o,name):
    own=a[o[:,None]-D-np.arange(L)[None,:],0]
    if name=="AR":
        return own
    process=a[o[:,None]-np.arange(L)[None,:],1:].reshape(len(o),-1)
    return np.c_[own,process]


def train_linear(rod):
    bind()
    dest=ROOT/rod/"linear.pkl"; manifest=dest.with_suffix(".json")
    if manifest.exists():
        assert sha(dest)==read(manifest)["sha256"]
        return
    o=issues(rod); a=values(rod); result={}
    for name in ("AR","ARX"):
        candidates=[]
        for target in RECIPES:
            val=[]
            for t,v in folds(o):
                x=linear_design(a,t,name); u=linear_design(a,v,name)
                sc=StandardScaler().fit(x); x=sc.transform(x); u=sc.transform(u)
                y=a[t+H,0]-(a[t-D,0] if target=="delta" else 0.)
                ref=a[v+H,0]-(a[v-D,0] if target=="delta" else 0.)
                mat=x.T@x; rhs=x.T@(y-y.mean())
                eig,V=np.linalg.eigh((mat+mat.T)/2)
                coef=V@((V.T@rhs)[:,None]/(eig[:,None]+np.asarray(ALPHAS)[None,:]))
                p=u@coef+y.mean()
                val.append((p-ref[:,None])**2)
            risk=np.vstack(val).mean(0)
            candidates.extend((float(v),aa,target) for aa,v in zip(ALPHAS,risk))
        value,alpha,target=min(candidates)
        x=linear_design(a,o,name); sc=StandardScaler().fit(x)
        y=a[o+H,0]-(a[o-D,0] if target=="delta" else 0.)
        fit=Ridge(alpha=alpha,solver="cholesky").fit(sc.transform(x),y)
        result[name]=dict(scaler=sc,ridge=fit,target=target,alpha=alpha,risk=value)
    save_pickle(dest,result)
    write(manifest,dict(sha256=sha(dest),selected={k:(v["alpha"],v["target"],v["risk"])
                                                 for k,v in result.items()}))
    print("FROZEN_LINEAR",rod,flush=True)


def evaluate_parent(rod):
    bind()
    dest = chosen(rod)
    m = model_runtime(dest.stem)
    model = pickle.loads(dest.read_bytes())
    a = values(rod,True)
    o = issues(rod,True)
    data = {rod:a}
    p,q,updates = pair(m,rod,data,model["upper"],a,int(o[0]),int(o[-1])+1,model["policy"])
    cut = int(o[H+D+35])
    poison = a.copy()
    poison[cut-D+1:,0] += 1234
    poison[cut+1:,1:] -= 1234
    trial,_,_ = pair(m,rod,{rod:poison},model["upper"],poison,int(o[0]),cut+1,model["policy"])
    np.testing.assert_allclose(trial,p[:len(trial)],rtol=1e-12,atol=1e-10)
    details = read(dest.with_suffix(".json"))
    if details["admitted"]:
        assert updates == len(o)-FEEDBACK
    write(ROOT/rod/"PRISM_EVALUATION.json",dict(**details,updates=int(updates),feedback=FEEDBACK,
                                                causal_prefix_check=True))
    np.savez(ROOT/rod/"prism_evaluation.npz",origin=o,target_index=o+H,PRISM_QC32=p,upstream_only=q)
    print("EVAL",rod,"PRISM_QC32",updates,flush=True)


def evaluate_linear(rod):
    a=values(rod,True); o=issues(rod,True)
    dest=ROOT/rod/"linear.pkl"
    assert sha(dest)==read(dest.with_suffix(".json"))["sha256"]
    obj=pickle.loads(dest.read_bytes()); out={"Persistence":a[o-D,0]}
    for name,head in obj.items():
        x=linear_design(a,o,name)
        p=head["ridge"].predict(head["scaler"].transform(x))+(a[o-D,0] if head["target"]=="delta" else 0.)
        cut=int(o[113]); poison=a.copy()
        poison[cut-D+1:,0]+=1234; poison[cut+1:,1:]-=1234
        np.testing.assert_array_equal(x[:114],linear_design(poison,o[:114],name))
        out[name]=p
    np.savez(ROOT/rod/"linear_evaluation.npz",origin=o,target_index=o+H,**out)


def main():
    mode,*args=sys.argv[1:]
    funcs=dict(parent=train_parent,linear=train_linear,
               evaluate=evaluate_parent,evaluate_linear=evaluate_linear)
    funcs[mode](*args)


if __name__=="__main__":
    main()
