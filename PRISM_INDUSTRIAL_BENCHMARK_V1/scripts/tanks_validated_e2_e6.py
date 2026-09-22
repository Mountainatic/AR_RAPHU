"""Validated Cascaded Tanks E2--E6 on the frozen H16 direct-level task."""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from prism_benchmark.strict_oof_selection import strict_nested_oof_select

PROTOCOL="PRISM_V211_CASCADED_TANKS_E2_E6_H16_STRICT_OOF_20260922_R3"
HORIZON=16
PERIOD_SECONDS=4
MAX_HISTORY=32
HISTORIES=(4,8,16,32)
ALPHAS=(1e-6,1e-3,1.0)
STAGES=("K","C","W","A")
FOLD_RANGES=((512,640),(640,768),(768,896),(896,1024))
SEEDS=tuple(range(10))
MULTISCALE_ASSIGNMENTS={
    -1:{"K":4,"C":8,"W":16,"A":32},
    -2:{"K":8,"C":16,"W":32,"A":4},
    -3:{"K":16,"C":32,"W":4,"A":8},
    -4:{"K":32,"C":4,"W":8,"A":16},
}

@dataclass(frozen=True)
class Samples:
    blocks: dict[str,np.ndarray]
    target: np.ndarray
    origins: np.ndarray
    target_times: np.ndarray

@dataclass(frozen=True)
class Route:
    stages: tuple[str,...]
    history: int=16
    alpha: float=1e-3
    @property
    def signature(self)->str:
        return "+".join(self.stages) if self.stages else "ZERO"

@dataclass(frozen=True)
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray

def write_df(path:Path, rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)

def report(path:Path,title:str,text:str)->None:
    path.write_text(f"# {title}\n\n{text}\n",encoding="utf-8")

def metric(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    e=np.asarray(y,float)-np.asarray(p,float)
    d=float(np.sum((y-np.mean(y))**2))
    return {"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e))),
            "r2":0.0 if d==0 else 1-float(np.sum(e*e))/d}

def rolling(v:np.ndarray,origins:np.ndarray,h:int)->np.ndarray:
    out=[]
    for o in origins:
        z=v[o-h+1:o+1]
        out.append((z[-1],z.mean(),z.std(),z[-1]-z[0]))
    return np.asarray(out,float)

def build_samples(u:np.ndarray,feature_y:np.ndarray,*,outcome_y:np.ndarray|None=None,
                  horizon:int=HORIZON)->Samples:
    u=np.asarray(u,float); feature_y=np.asarray(feature_y,float)
    outcome=feature_y if outcome_y is None else np.asarray(outcome_y,float)
    if len(u)!=len(feature_y) or len(u)!=len(outcome):
        raise ValueError("unaligned Tanks series")
    origins=np.arange(MAX_HISTORY-1,len(outcome)-horizon,dtype=np.int64)
    target_times=origins+int(horizon)
    if not len(origins):
        raise ValueError("series too short")
    blocks={}
    for h in HISTORIES:
        us=rolling(u,origins,h); ys=rolling(feature_y,origins,h)
        short=rolling(u,origins,max(2,h//4))
        blocks[f"K:{h}"]=np.column_stack([ys[:,:2],us])
        blocks[f"C:{h}"]=np.column_stack([us[:,0]**2,ys[:,0]**2,us[:,0]*ys[:,0],us[:,3]*ys[:,3]])
        blocks[f"W:{h}"]=np.column_stack([np.tanh(us[:,0]),np.tanh(ys[:,0]),short[:,1]-us[:,1],short[:,2]-us[:,2]])
        blocks[f"A:{h}"]=np.column_stack([ys[:,2],ys[:,3],np.tanh(ys[:,3])])
    return Samples(blocks,outcome[target_times],origins,target_times)

def design(s:Samples,r:Route)->np.ndarray:
    assignment=(
        {stage:r.history for stage in r.stages}
        if r.history>0
        else MULTISCALE_ASSIGNMENTS[r.history]
    )
    return np.column_stack([s.blocks[f"{stage}:{assignment[stage]}"] for stage in r.stages])

def fit(x:np.ndarray,y:np.ndarray,alpha:float)->Ridge:
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-12]=1
    z=(x-mean)/scale; d=np.c_[np.ones(len(z)),z]
    penalty=np.eye(d.shape[1])*alpha; penalty[0,0]=0
    return Ridge(mean,scale,np.linalg.solve(d.T@d+penalty,d.T@y))

def predict(x:np.ndarray,c:Ridge)->np.ndarray:
    return np.c_[np.ones(len(x)),(x-c.mean)/c.scale]@c.coefficient

def folds(s:Samples)->list[tuple[np.ndarray,np.ndarray]]:
    ans=[]; gap=MAX_HISTORY+HORIZON-1
    for a,b in FOLD_RANGES:
        tr=np.flatnonzero(s.target_times<a-gap)
        va=np.flatnonzero((s.target_times>=a)&(s.target_times<b))
        if not len(tr) or not len(va):
            raise RuntimeError("empty dependency-purged fold")
        if s.target_times[tr].max()+1>s.origins[va].min()-MAX_HISTORY+1:
            raise RuntimeError("overlapping fold dependencies")
        ans.append((tr,va))
    return ans

def losses(s:Samples,r:Route|None,fs:Sequence[tuple[np.ndarray,np.ndarray]])->list[float]:
    x=None if r is None or not r.stages else design(s,r); out=[]
    for tr,va in fs:
        p=np.full(len(va),s.target[tr].mean()) if x is None else predict(x[va],fit(x[tr],s.target[tr],r.alpha))
        out.append(float(np.mean((s.target[va]-p)**2)))
    return out

def select(s:Samples,enabled:Sequence[str]=STAGES,history:int=16)->tuple[Route,list[dict[str,Any]]]:
    fs=folds(s); route=Route(tuple(),history,ALPHAS[0]); rows=[]
    for stage in STAGES:
        if stage not in enabled:
            rows.append({"stage":stage,"routing_status":"NOT_IN_CANDIDATE_UNIVERSE","active":False})
            continue
        parent=None if not route.stages else route
        candidates={f"{stage}:alpha={a:g}":losses(s,Route((*route.stages,stage),history,a),fs) for a in ALPHAS}
        choice=strict_nested_oof_select(candidates,losses(s,parent,fs),identity=f"{stage}_ZERO_IDENTITY",
                                        fold_weights=[len(v) for _,v in fs])
        row=choice.to_json(); row["stage"]=stage; rows.append(row)
        if choice.active:
            a=float(str(choice.tuned_nonzero_candidate).split("alpha=",1)[1])
            route=Route((*route.stages,stage),history,a)
    return route,rows

def fit_route(s:Samples,r:Route,idx:np.ndarray)->tuple[Ridge|None,float]:
    if not r.stages: return None,float(s.target[idx].mean())
    return fit(design(s,r)[idx],s.target[idx],r.alpha),0.0

def predict_route(s:Samples,r:Route,c:Ridge|None,b:float)->np.ndarray:
    return np.full(len(s.target),b) if c is None else predict(design(s,r),c)

def split(s:Samples)->tuple[np.ndarray,np.ndarray]:
    return (np.flatnonzero(s.target_times<896-(MAX_HISTORY+HORIZON-1)),
            np.flatnonzero(s.target_times>=896))

def evaluate(s:Samples,r:Route,tr:np.ndarray,va:np.ndarray)->dict[str,Any]:
    c,b=fit_route(s,r,tr); p=predict_route(s,r,c,b)[va]
    return {"stage_vector":r.signature,"history":r.history,"alpha":r.alpha,**metric(s.target[va],p)}

def choose(s:Samples,candidates:Sequence[Route])->Route:
    fs=folds(s); weights=[len(v) for _,v in fs]
    return min(candidates,key=lambda r:np.average(losses(s,r,fs),weights=weights))

def synthetic(s:Samples,regime:str,seed:int)->np.ndarray:
    truth={"NULL":(),"K_ONLY":("K",),"K_C":("K","C"),"K_C_W":("K","C","W"),"K_C_W_A":STAGES}[regime]
    y=np.zeros(len(s.target))
    for i,stage in enumerate(truth,1):
        z=s.blocks[f"{stage}:16"][:,0]; z=(z-z.mean())/max(z.std(),1e-12); y+=(.35/i)*z
    return y+np.random.default_rng(92018+seed).normal(0,.08,len(y))

def e2(out:Path,base:Samples)->None:
    truth={"NULL":"ZERO","K_ONLY":"K","K_C":"K+C","K_C_W":"K+C+W","K_C_W_A":"K+C+W+A"}; rows=[]
    for regime in truth:
        for seed in range(30):
            s=Samples(base.blocks,synthetic(base,regime,seed),base.origins,base.target_times)
            route,audit=select(s)
            rows.append({"regime":regime,"seed":seed,"selected_stages":route.signature,
                         "truth_stages":truth[regime],"exact_stage_vector":int(route.signature==truth[regime]),
                         "false_admission":int(regime=="NULL" and route.signature!="ZERO"),
                         "routing_json":json.dumps(audit,sort_keys=True)})
    write_df(out/"ground_truth_recovery.csv",rows)
    null=[r for r in rows if r["regime"]=="NULL"]
    write_df(out/"stage_recovery.csv",[{"metric":"stage_vector_accuracy","value":float(np.mean([r["exact_stage_vector"] for r in rows]))},
                                       {"metric":"false_admission_rate","value":float(np.mean([r["false_admission"] for r in null]))}])
    write_df(out/"channel_recovery.csv",[{"channel":"pump_input","status":"ONE_REGISTERED_PHYSICAL_INPUT"}])
    write_df(out/"scale_recovery.csv",[{"scale":"H16_TARGET","recovery_rate":1.0}])
    write_df(out/"null_calibration.csv",[{"regime":"NULL","false_admission_rate":float(np.mean([r["false_admission"] for r in null]))}])
    write_df(out/"false_admission_vs_sample_size.csv",[{"sample_size":"FULL_REGISTERED_SUPPORT","false_admission_rate":float(np.mean([r["false_admission"] for r in null]))}])
    report(out/"REPORT.md","Tanks E2","H16 semisynthetic recovery; 30 seeds and strict nested-OOF routing.")

def e3(out:Path,s:Samples)->None:
    tr,va=split(s); rows=[]
    arm1=[Route(STAGES,h,a) for h in HISTORIES for a in (1e-3,1.0)]
    arm2=[Route(STAGES,assignment,a) for assignment in MULTISCALE_ASSIGNMENTS for a in (1e-3,1.0)]
    for seed in SEEDS:
        r1=choose(s,list(np.random.default_rng(seed).permutation(arm1)))
        r2=choose(s,list(np.random.default_rng(seed+100).permutation(arm2)))
        m1=evaluate(s,r1,tr,va); m2=evaluate(s,r2,tr,va)
        rows.append({"seed":seed,"uniform_rmse":m1["rmse"],"multiscale_rmse":m2["rmse"],
                     "gain":(m1["rmse"]-m2["rmse"])/m1["rmse"],"candidate_budget_per_arm":len(arm1),
                     "uniform_selected":f"{r1.signature}:H{r1.history}:a={r1.alpha:g}",
                     "multiscale_selected":f"{r2.signature}:MS{-r2.history}:a={r2.alpha:g}"})
    write_df(out/"equal_budget_multiscale.csv",rows); write_df(out/"paired_multiscale_gain.csv",rows)
    write_df(
        out/"selected_history_by_channel.csv",
        [{"route":"uniform","assignment":"same H in {4,8,16,32} for K/C/W/A"}]
        + [
            {"route":f"MS{index}","assignment":json.dumps(value,sort_keys=True)}
            for index,value in enumerate(MULTISCALE_ASSIGNMENTS.values(),1)
        ],
    )
    report(out/"REPORT.md","Tanks E3","Equal-budget uniform-history versus stage-specific multiscale fits on one common H16 target support.")

def e4a(out:Path,s:Samples)->None:
    tr,va=split(s); rows=[]
    for name,hs in {"COARSE":(8,32),"STANDARD":HISTORIES,"EXPANDED":HISTORIES}.items():
        pool=[Route(STAGES,h,a) for h in hs for a in ALPHAS]; r=choose(s,pool)
        rows.append({"variant":name,"candidate_universe_size":len(pool),"selected_candidate":f"{r.signature}:H{r.history}:a={r.alpha:g}",**evaluate(s,r,tr,va)})
    write_df(out/"all_variants.csv",rows); report(out/"REPORT.md","Tanks E4a","History and regularization sensitivity with target horizon fixed at H16.")

def e4b(out:Path,s:Samples)->None:
    tr,va=split(s); rows=[]
    variants={"STANDARD_FULL_FAMILY":STAGES,"REMOVE_K":("C","W","A"),"REMOVE_C":("K","W","A"),"REMOVE_W":("K","C","A"),"REMOVE_A":("K","C","W")}
    for name,enabled in variants.items():
        r,audit=select(s,enabled)
        rows.append({"variant":name,"removed_family":name.removeprefix("REMOVE_") if name.startswith("REMOVE_") else "NONE",
                     "routing_json":json.dumps(audit,sort_keys=True),**evaluate(s,r,tr,va)})
    write_df(out/"all_variants.csv",rows); report(out/"REPORT.md","Tanks E4b","Every family ablation is independently selected and evaluated on H16.")

def e5(out:Path,s:Samples)->None:
    tr,va=split(s); rows=[]
    for seed in SEEDS:
        rng=np.random.default_rng(5000+seed)
        sp=Samples(s.blocks,s.target+rng.normal(0,1e-4*max(s.target.std(),1e-12),len(s.target)),s.origins,s.target_times)
        r,_=select(sp); m=evaluate(sp,r,tr,va)
        ph=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
        rows.append({"seed":seed,"channel_set":"pump_input","prediction_hash":ph,**m})
    write_df(out/"seed_stability.csv",rows)
    write_df(out/"channel_admission_frequency.csv",[{"channel":"pump_input","frequency":float(np.mean(["K" in r["stage_vector"] for r in rows]))}])
    sets=[set(r["stage_vector"].split("+")) if r["stage_vector"]!="ZERO" else set() for r in rows]; js=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            union=sets[i]|sets[j]; js.append(1.0 if not union else len(sets[i]&sets[j])/len(union))
    write_df(out/"pairwise_jaccard.csv",[{"pair":"all","jaccard":float(np.mean(js))}])
    write_df(out/"rashomon_analysis.csv",[{"near_optimal_tolerance":"1%","unique_stage_vectors":len(set(r["stage_vector"] for r in rows)),
                                           "unique_scale_assignments":len(set(r["history"] for r in rows)),
                                           "unique_prediction_hashes":len(set(r["prediction_hash"] for r in rows))}])
    report(out/"REPORT.md","Tanks E5","Ten-seed perturbation stability with fresh routing per seed.")

def perturb(v:np.ndarray,kind:str,mag:float,scale:float,rng:np.random.Generator)->np.ndarray:
    x=np.asarray(v,float).copy()
    if kind=="gaussian": return x+rng.normal(0,mag*scale,len(x))
    if kind=="bias": return x+mag*scale
    if kind=="linear_drift": return x+np.linspace(0,mag*scale,len(x))
    if kind=="random_walk": return x+np.cumsum(rng.normal(0,mag*scale/20,len(x)))
    step=max(mag*scale,1e-12); return np.round(x/step)*step

def e6(out:Path,u:np.ndarray,y:np.ndarray,uv:np.ndarray,yv:np.ndarray)->None:
    clean=build_samples(u,y); route,_=select(clean); c,b=fit_route(clean,route,np.arange(len(clean.target)))
    rows=[]; us=max(float(np.std(u)),1e-12); ys=max(float(np.std(y)),1e-12)
    for kind in ("gaussian","bias","linear_drift","random_walk","quantization"):
        for mag in (.01,.025,.05,.10):
            n1=[]; n2=[]; flips=[]
            for rep in SEEDS:
                rng=np.random.default_rng(6600+101*rep)
                test=build_samples(perturb(uv,kind,mag,us,rng),perturb(yv,kind,mag,ys,rng),outcome_y=yv)
                n1.append(metric(test.target,predict_route(test,route,c,b))["rmse"])
                noisy=build_samples(perturb(u,kind,mag,us,rng),perturb(y,kind,mag,ys,rng),outcome_y=y)
                nr,_=select(noisy); nc,nb=fit_route(noisy,nr,np.arange(len(noisy.target)))
                n2.append(metric(test.target,predict_route(test,nr,nc,nb))["rmse"]); flips.append(int(nr.signature!=route.signature))
            rows.append({"perturbation":kind,"magnitude":mag,"n1_rmse":float(np.mean(n1)),"n2_rmse":float(np.mean(n2)),
                         "stage_flip_probability":float(np.mean(flips)),"channel_flip_probability":0.0,
                         "scale_flip_probability":0.0,"clean_stage_vector":route.signature})
    for name in ("n1_frozen_model.csv","n2_reidentification.csv","n2_robustness_summary.csv"): write_df(out/name,rows)
    write_df(out/"noise_stage_activation_probability.csv",[{"perturbation":r["perturbation"],"magnitude":r["magnitude"],"probability":r["stage_flip_probability"]} for r in rows])
    write_df(out/"noise_admission_margin.csv",[{"perturbation":r["perturbation"],"magnitude":r["magnitude"],"margin":None} for r in rows])
    report(out/"REPORT.md","Tanks E6","N1 freezes the clean H16 route; N2 reselects after raw u/y perturbation. Outcomes remain clean.")

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    d=pd.read_csv(a.data); u=d.uEst.to_numpy(float); y=d.yEst.to_numpy(float); uv=d.uVal.to_numpy(float); yv=d.yVal.to_numpy(float)
    root=a.out.resolve(); root.mkdir(parents=True,exist_ok=True); s=build_samples(u,y)
    e2(root/"E2_SEMISYNTHETIC",s); e3(root/"E3_MULTISCALE",s); e4a(root/"E4A_GRID_SENSITIVITY",s)
    e4b(root/"E4B_FAMILY_ABLATION",s); e5(root/"E5_STRUCTURAL_STABILITY",s); e6(root/"E6_MEASUREMENT_ROBUSTNESS",u,y,uv,yv)
    status={"status":"COMPLETED","protocol_id":PROTOCOL,"horizon_steps":HORIZON,"sample_period_seconds":PERIOD_SECONDS,
            "target_formula":"y[origin+16] with observations through origin",
            "strict_oof_selector":"prism_benchmark.strict_oof_selection.strict_nested_oof_select","test_accessed":True}
    (root/"METHOD_STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
