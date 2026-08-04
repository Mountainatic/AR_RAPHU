from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec, load_samples
from .stage0 import write_json
from .v2_config import load_frozen_config
from .v2_views import development_dynamic_views, development_input_views


def _read_prediction(path:Path)->pd.DataFrame:
    return pd.read_parquet(path,columns=["sample_id","base_origin_id","target_head","split","information_set","availability_scenario","proxy_policy","model","y_true","y_pred","core_history_steps"])


def _prediction_index(output:Path)->dict[tuple[str,...],Path]:
    paths=[*sorted((output/"PREDICTIONS").glob("LEVEL_*/*/*/*/*/*/*.parquet")),*sorted((output/"BASELINE_PREDICTIONS").rglob("*.parquet"))]
    result={}
    for path in paths:
        frame=pd.read_parquet(path,columns=["target_head","split","information_set","availability_scenario","proxy_policy","model"])
        if frame.empty:continue
        row=frame.iloc[0];key=tuple(str(row[name]) for name in ("target_head","split","information_set","availability_scenario","proxy_policy","model"));result[key]=path
    return result


def _metrics(path:Path,level:str)->dict[str,Any]:
    frame=_read_prediction(path);error=frame["y_true"].to_numpy(dtype=np.float64)-frame["y_pred"].to_numpy(dtype=np.float64);row=frame.iloc[0]
    return {"level":level,"target_head":row["target_head"],"split":row["split"],"information_set":row["information_set"],"availability_scenario":row["availability_scenario"],
            "proxy_policy":row["proxy_policy"],"model":row["model"],"rows":len(frame),"mse":float(np.mean(error*error)),"rmse":float(np.sqrt(np.mean(error*error))),
            "mae":float(np.mean(np.abs(error))),"prediction_path":str(path)}


def _c6_metrics(archive:Path)->pd.DataFrame:
    with tarfile.open(archive) as handle:
        member=next(item for item in handle.getmembers() if item.name.endswith("CPU_FINAL_METRICS.csv"));stream=handle.extractfile(member)
        if stream is None:raise RuntimeError("CPU_FINAL_METRICS.csv absent from C6 summary")
        frame=pd.read_csv(stream)
    frame["level"]="LEVEL_B_PRIMARY_EXPLORATORY_BASELINE_REUSE";frame["availability_scenario"]=frame.get("availability_scenario","record_time");frame["proxy_policy"]=frame.get("proxy_policy","primary")
    return frame


def _block_draws(difference:np.ndarray,reference_loss:np.ndarray,entities:np.ndarray,block:int,replicates:int,seed:int)->np.ndarray:
    labels=pd.unique(entities);prepared=[]
    for label in labels:
        values=difference[entities==label];reference=reference_loss[entities==label];count=len(values)
        if count==0:continue
        if count<block:
            prepared.append((np.asarray([values.sum()]),np.asarray([reference.sum()]),count,1));continue
        dcumulative=np.concatenate([[0.0],np.cumsum(values,dtype=np.float64)]);rcumulative=np.concatenate([[0.0],np.cumsum(reference,dtype=np.float64)])
        dsum=dcumulative[block:]-dcumulative[:-block];rsum=rcumulative[block:]-rcumulative[:-block]
        prepared.append((dsum,rsum,count,int(math.ceil(count/block))))
    rng=np.random.default_rng(seed);draws=np.empty(replicates,dtype=np.float64)
    for replicate in range(replicates):
        total_d=0.0;total_r=0.0
        for selected in rng.integers(0,len(prepared),size=len(prepared)):
            dsum,rsum,_,blocks=prepared[int(selected)];indices=rng.integers(0,len(dsum),size=blocks);total_d+=float(dsum[indices].sum());total_r+=float(rsum[indices].sum())
        draws[replicate]=total_d/max(total_r,1e-300)
    return draws


def _holm(frame:pd.DataFrame)->pd.DataFrame:
    frame=frame.copy();frame["holm_reject"]=False;frame["holm_adjusted_p"]=np.nan
    groups=["target_head","split","information_set","availability_scenario","proxy_policy","block_length"]
    for _,indices in frame.groupby(groups,dropna=False).groups.items():
        ordered=sorted(indices,key=lambda index:frame.at[index,"p_two_sided"]);running=0.0;m=len(ordered);continue_reject=True
        for rank,index in enumerate(ordered):
            adjusted=min(1.0,max(running,(m-rank)*float(frame.at[index,"p_two_sided"])));running=adjusted;frame.at[index,"holm_adjusted_p"]=adjusted
            if continue_reject and float(frame.at[index,"p_two_sided"])<=0.05/(m-rank):frame.at[index,"holm_reject"]=True
            else:continue_reject=False
    return frame


def _bootstrap(output:Path,shared:Path,index:dict[tuple[str,...],Path],config:dict[str,Any])->pd.DataFrame:
    views={}
    for view in [*development_input_views(shared),*development_dynamic_views(shared)]:views[(view.head.head_id,view.information_set,view.availability_scenario,view.proxy_policy)]=view
    by_view={}
    for key,path in index.items():by_view.setdefault(key[:5],{})[key[5]]=path
    comparisons=[]
    for view_key,models in by_view.items():
        if "PRISM_V2_FINAL_ASSEMBLY" not in models:continue
        baseline_names=[name for name in models if name not in {"PRISM_V2_FINAL_ASSEMBLY","PRISM_V2_JOINT_PREDICTIVE"} and not name.startswith("K_") and name!="A_ONLY"]
        if baseline_names:
            losses={name:_metrics(models[name],"TEMP")["mse"] for name in baseline_names};best=min(losses,key=losses.get);comparisons.append((view_key,best,"PRISM_V2_FINAL_ASSEMBLY"))
        for reference,candidate in (("K_COMPRESSED","K_JOINT_BASIS"),("K_JOINT_BASIS","K_JOINT_BASIS_W"),("K_JOINT_BASIS","K_JOINT_BASIS_A"),("K_JOINT_BASIS_W","K_JOINT_BASIS_W_A")):
            if reference in models and candidate in models:comparisons.append((view_key,reference,candidate))
    rows=[];boot=config["bootstrap"]
    for comparison_index,(key,reference,candidate) in enumerate(comparisons):
        head,split,information,availability,proxy=key;view=views[(head,information,availability,proxy)];left=_read_prediction(by_view[key][reference]);right=_read_prediction(by_view[key][candidate])
        if not np.array_equal(left["sample_id"].to_numpy(),right["sample_id"].to_numpy()):raise RuntimeError(f"paired sample mismatch: {key} {reference} {candidate}")
        samples=load_samples(shared,view,split);lookup=samples.set_index("view_sample_id")["entity_id"];entities=lookup.loc[left["sample_id"]].astype(str).to_numpy()
        y=left["y_true"].to_numpy(dtype=np.float64);reference_loss=(y-left["y_pred"].to_numpy(dtype=np.float64))**2;candidate_loss=(y-right["y_pred"].to_numpy(dtype=np.float64))**2;difference=reference_loss-candidate_loss
        core=int(max(left["core_history_steps"].iloc[0],right["core_history_steps"].iloc[0]));blocks=sorted(set([view.head.h_steps+view.head.w_steps,2*(view.head.h_steps+view.head.w_steps),int(math.ceil(core/4))]));longest=max(blocks)
        for block in blocks:
            draws=_block_draws(difference,reference_loss,entities,max(1,block),int(boot["replicates"]),int(boot["seed"])+comparison_index*101+block)
            p=min(1.0,2*min((np.count_nonzero(draws<=0)+1)/(len(draws)+1),(np.count_nonzero(draws>=0)+1)/(len(draws)+1)))
            rows.append({"target_head":head,"split":split,"information_set":information,"availability_scenario":availability,"proxy_policy":proxy,"reference":reference,"candidate":candidate,
                         "block_length":block,"is_longest_block":block==longest,"relative_mse_improvement":float(difference.mean()/max(reference_loss.mean(),1e-300)),
                         "ci_low":float(np.quantile(draws,0.025)),"ci_high":float(np.quantile(draws,0.975)),"positive_probability":float(np.mean(draws>0)),"p_two_sided":float(p)})
    return _holm(pd.DataFrame(rows)) if rows else pd.DataFrame()


def build_v2_report(shared:Path,project:Path,output:Path,c6_archive:Path)->dict[str,Any]:
    config=load_frozen_config(project);index=_prediction_index(output);metrics=[]
    for key,path in index.items():metrics.append(_metrics(path,"LEVEL_C_CONFIRMATION" if "LEVEL_C_CONFIRMATION" in str(path) else "LEVEL_B_PRIMARY_EXPLORATORY"))
    frame=pd.DataFrame(metrics);c6=_c6_metrics(c6_archive);common=sorted(set(frame.columns)&set(c6.columns));combined=pd.concat([frame,c6[common]],ignore_index=True,sort=False)
    combined.to_csv(output/"PRISM_V2_FINAL_METRICS.csv",index=False)
    rank_rows=[]
    for keys,group in combined.groupby(["level","split","information_set"],dropna=False):
        task_model=group.groupby(["target_head","model"],as_index=False)["mse"].mean();task_model["rank"]=task_model.groupby("target_head")["mse"].rank(method="average")
        for model,value in task_model.groupby("model")["rank"].mean().items():rank_rows.append({"level":keys[0],"split":keys[1],"information_set":keys[2],"model":model,"mean_rank":value,"tasks":int((task_model["model"]==model).sum())})
    ranks=pd.DataFrame(rank_rows);ranks.to_csv(output/"PRISM_V2_CROSS_TASK_RANKS.csv",index=False)
    activations=[]
    for path in sorted((output/"ASSEMBLY_CARDS").rglob("ASSEMBLY_CARD.json")):
        card=json.loads(path.read_text());activations.append({key:card.get(key) for key in ("target_head","information_set","availability_scenario","proxy_policy","selected_assembly")})
    activation_frame=pd.DataFrame(activations);activation_frame.to_csv(output/"PRISM_V2_MODULE_ACTIVATION.csv",index=False)
    bootstrap=_bootstrap(output,shared,index,config);(output/"BOOTSTRAP").mkdir(exist_ok=True);bootstrap.to_csv(output/"BOOTSTRAP"/"PAIRED_BLOCK_BOOTSTRAP.csv",index=False)
    supported=0
    if not bootstrap.empty:
        grouped=bootstrap.groupby(["target_head","split","information_set","availability_scenario","proxy_policy","reference","candidate"])
        for _,group in grouped:
            if int(group["holm_reject"].sum())>=2 and bool(group.loc[group["is_longest_block"],"holm_reject"].any()) and float(group["positive_probability"].min())>=0.95:supported+=1
    decision={"status":"COMPLETED","protocol_id":config["protocol_id"],"level_c_prediction_files":sum("LEVEL_C_CONFIRMATION" in str(path) for path in index.values()),
              "level_b_prediction_files":sum("LEVEL_B_PRIMARY_EXPLORATORY" in str(path) for path in index.values()),"statistically_supported_comparisons":supported,
              "engineering_success":bool(activations) and all(item.get("selected_assembly") for item in activations),"test_accessed":True}
    write_json(output/"PRISM_V2_FINAL_DECISION.json",decision)
    best=ranks.sort_values(["level","split","information_set","mean_rank"]).groupby(["level","split","information_set"],as_index=False).first() if not ranks.empty else ranks
    report=["# PRISM V2 Modular CPU Benchmark — Final Report","",f"- Protocol: `{config['protocol_id']}`",f"- Status: `{decision['status']}`",f"- Assembly cards: {len(activations)}",f"- Statistically supported paired comparisons: {supported}","","## Average-rank leaders",""]
    if not best.empty:report.extend(["| Level | Split | Information | Model | Mean rank |","|---|---|---|---|---:|",*[f"| {r.level} | {r.split} | {r.information_set} | {r.model} | {r.mean_rank:.3f} |" for r in best.itertuples()]])
    report.extend(["","Level C is prospective internal confirmation. Level B is explicitly post-hoc exploratory and reuses the frozen C6 V2 baseline summary.","All retained failures remain present in the result tree; raw C1 data are not included in release artifacts."])
    (output/"PRISM_V2_FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return decision
