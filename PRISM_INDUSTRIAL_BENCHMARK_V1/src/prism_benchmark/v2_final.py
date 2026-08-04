from __future__ import annotations

import ast
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, load_samples, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_a import _fit_residual, mature_features
from .v2_c import _load_active_channels, _pair_columns, _ridge_fit, _ridge_predict, fit_physical_features
from .v2_config import V2Paths, load_frozen_config
from .v2_j import _joint_fit, _single_fit, _state_features
from .v2_k import _cap, profile_values
from .v2_state import _feasible_state_rows, _quadratic, _standardized_fit, predict_state
from .v2_urysohn import basis_from_metadata, predict_contract
from .v2_views import development_dynamic_views, development_input_views, evaluation_level
from .v2_w import fit_w_candidate, predict_w_contract


def _development_samples(shared:Path,view:ViewSpec)->pd.DataFrame:
    return pd.concat([load_samples(shared,view,"train"),load_samples(shared,view,"validation")],ignore_index=True).sort_values(["entity_id","origin"]).reset_index(drop=True)


def _input_view(view:ViewSpec)->ViewSpec:
    return ViewSpec(view.head,"input_only","record_time",view.proxy_policy)


def _load_result(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if value.get("status")!="PASS":raise RuntimeError(f"prerequisite not PASS: {path}")
    return value


def _card(output:Path,view:ViewSpec)->dict[str,Any]:
    return _load_result(output/"ASSEMBLY_CARDS"/view.head.head_id/view.information_set/view.availability_scenario/view.proxy_policy/"ASSEMBLY_CARD.json")


def _physical_results(output:Path,view:ViewSpec)->tuple[dict[str,Any],dict[str,Any],list[dict[str,Any]]]:
    c=_load_result(output/"DEVELOPMENT"/"JOINT_BASIS"/view.head.head_id/view.proxy_policy/"RESULT.json")
    w=_load_result(output/"DEVELOPMENT"/"WIENER"/view.head.head_id/view.proxy_policy/"RESULT.json")
    return c,w,_load_active_channels(output,_input_view(view))


def fit_final_physical(shared:Path,view:ViewSpec,output:Path,config:dict[str,Any])->dict[str,Any]:
    input_view=_input_view(view);c,w,active=_physical_results(output,view);samples=_development_samples(shared,input_view)
    fit=_cap(samples,int(config["row_caps"]["joint_physical_fit"]));features=fit_physical_features(shared,input_view,fit,fit,active,config,fit_split="validation",evaluation_split="validation")
    target=fit["y_true"].to_numpy(dtype=np.float64);fusion={}
    compressed_alpha=float(c["component_files"]["K_COMPRESSED"]["contract"].get("alpha",c["selected_alpha"]))
    joint_alpha=float(c["component_files"]["K_JOINT_BASIS"]["contract"].get("alpha",c["selected_alpha"]))
    for name,key,alpha in (("K_COMPRESSED","compressed",compressed_alpha),("K_JOINT_BASIS","joint",joint_alpha)):
        x=features[f"{key}_train"]
        if x.shape[1]:prediction,contract=_ridge_fit(x,target,x,alpha)
        else:prediction=np.zeros(len(fit));contract={"status":"EXACT_ZERO","mean":[],"scale":[],"coefficient":[],"intercept":0.0,"alpha":alpha}
        fusion[name]={"contract":contract,"train_prediction":prediction}
    pair_train=features["joint_train"];pair_contracts=[]
    for pair_value in c.get("selected_pairs",[]):
        pair=tuple(int(x) for x in pair_value);columns,_,contract=_pair_columns(features["compressed_train"],features["compressed_train"],pair,4)
        pair_train=np.concatenate([pair_train,columns],axis=1);pair_contracts.append(contract)
    if c.get("selected_pairs"):
        pair_prediction,pair_fusion=_ridge_fit(pair_train,target,pair_train,float(c["selected_alpha"]))
        fusion["K_PAIRWISE"]={"contract":pair_fusion,"train_prediction":pair_prediction}
    c_name="K_PAIRWISE" if c.get("selected_family")=="SPARSE_PAIRWISE_ANOVA_MAX3" else ("K_JOINT_BASIS" if c.get("selected_family")=="ADDITIVE_JOINT_BASIS" else "K_COMPRESSED")
    latent=fusion[c_name]["train_prediction"];template=w["w_contract"]
    _,w_contract=fit_w_candidate(latent,target,latent,template["family"],int(template.get("knot_count",0)),float(template.get("smoothness",0.0)),int(template.get("direction",1)))
    return {"fit":fit,"input_view":input_view,"channels":features["channels"],"channel_contracts":features["channel_contracts"],
            "global_joint_columns":features.get("global_joint_columns",[]),"compressed_train":features["compressed_train"],"joint_train":features["joint_train"],
            "pair_train":pair_train,"pair_contracts":pair_contracts,"c_result":c,"w_result":w,"w_contract":w_contract,"fusion":fusion}


def physical_features_for_samples(
    shared:Path,
    context:dict[str,Any],
    samples:pd.DataFrame,
    split:str,
    accessor:BaseAccessor|None=None,
)->dict[str,np.ndarray]:
    contracts=context["channel_contracts"]
    if not contracts:return {"compressed":np.empty((len(samples),0)),"joint":np.empty((len(samples),0))}
    if accessor is None:
        accessor=BaseAccessor(shared,context["input_view"].head.dataset,split,[item["channel"] for item in contracts])
    compressed=[];joint=[]
    for item in contracts:
        values,_=profile_values(accessor,samples,item["channel"],tuple(item["profile"]),int(item["m_tau"]))
        compressed.append(predict_contract(values,item["k_contract"]))
        basis=basis_from_metadata(item["k_contract"]["basis"]);raw=basis.transform(values).reshape(len(values),-1)
        joint.append(raw[:,np.asarray(item["joint_columns"],dtype=np.int64)])
    joint_matrix=np.concatenate(joint,axis=1);global_columns=np.asarray(context["global_joint_columns"],dtype=np.int64)
    if len(global_columns) and len(global_columns)<joint_matrix.shape[1]:joint_matrix=joint_matrix[:,global_columns]
    return {"compressed":np.column_stack(compressed),"joint":joint_matrix}


def physical_predictions(context:dict[str,Any],features:dict[str,np.ndarray])->dict[str,np.ndarray]:
    result={}
    for name,key in (("K_COMPRESSED","compressed"),("K_JOINT_BASIS","joint")):
        contract=context["fusion"][name]["contract"]
        result[name]=np.zeros(len(features[key])) if contract.get("status")=="EXACT_ZERO" else _ridge_predict(features[key],contract)
    if "K_PAIRWISE" in context["fusion"]:
        x=features["joint"]
        for pair_value in context["c_result"]["selected_pairs"]:
            pair=tuple(int(v) for v in pair_value);_,columns,_=_pair_columns(context["compressed_train"],features["compressed"],pair,4);x=np.concatenate([x,columns],axis=1)
        result["K_PAIRWISE"]=_ridge_predict(x,context["fusion"]["K_PAIRWISE"]["contract"])
    return result


def _physical_for_assembly(context:dict[str,Any],predictions:dict[str,np.ndarray],assembly:str)->np.ndarray:
    if assembly=="K_COMPRESSED":return predictions["K_COMPRESSED"]
    if assembly.startswith("K_PAIRWISE") and "K_PAIRWISE" in predictions:base=predictions["K_PAIRWISE"]
    elif "_W" in assembly:
        selected=context["c_result"].get("selected_family")
        base=predictions["K_PAIRWISE"] if selected=="SPARSE_PAIRWISE_ANOVA_MAX3" and "K_PAIRWISE" in predictions else (predictions["K_JOINT_BASIS"] if selected=="ADDITIVE_JOINT_BASIS" else predictions["K_COMPRESSED"])
    else:base=predictions["K_JOINT_BASIS"]
    if "_W" in assembly:return predict_w_contract(base,context["w_contract"])
    return base


def _chunked_physical_prediction(
    shared:Path,
    context:dict[str,Any],
    samples:pd.DataFrame,
    split:str,
    assembly:str,
    chunk_size:int=100000,
)->np.ndarray:
    return _chunked_physical_predictions(shared,context,samples,split,[assembly],chunk_size)[assembly]


def _chunked_physical_predictions(
    shared:Path,
    context:dict[str,Any],
    samples:pd.DataFrame,
    split:str,
    assemblies:list[str],
    chunk_size:int=100000,
)->dict[str,np.ndarray]:
    contracts=context["channel_contracts"]
    accessor=None
    if contracts:
        accessor=BaseAccessor(
            shared,
            context["input_view"].head.dataset,
            split,
            [item["channel"] for item in contracts],
        )
    chunks={assembly:[] for assembly in assemblies}
    for start in range(0,len(samples),chunk_size):
        chunk=samples.iloc[start:start+chunk_size]
        features=physical_features_for_samples(shared,context,chunk,split,accessor=accessor)
        predictions=physical_predictions(context,features)
        for assembly in assemblies:chunks[assembly].append(_physical_for_assembly(context,predictions,assembly))
    return {assembly:(np.concatenate(values) if values else np.empty(0,dtype=np.float64)) for assembly,values in chunks.items()}


def _a_contract(output:Path,view:ViewSpec,assembly:str)->dict[str,Any]:
    result=_load_result(output/"DEVELOPMENT"/"RESIDUAL_STATE"/view.head.head_id/view.availability_scenario/view.proxy_policy/"RESULT.json")
    return result["identity_w_a"]["a_contract"] if assembly=="K_JOINT_BASIS_A" else result["a_contract"]


def _predict_a(features:np.ndarray,physical:np.ndarray,contract:dict[str,Any])->np.ndarray:
    if contract["family"]=="EXACT_ZERO":return np.zeros(len(features),dtype=np.float64)
    x=(features-np.asarray(contract["mean"],dtype=np.float64))/np.asarray(contract["scale"],dtype=np.float64)
    if contract["family"]=="ORTHOGONAL_MATURE_RESIDUAL_AR":x=x-np.column_stack([np.ones(len(x)),physical])@np.asarray(contract["projection"],dtype=np.float64)
    return x@np.asarray(contract["coefficient"],dtype=np.float64)+float(contract["intercept"])


def _refit_a_contract(
    fit_frame:pd.DataFrame,
    source:pd.DataFrame,
    view:ViewSpec,
    template:dict[str,Any],
)->dict[str,Any]:
    if template["family"]=="EXACT_ZERO":return {"family":"EXACT_ZERO","parameter_count":0}
    profile=tuple(int(value) for value in template["profile"]);residual_mean=float(fit_frame["residual"].mean())
    features,_=mature_features(fit_frame,source,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
    _,contract=_fit_residual(features,fit_frame["residual"].to_numpy(dtype=np.float64),features,float(template["alpha"]),template["family"],fit_frame["physical"].to_numpy(dtype=np.float64),fit_frame["physical"].to_numpy(dtype=np.float64))
    contract["profile"]=list(profile);return contract


def _fit_state_final(shared:Path,view:ViewSpec,output:Path,config:dict[str,Any])->tuple[dict[str,Any],BaseAccessor]:
    result=_load_result(output/"DEVELOPMENT"/"STATE_ONLY"/view.head.head_id/view.availability_scenario/"RESULT.json");selected=result["contract"]
    accessor=BaseAccessor(shared,view.head.dataset,"validation",[view.head.target])
    if selected["family"]=="EXACT_ZERO":return {"family":"EXACT_ZERO"},accessor
    fit=_cap(_development_samples(shared,view),int(config["row_caps"]["state_fit"]));profile=tuple(int(x) for x in selected["profile"])
    fit=fit.iloc[np.flatnonzero(_feasible_state_rows(fit,accessor,profile))];x=accessor.target_state(fit,view.head.target,*profile)
    if selected["family"]=="NAR_TARGET_QUADRATIC":x=_quadratic(x,int(selected["maximum_linear_state_features_before_expansion"]))
    _,contract=_standardized_fit(x,fit["y_true"].to_numpy(dtype=np.float64),x,float(selected["alpha"]));contract.update({"family":selected["family"],"profile":list(profile)})
    if selected["family"]=="NAR_TARGET_QUADRATIC":contract["maximum_linear_state_features_before_expansion"]=int(selected["maximum_linear_state_features_before_expansion"])
    return contract,accessor


def _state_prediction(samples:pd.DataFrame,view:ViewSpec,contract:dict[str,Any],accessor:BaseAccessor)->np.ndarray:
    if contract["family"]=="EXACT_ZERO":return np.zeros(len(samples))
    chunks=[]
    for start in range(0,len(samples),100000):
        features=accessor.target_state(samples.iloc[start:start+100000],view.head.target,*tuple(contract["profile"]));chunks.append(predict_state(features,contract))
    return np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float64)


def _j_selection(result:dict[str,Any])->str|tuple[Any,...]:
    value=result["selected_candidate"]
    if isinstance(value,str) and value.startswith("("):return ast.literal_eval(value)
    return value


def _fit_final_j(shared:Path,view:ViewSpec,output:Path,config:dict[str,Any],context:dict[str,Any])->dict[str,Any]:
    result=_load_result(output/"DEVELOPMENT"/"JOINT_PREDICTIVE"/view.head.head_id/view.availability_scenario/view.proxy_policy/"RESULT.json")
    state_result=_load_result(output/"DEVELOPMENT"/"STATE_ONLY"/view.head.head_id/view.availability_scenario/"RESULT.json");selected=_j_selection(result)
    fit=_cap(_development_samples(shared,view),int(config["row_caps"]["joint_predictive_fit"]));state_template=state_result["contract"]
    uses_state=selected not in {"EXACT_BOTH_ZERO","EXACT_STATE_ZERO"}
    if uses_state and state_template["family"]!="EXACT_ZERO":
        state_accessor=BaseAccessor(shared,view.head.dataset,"validation",[view.head.target]);profile=tuple(int(x) for x in state_template["profile"])
        fit=fit.iloc[np.flatnonzero(_feasible_state_rows(fit,state_accessor,profile))].reset_index(drop=True)
    contracts=context["channel_contracts"];input_accessor=BaseAccessor(shared,view.head.dataset,"validation",[item["channel"] for item in contracts]) if contracts else None
    k_fit=physical_features_for_samples(shared,context,fit,"validation",input_accessor)["joint"]
    state_fit=_state_features(shared,view,fit,"validation",state_result) if uses_state else np.empty((len(fit),0),dtype=np.float64)
    target=fit["y_true"].to_numpy(dtype=np.float64);w_contract=None;k_scalar_contract=None
    if selected=="EXACT_BOTH_ZERO":contract={"family":selected,"parameter_count":0}
    elif selected=="EXACT_K_ZERO":_,contract=_single_fit(state_fit,target,state_fit,float(config["J_module"]["ridge_alpha_grid"][0]));contract["family"]=selected
    elif selected=="EXACT_STATE_ZERO":_,contract=_single_fit(k_fit,target,k_fit,float(config["J_module"]["ridge_alpha_grid"][0]));contract["family"]=selected
    else:
        family,alpha,ratio=selected
        if family=="JOINT_KW_STATE_LINEAR":
            k_scalar,k_scalar_contract=_single_fit(k_fit,target,k_fit,float(config["J_module"]["ridge_alpha_grid"][0]));template=context["w_result"]["w_contract"]
            kw,w_contract=fit_w_candidate(k_scalar,target,k_scalar,template["family"],int(template.get("knot_count",0)),float(template.get("smoothness",0.0)),int(template.get("direction",1)))
            k_fit=np.column_stack([k_fit,kw])
        _,contract=_joint_fit(k_fit,state_fit,target,k_fit,state_fit,float(alpha),float(ratio));contract["family"]=family
    return {"selected":selected,"contract":contract,"state_result":state_result,"k_scalar_contract":k_scalar_contract,"w_contract":w_contract}


def _linear_contract_prediction(features:np.ndarray,contract:dict[str,Any])->np.ndarray:
    if contract.get("family")=="EXACT_BOTH_ZERO":return np.zeros(len(features),dtype=np.float64)
    return (features-np.asarray(contract["mean"],dtype=np.float64))/np.asarray(contract["scale"],dtype=np.float64)@np.asarray(contract["coefficient"],dtype=np.float64)+float(contract["intercept"])


def _predict_final_j(shared:Path,view:ViewSpec,samples:pd.DataFrame,split:str,context:dict[str,Any],j:dict[str,Any],chunk_size:int)->np.ndarray:
    selected=j["selected"];contracts=context["channel_contracts"];input_accessor=BaseAccessor(shared,view.head.dataset,split,[item["channel"] for item in contracts]) if contracts else None
    state_accessor=BaseAccessor(shared,view.head.dataset,split,[view.head.target]);chunks=[]
    for start in range(0,len(samples),chunk_size):
        chunk=samples.iloc[start:start+chunk_size];k=physical_features_for_samples(shared,context,chunk,split,input_accessor)["joint"]
        state=_state_features(shared,view,chunk,split,j["state_result"]) if selected not in {"EXACT_BOTH_ZERO","EXACT_STATE_ZERO"} else np.empty((len(chunk),0),dtype=np.float64)
        if selected=="EXACT_BOTH_ZERO":features=np.empty((len(chunk),0))
        elif selected=="EXACT_K_ZERO":features=state
        elif selected=="EXACT_STATE_ZERO":features=k
        else:
            if selected[0]=="JOINT_KW_STATE_LINEAR":
                scalar=_linear_contract_prediction(k,j["k_scalar_contract"]);k=np.column_stack([k,predict_w_contract(scalar,j["w_contract"])])
            features=np.concatenate([k,state],axis=1)
        chunks.append(_linear_contract_prediction(features,j["contract"]))
    return np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float64)


def _prediction_frame(samples:pd.DataFrame,view:ViewSpec,split:str,model:str,assembly:str,prediction:np.ndarray,parameter_count:int,core_history:int)->pd.DataFrame:
    frame=samples[["view_sample_id","base_origin_id","dataset","task_id","target_head","split","y_true"]].copy()
    frame=frame.rename(columns={"view_sample_id":"sample_id"});frame["information_set"]=view.information_set;frame["availability_scenario"]=view.availability_scenario
    frame["proxy_policy"]=view.proxy_policy;frame["model"]=model;frame["assembly_id"]=assembly;frame["y_pred"]=prediction;frame["parameter_count"]=parameter_count
    frame["dtype"]="float64";frame["core_history_steps"]=core_history;frame["support_flag"]="INHERITED_TRAIN_SUPPORT_AUDIT_PENDING";frame["ood_flag"]=split=="ood"
    return frame


def evaluate_view_split(shared:Path,project:Path,output:Path,view:ViewSpec,split:str,level:str)->list[dict[str,Any]]:
    started=time.time();config=load_frozen_config(project);card=_card(output,view);assembly=card["selected_assembly"];context=fit_final_physical(shared,view,output,config)
    samples=load_samples(shared,view,split);chunk_size=100000
    candidates=list(card["candidate_predictions"]);base_assemblies=sorted({name[:-2] if name.endswith("_A") else name for name in candidates if name!="A_ONLY"})
    physical=_chunked_physical_predictions(shared,context,samples,split,base_assemblies,chunk_size);predictions={};parameters={}
    for name in candidates:
        if name!="A_ONLY" and not name.endswith("_A"):
            predictions[name]=physical[name]
            key="K_COMPRESSED" if name=="K_COMPRESSED" else ("K_PAIRWISE" if name.startswith("K_PAIRWISE") else "K_JOINT_BASIS")
            parameters[name]=len(context["fusion"][key]["contract"].get("coefficient",[]))+1
    if "A_ONLY" in candidates:
        state_contract,_=_fit_state_final(shared,view,output,config);accessor=BaseAccessor(shared,view.head.dataset,split,[view.head.target])
        predictions["A_ONLY"]=_state_prediction(samples,view,state_contract,accessor);parameters["A_ONLY"]=len(state_contract.get("coefficient",[]))+1
    a_names=[name for name in candidates if name.endswith("_A")]
    if a_names:
        fit_dynamic=_development_samples(shared,view);input_fit=_development_samples(shared,_input_view(view));fit_bases=sorted({name[:-2] for name in a_names})
        fit_physical=_chunked_physical_predictions(shared,context,input_fit,"validation",fit_bases,chunk_size)
        for name in a_names:
            base=name[:-2];template=_a_contract(output,view,name);fit_aligned=fit_dynamic.merge(pd.DataFrame({"base_origin_id":input_fit["base_origin_id"],"physical":fit_physical[base]}),on="base_origin_id",how="inner")
            eval_frame=samples.copy();eval_frame["physical"]=physical[base]
            fit_aligned["residual"]=fit_aligned["y_true"]-fit_aligned["physical"];eval_frame["residual"]=eval_frame["y_true"]-eval_frame["physical"]
            source=pd.concat([fit_aligned[["entity_id","origin","residual"]],eval_frame[["entity_id","origin","residual"]]],ignore_index=True)
            contract=_refit_a_contract(fit_aligned,fit_aligned[["entity_id","origin","residual"]],view,template);residual_mean=float(fit_aligned["residual"].mean());profile=tuple(contract.get("profile",[1,1]));residual=[]
            for start in range(0,len(eval_frame),chunk_size):
                chunk=eval_frame.iloc[start:start+chunk_size];x,_=mature_features(chunk,source,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
                residual.append(_predict_a(x,chunk["physical"].to_numpy(),contract))
            predictions[name]=physical[base]+np.concatenate(residual);parameters[name]=len(contract.get("coefficient",[]))+1
    predictions["PRISM_V2_FINAL_ASSEMBLY"]=predictions[assembly];parameters["PRISM_V2_FINAL_ASSEMBLY"]=parameters[assembly]
    if view.information_set=="dynamic":
        j=_fit_final_j(shared,view,output,config,context);predictions["PRISM_V2_JOINT_PREDICTIVE"]=_predict_final_j(shared,view,samples,split,context,j,chunk_size)
        parameters["PRISM_V2_JOINT_PREDICTIVE"]=int(j["contract"].get("parameter_count",len(j["contract"].get("coefficient",[]))+1))
    root=output/"PREDICTIONS"/level/view.head.head_id/split/view.information_set/view.availability_scenario/view.proxy_policy;root.mkdir(parents=True,exist_ok=True);audits=[]
    for model,prediction in predictions.items():
        path=root/f"{model}.parquet";frame=_prediction_frame(samples,view,split,model,assembly if model=="PRISM_V2_FINAL_ASSEMBLY" else model,prediction,parameters[model],max(view.head.h_steps*8,view.head.w_steps));frame.to_parquet(path,index=False,compression="zstd")
        audit={"status":"PASS","target_head":view.head.head_id,"split":split,"information_set":view.information_set,"availability_scenario":view.availability_scenario,
               "proxy_policy":view.proxy_policy,"model":model,"assembly":assembly if model=="PRISM_V2_FINAL_ASSEMBLY" else model,"rows":len(frame),"prediction_path":str(path.relative_to(output)),
               "prediction_sha256":sha256_file(path),"fit_partition":"train_plus_validation_only","test_accessed":True,"elapsed_seconds":time.time()-started,
               **regression_metrics(frame["y_true"].to_numpy(),frame["y_pred"].to_numpy())}
        audit_path=output/"MODEL_AUDIT"/level/view.head.head_id/split/view.information_set/view.availability_scenario/view.proxy_policy/f"{model}.json";write_json(audit_path,audit);audits.append(audit)
    return audits


def run_v8_level(shared:Path,project:Path,output:Path,level:str,n_jobs:int)->dict[str,Any]:
    freeze=output/"FREEZE"/"V2_FINAL_FREEZE_MANIFEST.json"
    if not freeze.is_file() or json.loads(freeze.read_text()).get("test_accessed") is not False:raise RuntimeError("G3 freeze manifest missing or invalid")
    views=[*development_input_views(shared),*development_dynamic_views(shared)];views=[view for view in views if evaluation_level(view,shared)==level]
    jobs=[]
    for view in views:
        for split in ("test","ood"):
            if (shared/"sample_ids"/view.relative_root/f"{split}.parquet").is_file():jobs.append((view,split))
    results=[]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures=[executor.submit(evaluate_view_split,shared,project,output,view,split,level) for view,split in jobs]
        for future in as_completed(futures):
            try:results.extend(future.result())
            except Exception as error:results.append({"status":"SOLVER_FAILED_RETAINED","error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
    summary={"status":"PASS" if all(x["status"]=="PASS" for x in results) else "COMPLETED_WITH_RETAINED_FAILURES","stage":level,"jobs":len(jobs),"pass":sum(x["status"]=="PASS" for x in results),"results":results}
    write_json(output/"PREDICTIONS"/level/"SUMMARY.json",summary);return summary
