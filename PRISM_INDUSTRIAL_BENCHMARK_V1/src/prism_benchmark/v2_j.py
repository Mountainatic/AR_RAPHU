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

from .cpu_data import BaseAccessor, ViewSpec, inner_folds, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_c import _load_active_channels, fit_physical_features
from .v2_config import load_frozen_config
from .v2_k import _cap
from .v2_numerics import solve_certified
from .v2_selection import one_se_select, practical_activation
from .v2_state import _quadratic
from .v2_views import development_dynamic_views
from .v2_w import fit_w_candidate


def _load_prerequisites(output: Path, view: ViewSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c_path=output/"DEVELOPMENT"/"JOINT_BASIS"/view.head.head_id/view.proxy_policy/"RESULT.json"
    w_path=output/"DEVELOPMENT"/"WIENER"/view.head.head_id/view.proxy_policy/"RESULT.json"
    s_path=output/"DEVELOPMENT"/"STATE_ONLY"/view.head.head_id/view.availability_scenario/"RESULT.json"
    values=[json.loads(path.read_text()) for path in (c_path,w_path,s_path)]
    if any(value.get("status")!="PASS" for value in values): raise RuntimeError("V7 prerequisite is not PASS")
    return tuple(values)  # type: ignore[return-value]


def _state_features(shared:Path,view:ViewSpec,samples:pd.DataFrame,split:str,state_result:dict[str,Any])->np.ndarray:
    contract=state_result["contract"]
    if contract["family"]=="EXACT_ZERO": return np.empty((len(samples),0),dtype=np.float64)
    profile=tuple(int(value) for value in contract["profile"])
    accessor=BaseAccessor(shared,view.head.dataset,split,[view.head.target])
    features=accessor.target_state(samples,view.head.target,*profile)
    if contract["family"]=="NAR_TARGET_QUADRATIC": features=_quadratic(features,int(contract["maximum_linear_state_features_before_expansion"]))
    return features


def _joint_fit(k_train:np.ndarray,state_train:np.ndarray,target:np.ndarray,k_eval:np.ndarray,state_eval:np.ndarray,alpha:float,ratio:float)->tuple[np.ndarray,dict[str,Any]]:
    train=np.concatenate([k_train,state_train],axis=1); evaluation=np.concatenate([k_eval,state_eval],axis=1)
    mean=train.mean(axis=0,dtype=np.float64); scale=train.std(axis=0,dtype=np.float64); scale[scale*scale<1e-12]=1.0
    x=(train-mean)/scale; z=(evaluation-mean)/scale
    penalty=np.diag(np.concatenate([np.full(k_train.shape[1],alpha*ratio),np.full(state_train.shape[1],alpha)]))
    target_mean=float(np.mean(target)); coefficient,certificate=solve_certified(x,target-target_mean,penalty)
    return z@coefficient+target_mean,{"mean":mean.tolist(),"scale":scale.tolist(),"coefficient":coefficient.tolist(),"intercept":target_mean,
                                               "alpha":alpha,"k_over_state_ratio":ratio,"k_columns":k_train.shape[1],"state_columns":state_train.shape[1],
                                               "numerical_certificate":certificate.to_json(),"parameter_count":len(coefficient)+1}


def _single_fit(train:np.ndarray,target:np.ndarray,evaluation:np.ndarray,alpha:float)->tuple[np.ndarray,dict[str,Any]]:
    return _joint_fit(train,np.empty((len(train),0)),target,evaluation,np.empty((len(evaluation),0)),alpha,1.0)


def _kw_scalar(fit_latent:np.ndarray,target:np.ndarray,evaluation_latent:np.ndarray,w_result:dict[str,Any])->np.ndarray:
    contract=w_result["w_contract"]
    if contract["family"]=="IDENTITY": return evaluation_latent
    family=contract["family"]; knots=int(contract["knot_count"]); smoothness=float(contract["smoothness"]); direction=int(contract["direction"])
    return fit_w_candidate(fit_latent,target,evaluation_latent,family,knots,smoothness,direction)[0]


def run_j_view(shared:Path,project:Path,output:Path,view:ViewSpec)->dict[str,Any]:
    started=time.time(); destination=output/"DEVELOPMENT"/"JOINT_PREDICTIVE"/view.head.head_id/view.availability_scenario/view.proxy_policy
    destination.mkdir(parents=True,exist_ok=True)
    try:
        config=load_frozen_config(project);c_result,w_result,state_result=_load_prerequisites(output,view);active=_load_active_channels(output,view)
        train=load_samples(shared,view,"train");validation=load_samples(shared,view,"validation");folds=inner_folds(train,4)
        alphas=[float(x) for x in config["J_module"]["ridge_alpha_grid"]];ratios=[float(x) for x in config["J_module"]["block_penalty_ratios_k_over_state"]]
        candidates=["EXACT_BOTH_ZERO","EXACT_K_ZERO","EXACT_STATE_ZERO"]+[(family,alpha,ratio) for family in ("JOINT_K_STATE_LINEAR","JOINT_KW_STATE_LINEAR") for alpha in alphas for ratio in ratios]
        losses={candidate:[] for candidate in candidates};fold_cache=[]
        for fit_index,evaluation_index in folds:
            fit=_cap(train.iloc[fit_index],int(config["row_caps"]["joint_predictive_fit"]));evaluation=_cap(train.iloc[evaluation_index],int(config["row_caps"]["validation_selection_per_fold"]))
            features=fit_physical_features(shared,view,fit,evaluation,active,config,fit_split="train",evaluation_split="train")
            k_fit=features["joint_train"];k_eval=features["joint_evaluation"]
            state_fit=_state_features(shared,view,fit,"train",state_result);state_eval=_state_features(shared,view,evaluation,"train",state_result)
            y_fit=fit["y_true"].to_numpy(dtype=np.float64);y_eval=evaluation["y_true"].to_numpy(dtype=np.float64)
            losses["EXACT_BOTH_ZERO"].append(float(np.mean(y_eval*y_eval)))
            if state_fit.shape[1]: losses["EXACT_K_ZERO"].append(mse(y_eval,_single_fit(state_fit,y_fit,state_eval,alphas[0])[0]))
            else: losses["EXACT_K_ZERO"].append(losses["EXACT_BOTH_ZERO"][-1])
            if k_fit.shape[1]: losses["EXACT_STATE_ZERO"].append(mse(y_eval,_single_fit(k_fit,y_fit,k_eval,alphas[0])[0]))
            else: losses["EXACT_STATE_ZERO"].append(losses["EXACT_BOTH_ZERO"][-1])
            k_train_scalar=_single_fit(k_fit,y_fit,k_fit,alphas[0])[0] if k_fit.shape[1] else np.zeros(len(fit))
            k_eval_scalar=_single_fit(k_fit,y_fit,k_eval,alphas[0])[0] if k_fit.shape[1] else np.zeros(len(evaluation))
            kw_fit=_kw_scalar(k_train_scalar,y_fit,k_train_scalar,w_result)
            kw_eval=_kw_scalar(k_train_scalar,y_fit,k_eval_scalar,w_result)
            for family,alpha,ratio in candidates[3:]:
                current_k_fit=k_fit if family=="JOINT_K_STATE_LINEAR" else np.column_stack([k_fit,kw_fit])
                current_k_eval=k_eval if family=="JOINT_K_STATE_LINEAR" else np.column_stack([k_eval,kw_eval])
                prediction,_=_joint_fit(current_k_fit,state_fit,y_fit,current_k_eval,state_eval,alpha,ratio)
                losses[(family,alpha,ratio)].append(mse(y_eval,prediction))
            fold_cache.append((fit,evaluation,features,state_fit,state_eval,y_fit,y_eval))
        def complexity(value):
            if value=="EXACT_BOTH_ZERO":return(0,)
            if value in {"EXACT_K_ZERO","EXACT_STATE_ZERO"}:return(1,0 if value=="EXACT_K_ZERO" else 1)
            family,alpha,ratio=value;return(2,0 if family=="JOINT_K_STATE_LINEAR" else 1,-alpha,abs(np.log2(ratio)))
        selection=one_se_select(losses,complexity,neutral="EXACT_BOTH_ZERO",minimum_usable_folds=3)
        selected=selection.selected
        exact_best=min(candidates[:3],key=lambda value:np.mean(losses[value]));activation={"pass":False,"reason":"NESTED_EXACT_ZERO_SELECTED"}
        if not isinstance(selected,str):
            activation=practical_activation(losses[exact_best],losses[selected],minimum_relative_improvement=float(config["J_module"]["activation_relative_mse_min_over_best_nested_exact_zero"]),minimum_positive_fraction=0.75)
            if not activation["pass"]:selected=exact_best
        fit=_cap(train,int(config["row_caps"]["joint_predictive_fit"]));features=fit_physical_features(shared,view,fit,validation,active,config,fit_split="train",evaluation_split="validation")
        k_fit=features["joint_train"];k_eval=features["joint_evaluation"];state_fit=_state_features(shared,view,fit,"train",state_result);state_eval=_state_features(shared,view,validation,"validation",state_result);y_fit=fit["y_true"].to_numpy(dtype=np.float64)
        if selected=="EXACT_BOTH_ZERO": prediction=np.zeros(len(validation));contract={"family":selected,"parameter_count":0}
        elif selected=="EXACT_K_ZERO": prediction,contract=_single_fit(state_fit,y_fit,state_eval,alphas[0]);contract["family"]=selected
        elif selected=="EXACT_STATE_ZERO": prediction,contract=_single_fit(k_fit,y_fit,k_eval,alphas[0]);contract["family"]=selected
        else:
            family,alpha,ratio=selected
            if family=="JOINT_KW_STATE_LINEAR":
                k_train_scalar=_single_fit(k_fit,y_fit,k_fit,alphas[0])[0];k_eval_scalar=_single_fit(k_fit,y_fit,k_eval,alphas[0])[0]
                k_fit=np.column_stack([k_fit,_kw_scalar(k_train_scalar,y_fit,k_train_scalar,w_result)]);k_eval=np.column_stack([k_eval,_kw_scalar(k_train_scalar,y_fit,k_eval_scalar,w_result)])
            prediction,contract=_joint_fit(k_fit,state_fit,y_fit,k_eval,state_eval,alpha,ratio);contract["family"]=family
        frame=validation[["base_origin_id","view_sample_id","entity_id","origin","y_true"]].copy();frame["y_pred"]=prediction;frame["model"]="PRISM_V2_JOINT_PREDICTIVE";frame["dtype"]="float64"
        prediction_path=destination/"validation.parquet";frame.to_parquet(prediction_path,index=False,compression="zstd")
        result={"status":"PASS","stage":"V7_JOINT_PREDICTIVE","target_head":view.head.head_id,"availability_scenario":view.availability_scenario,"proxy_policy":view.proxy_policy,
                "selected_candidate":str(selected),"j_contract":contract,"one_se":selection.to_json(),"activation":activation,"fold_losses":{str(k):v for k,v in losses.items()},
                "prediction_path":str(prediction_path.relative_to(output)),"prediction_sha256":sha256_file(prediction_path),"test_accessed":False,"elapsed_seconds":time.time()-started,
                **regression_metrics(frame["y_true"].to_numpy(),prediction)}
    except Exception as error:
        result={"status":"SOLVER_FAILED_RETAINED","stage":"V7_JOINT_PREDICTIVE","target_head":view.head.head_id,"availability_scenario":view.availability_scenario,"proxy_policy":view.proxy_policy,
                "test_accessed":False,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc(),"elapsed_seconds":time.time()-started}
    write_json(destination/"RESULT.json",result);return result


def run_v7_j(shared:Path,project:Path,output:Path,n_jobs:int)->dict[str,Any]:
    results=[];pending=[]
    for view in development_dynamic_views(shared):
        path=output/"DEVELOPMENT"/"JOINT_PREDICTIVE"/view.head.head_id/view.availability_scenario/view.proxy_policy/"RESULT.json"
        if path.is_file():
            prior=json.loads(path.read_text());
            if prior.get("status") in {"PASS","SOLVER_FAILED_RETAINED"}:results.append(prior);continue
        pending.append(view)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures=[executor.submit(run_j_view,shared,project,output,view) for view in pending]
        for future in as_completed(futures):results.append(future.result())
    summary={"status":"PASS" if all(x["status"]=="PASS" for x in results) else "COMPLETED_WITH_RETAINED_FAILURES","stage":"V7_JOINT_PREDICTIVE",
             "views":len(results),"pass":sum(x["status"]=="PASS" for x in results),"joint":sum("JOINT_" in x.get("selected_candidate","") for x in results),"test_accessed":False}
    write_json(output/"DEVELOPMENT"/"JOINT_PREDICTIVE"/"SUMMARY.json",summary);return summary

