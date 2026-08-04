from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec, load_samples, realized_state_profiles, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_config import load_frozen_config
from .v2_k import _cap
from .v2_numerics import residualize, solve_certified
from .v2_selection import one_se_select, practical_activation
from .v2_runtime import run_parallel
from .v2_views import development_dynamic_views


def _input_view_result(output: Path, view: ViewSpec) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = output / "DEVELOPMENT" / "WIENER" / view.head.head_id / view.proxy_policy
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"V4 prerequisite is not PASS: {root}")
    return result, pd.read_parquet(output / result["oof_path"]), pd.read_parquet(output / result["prediction_path"])


def _residual_lookup(frame: pd.DataFrame, residual_mean: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for entity, group in frame.groupby("entity_id", sort=False):
        ordered = group.sort_values("origin")
        result[str(entity)] = (ordered["origin"].to_numpy(dtype=np.int64), ordered["residual"].to_numpy(dtype=np.float64) - residual_mean)
    return result


def mature_features(
    samples: pd.DataFrame,
    residuals: pd.DataFrame,
    h_steps: int,
    w_steps: int,
    delta: int,
    history: int,
    maximum_lags: int,
    residual_mean: float,
) -> tuple[np.ndarray, float]:
    count = min(maximum_lags, max(1, history // max(delta, 1)))
    offsets = np.unique(np.rint(np.linspace(0, max(0, history - delta), count)).astype(np.int64))
    lookup = _residual_lookup(residuals, residual_mean)
    result = np.zeros((len(samples), len(offsets)), dtype=np.float64)
    observed = np.zeros_like(result, dtype=bool)
    entities = samples["entity_id"].astype(str).to_numpy(); origins = samples["origin"].to_numpy(dtype=np.int64)
    for entity in np.unique(entities):
        mask = np.flatnonzero(entities == entity)
        if entity not in lookup: continue
        source_origins, source_values = lookup[entity]
        latest = origins[mask] - h_steps - w_steps
        queries = latest[:, None] - offsets[None, :]
        positions = np.searchsorted(source_origins, queries)
        valid = positions < len(source_origins)
        safe = np.minimum(positions, max(len(source_origins) - 1, 0))
        valid &= source_origins[safe] == queries
        values = np.zeros(queries.shape, dtype=np.float64); values[valid] = source_values[safe[valid]]
        result[mask] = values; observed[mask] = valid
    return result, float(observed.mean()) if observed.size else 0.0


def _fit_residual(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    alpha: float,
    family: str,
    train_physical: np.ndarray,
    evaluation_physical: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_x.mean(axis=0, dtype=np.float64); scale = train_x.std(axis=0, dtype=np.float64); scale[scale * scale < 1e-12] = 1.0
    x = (train_x - mean) / scale; z = (evaluation_x - mean) / scale
    projection = None; orthogonality = None
    if family == "ORTHOGONAL_MATURE_RESIDUAL_AR":
        against = np.column_stack([np.ones(len(x)), train_physical])
        x, orthogonality = residualize(x, against, relative_tolerance=1e-8)
        projection = np.linalg.lstsq(against, (train_x - mean) / scale, rcond=1e-10)[0]
        z = z - np.column_stack([np.ones(len(z)), evaluation_physical]) @ projection
    target_mean = float(np.mean(train_y, dtype=np.float64))
    coefficient, certificate = solve_certified(x, train_y - target_mean, alpha)
    prediction = z @ coefficient + target_mean
    return prediction, {"family": family, "mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(),
                        "intercept": target_mean, "alpha": alpha, "projection": None if projection is None else projection.tolist(),
                        "orthogonality": orthogonality, "numerical_certificate": certificate.to_json(),
                        "parameter_count": len(coefficient) + 1}


def _select_a_variant(oof:pd.DataFrame,validation_frame:pd.DataFrame,physical_oof_column:str,physical_validation_column:str,view:ViewSpec,config:dict[str,Any])->dict[str,Any]:
    work=oof.copy();work["residual"]=work["y_true"]-work[physical_oof_column]
    validation_work=validation_frame.copy();validation_work["residual"]=validation_work["y_true"]-validation_work[physical_validation_column]
    profiles=realized_state_profiles(view.head);alphas=[float(x) for x in config["A_module"]["ridge_alpha_grid"]]
    residual_mean=float(work["residual"].mean());usable_folds=sorted(int(x) for x in work["oof_fold"].unique())[1:]
    losses:dict[Any,list[float]]={"EXACT_ZERO":[]}
    for family in ("MATURE_RESIDUAL_AR","ORTHOGONAL_MATURE_RESIDUAL_AR"):
        for profile in profiles:
            for alpha in alphas:losses[(family,profile,alpha)]=[]
    for fold in usable_folds:
        fit=work[work["oof_fold"]<fold];evaluation=work[work["oof_fold"]==fold]
        y_fit=fit["residual"].to_numpy(dtype=np.float64);y_eval=evaluation["residual"].to_numpy(dtype=np.float64);losses["EXACT_ZERO"].append(float(np.mean(y_eval*y_eval)))
        for profile in profiles:
            x_fit,_=mature_features(fit,work,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
            x_eval,_=mature_features(evaluation,work,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
            for family in ("MATURE_RESIDUAL_AR","ORTHOGONAL_MATURE_RESIDUAL_AR"):
                for alpha in alphas:
                    pred,_=_fit_residual(x_fit,y_fit,x_eval,alpha,family,fit[physical_oof_column].to_numpy(),evaluation[physical_oof_column].to_numpy())
                    losses[(family,profile,alpha)].append(mse(y_eval,pred))
    def complexity(value):
        if value=="EXACT_ZERO":return(0,)
        family,profile,alpha=value;return(1,0 if family=="MATURE_RESIDUAL_AR" else 1,-alpha,profile[1],-profile[0])
    selection=one_se_select(losses,complexity,neutral="EXACT_ZERO",minimum_usable_folds=3);selected=selection.selected
    activation={"pass":False,"reason":"EXACT_ZERO_SELECTED"}
    if selected!="EXACT_ZERO":
        activation=practical_activation(losses["EXACT_ZERO"],losses[selected],minimum_relative_improvement=0.01,minimum_positive_fraction=0.75)
        if not activation["pass"]:selected="EXACT_ZERO"
    combined=pd.concat([work[["entity_id","origin","residual"]],validation_work[["entity_id","origin","residual"]]],ignore_index=True)
    if selected=="EXACT_ZERO":residual_prediction=np.zeros(len(validation_work));contract={"family":"EXACT_ZERO","parameter_count":0};observed=1.0
    else:
        family,profile,alpha=selected
        x_train,observed_train=mature_features(work,work,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
        x_validation,observed_validation=mature_features(validation_work,combined,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
        residual_prediction,contract=_fit_residual(x_train,work["residual"].to_numpy(),x_validation,alpha,family,work[physical_oof_column].to_numpy(),validation_work[physical_validation_column].to_numpy())
        contract["profile"]=list(profile);observed=min(observed_train,observed_validation)
    return {"selected_candidate":str(selected),"a_contract":contract,"one_se":selection.to_json(),"activation":activation,
            "fold_losses":{str(k):v for k,v in losses.items()},"raw_fold_losses":losses,"observed_mature_feature_fraction":observed,
            "residual_prediction":residual_prediction,"prediction":validation_work[physical_validation_column].to_numpy()+residual_prediction}


def run_a_view(shared: Path, project: Path, output: Path, view: ViewSpec) -> dict[str, Any]:
    started=time.time(); destination=output/"DEVELOPMENT"/"RESIDUAL_STATE"/view.head.head_id/view.availability_scenario/view.proxy_policy
    destination.mkdir(parents=True,exist_ok=True)
    try:
        config=load_frozen_config(project); _, physical_oof, physical_validation=_input_view_result(output,view)
        train=load_samples(shared,view,"train"); validation=load_samples(shared,view,"validation")
        oof=train.merge(physical_oof[["base_origin_id","physical_oof","physical_w_oof","oof_fold"]],on="base_origin_id",how="inner",validate="one_to_one")
        oof["residual"]=oof["y_true"]-oof["physical_w_oof"]
        validation_frame=validation.merge(physical_validation[["base_origin_id","physical_latent","y_pred"]].rename(columns={"y_pred":"physical_w"}),on="base_origin_id",how="inner",validate="one_to_one")
        validation_frame["residual"]=validation_frame["y_true"]-validation_frame["physical_w"]
        if len(validation_frame)!=len(validation): raise RuntimeError("dynamic/input-only base_origin_id mismatch")
        profiles=realized_state_profiles(view.head); alphas=[float(x) for x in config["A_module"]["ridge_alpha_grid"]]
        residual_mean=float(oof["residual"].mean()); usable_folds=sorted(int(x) for x in oof["oof_fold"].unique())[1:]
        losses:dict[Any,list[float]]={"EXACT_ZERO":[]}; observed_by_profile={}
        for family in ("MATURE_RESIDUAL_AR","ORTHOGONAL_MATURE_RESIDUAL_AR"):
            for profile in profiles:
                for alpha in alphas: losses[(family,profile,alpha)]=[]
        for fold in usable_folds:
            fit=oof[oof["oof_fold"]<fold]; evaluation=oof[oof["oof_fold"]==fold]
            y_fit=fit["residual"].to_numpy(dtype=np.float64); y_eval=evaluation["residual"].to_numpy(dtype=np.float64)
            losses["EXACT_ZERO"].append(float(np.mean(y_eval*y_eval,dtype=np.float64)))
            for profile in profiles:
                x_fit, observed_fit=mature_features(fit,oof,view.head.h_steps,view.head.w_steps,*profile,int(config["A_module"]["state_profile"]["maximum_lags"]),residual_mean)
                x_eval, observed_eval=mature_features(evaluation,oof,view.head.h_steps,view.head.w_steps,*profile,int(config["A_module"]["state_profile"]["maximum_lags"]),residual_mean)
                observed_by_profile[str(profile)]=[observed_fit,observed_eval]
                for family in ("MATURE_RESIDUAL_AR","ORTHOGONAL_MATURE_RESIDUAL_AR"):
                    for alpha in alphas:
                        pred,_=_fit_residual(x_fit,y_fit,x_eval,alpha,family,fit["physical_w_oof"].to_numpy(),evaluation["physical_w_oof"].to_numpy())
                        losses[(family,profile,alpha)].append(mse(y_eval,pred))
        def complexity(value):
            if value=="EXACT_ZERO": return (0,)
            family,profile,alpha=value; return (1,0 if family=="MATURE_RESIDUAL_AR" else 1,-alpha,profile[1],-profile[0])
        selection=one_se_select(losses,complexity,neutral="EXACT_ZERO",minimum_usable_folds=int(config["A_module"]["oof"]["minimum_usable_residual_folds"]))
        selected=selection.selected; activation={"pass":False,"reason":"EXACT_ZERO_SELECTED"}
        if selected!="EXACT_ZERO":
            activation=practical_activation(losses["EXACT_ZERO"],losses[selected],minimum_relative_improvement=0.01,minimum_positive_fraction=0.75)
            if not activation["pass"]: selected="EXACT_ZERO"
        combined=pd.concat([oof[["entity_id","origin","residual"]],validation_frame[["entity_id","origin","residual"]]],ignore_index=True)
        if selected=="EXACT_ZERO":
            residual_prediction=np.zeros(len(validation)); contract={"family":"EXACT_ZERO","parameter_count":0}; selected_observed=1.0
        else:
            family,profile,alpha=selected
            x_train,observed_train=mature_features(oof,oof,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
            x_validation,observed_validation=mature_features(validation_frame,combined,view.head.h_steps,view.head.w_steps,*profile,64,residual_mean)
            residual_prediction,contract=_fit_residual(x_train,oof["residual"].to_numpy(),x_validation,alpha,family,oof["physical_w_oof"].to_numpy(),validation_frame["physical_w"].to_numpy())
            contract["profile"]=list(profile); selected_observed=min(observed_train,observed_validation)
        prediction=validation_frame["physical_w"].to_numpy()+residual_prediction
        identity_variant=_select_a_variant(oof,validation_frame,"physical_oof","physical_latent",view,config)
        frame=validation_frame[["base_origin_id","view_sample_id","entity_id","origin","y_true"]].copy(); frame["physical_w"]=validation_frame["physical_w"]
        frame["residual_pred"]=residual_prediction; frame["y_pred"]=prediction; frame["model"]="PRISM_V2_PHYSICS_FIRST"; frame["dtype"]="float64"
        prediction_path=destination/"validation.parquet"; frame.to_parquet(prediction_path,index=False,compression="zstd")
        identity_frame=frame[["base_origin_id","view_sample_id","entity_id","origin","y_true"]].copy();identity_frame["y_pred"]=identity_variant["prediction"]
        identity_frame["model"]="PRISM_V2_K_JOINT_BASIS_A";identity_frame["dtype"]="float64"
        identity_path=destination/"validation_no_w_a.parquet";identity_frame.to_parquet(identity_path,index=False,compression="zstd")
        structural_claim=selected_observed>=float(config["A_module"]["oof"]["observed_mature_feature_fraction_for_structural_claim_min"])
        if contract.get("family")=="ORTHOGONAL_MATURE_RESIDUAL_AR": structural_claim &= bool(contract.get("orthogonality",{}).get("pass"))
        result={"status":"PASS","stage":"V5_RESIDUAL_STATE","target_head":view.head.head_id,"availability_scenario":view.availability_scenario,
                "proxy_policy":view.proxy_policy,"selected_candidate":str(selected),"a_contract":contract,"one_se":selection.to_json(),"activation":activation,
                "fold_losses":{str(k):v for k,v in losses.items()},"usable_residual_folds":len(usable_folds),"maturity_rule":"s_plus_h_plus_w_le_t",
                "observed_mature_feature_fraction":selected_observed,"structural_claim_allowed":bool(structural_claim),
                "identity_w_a":{"selected_candidate":identity_variant["selected_candidate"],"a_contract":identity_variant["a_contract"],"one_se":identity_variant["one_se"],
                                "activation":identity_variant["activation"],"fold_losses":identity_variant["fold_losses"],
                                "observed_mature_feature_fraction":identity_variant["observed_mature_feature_fraction"],
                                "prediction_path":str(identity_path.relative_to(output)),"prediction_sha256":sha256_file(identity_path)},
                "prediction_path":str(prediction_path.relative_to(output)),"prediction_sha256":sha256_file(prediction_path),"test_accessed":False,
                "elapsed_seconds":time.time()-started,**regression_metrics(frame["y_true"].to_numpy(),prediction)}
    except Exception as error:
        result={"status":"SOLVER_FAILED_RETAINED","stage":"V5_RESIDUAL_STATE","target_head":view.head.head_id,"availability_scenario":view.availability_scenario,
                "proxy_policy":view.proxy_policy,"test_accessed":False,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc(),"elapsed_seconds":time.time()-started}
    write_json(destination/"RESULT.json",result); return result


def run_v5_a(shared:Path,project:Path,output:Path,n_jobs:int)->dict[str,Any]:
    results=[];pending=[]
    for view in development_dynamic_views(shared):
        path=output/"DEVELOPMENT"/"RESIDUAL_STATE"/view.head.head_id/view.availability_scenario/view.proxy_policy/"RESULT.json"
        if path.is_file():
            prior=json.loads(path.read_text());
            if prior.get("status") in {"PASS","SOLVER_FAILED_RETAINED"}:results.append(prior);continue
        pending.append(view)
    results.extend(
        run_parallel(
            run_a_view,
            [(shared, project, output, view) for view in pending],
            n_jobs,
            per_worker_gib=3.0,
            label="V5_RESIDUAL_STATE",
        )
    )
    summary={"status":"PASS" if all(x["status"]=="PASS" for x in results) else "COMPLETED_WITH_RETAINED_FAILURES","stage":"V5_RESIDUAL_STATE",
             "views":len(results),"pass":sum(x["status"]=="PASS" for x in results),"activated":sum(x.get("selected_candidate")!="EXACT_ZERO" for x in results),
             "orthogonal":sum("ORTHOGONAL" in x.get("selected_candidate","") for x in results),"test_accessed":False}
    write_json(output/"DEVELOPMENT"/"RESIDUAL_STATE"/"SUMMARY.json",summary);return summary
