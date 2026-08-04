from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .c2_models import run_job as run_c2_job
from .c3_models import run_job as run_c3_job
from .c4_prism import _run_channel, _run_joint_and_ablations
from .c5_models import _run_view as run_c5_view
from .c6_full_final import _evaluate_baseline_job, _evaluate_prism_job
from .cpu_data import ViewSpec, input_columns
from .cpu_data import sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_views import development_dynamic_views, development_input_views, evaluation_level
from .v2_runtime import run_parallel


def _level_c_input_views(shared:Path)->list[ViewSpec]:
    return [view for view in development_input_views(shared) if evaluation_level(view,shared)=="LEVEL_C_CONFIRMATION"]


def _level_c_dynamic_views(shared:Path)->list[ViewSpec]:
    return [view for view in development_dynamic_views(shared) if evaluation_level(view,shared)=="LEVEL_C_CONFIRMATION"]


def _run_parallel(
    function:Callable[...,dict[str,Any]],
    jobs:list[tuple[Any,...]],
    n_jobs:int,
    *,
    per_worker_gib:float=2.5,
    label:str|None=None,
)->list[dict[str,Any]]:
    return run_parallel(
        function,
        jobs,
        n_jobs,
        per_worker_gib=per_worker_gib,
        label=label or f"BASELINE:{function.__name__}",
    )


def _complete(path:Path)->dict[str,Any]|None:
    if not path.is_file():return None
    value=json.loads(path.read_text())
    return value if value.get("status") in {"PASS","FAILED_RETAINED"} else None


def _normalize_final_prediction(result:dict[str,Any])->dict[str,Any]:
    if result.get("status")!="PASS":return result
    path=Path(result["prediction_path"]);frame=pd.read_parquet(path)
    frame["assembly_id"]=frame["model"];frame["support_flag"]="INHERITED_TRAIN_SUPPORT_AUDIT_PENDING";frame["ood_flag"]=frame["split"].astype(str).eq("ood")
    frame.to_parquet(path,index=False,compression="zstd");result["prediction_sha256"]=sha256_file(path);result.update(regression_metrics(frame["y_true"].to_numpy(),frame["y_pred"].to_numpy()))
    row=frame.iloc[0];audit_path=path.parents[3]/"MODEL_AUDIT"/str(row["target_head"])/str(row["split"])/f"{row['information_set']}__{row['model']}.json"
    if audit_path.is_file():
        audit=json.loads(audit_path.read_text());audit.update({key:result[key] for key in ("prediction_sha256","mse","rmse","mae")});write_json(audit_path,audit)
    return result


def run_level_c_baseline_development(shared:Path,project:Path,output:Path,n_jobs:int)->dict[str,Any]:
    root=output/"BASELINE_DEVELOPMENT";c2=root/"C2";c3=root/"C3";c4=root/"C4";c5=root/"C5"
    input_views=_level_c_input_views(shared);dynamic_views=_level_c_dynamic_views(shared);results=[]

    jobs=[]
    for view in input_views:
        for model in ("DPLS","XGBOOST"):
            path=c2/"PREDICTIONS"/model/view.relative_root/"RESULT.json";prior=_complete(path)
            if prior is None:jobs.append((shared,project,c2,view,model))
            else:results.append(prior)
    results.extend(_run_parallel(run_c2_job,jobs,n_jobs))

    for models in (("AR",), ("ARX",), ("LINEAR_NARX",), ("PARALLEL_HAMMERSTEIN",), ("HAMMERSTEIN_WIENER",)):
        jobs=[]
        for view in dynamic_views:
            for model in models:
                path=c3/"PREDICTIONS"/model/view.relative_root/"RESULT.json";prior=_complete(path)
                if prior is None:jobs.append((shared,project,c3,view,model))
                else:results.append(prior)
        model_label = models[0]
        heavy = model_label in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}
        if heavy:
            # Long Metro/PMSM/TEP Hammerstein jobs peak together.  A mixed
            # ten-job pool repeatedly reached the platform memory watchdog
            # without a cgroup OOM record.  Dataset-sized pools preserve all
            # useful within-dataset parallelism while separating those peaks.
            grouped:dict[str,list[tuple[Any,...]]]={}
            for job in jobs:
                grouped.setdefault(job[3].head.dataset,[]).append(job)
            for dataset,dataset_jobs in sorted(grouped.items(),key=lambda item:(-len(item[1]),item[0])):
                results.extend(_run_parallel(
                    run_c3_job,
                    dataset_jobs,
                    min(n_jobs,len(dataset_jobs)),
                    per_worker_gib=3.5,
                    label=f"BASELINE:C3:{model_label}:{dataset}",
                ))
        else:
            results.extend(_run_parallel(
                run_c3_job,
                jobs,
                n_jobs,
                per_worker_gib=2.5,
                label=f"BASELINE:C3:{model_label}",
            ))

    jobs=[]
    for view in input_views:
        for channel in input_columns(shared,view.head.task_id,view.proxy_policy):
            path=c4/"CHANNELS"/view.head.head_id/view.proxy_policy/channel/"RESULT.json";prior=_complete(path)
            if prior is None:jobs.append((shared,project,c4,view,channel))
            else:results.append(prior)
    results.extend(_run_parallel(_run_channel,jobs,n_jobs))

    jobs=[]
    for view in input_views:
        path=c4/"JOINT"/view.head.head_id/view.proxy_policy/"RESULT.json";prior=_complete(path)
        if prior is None:jobs.append((shared,project,c4,view))
        else:results.append(prior)
    results.extend(_run_parallel(_run_joint_and_ablations,jobs,n_jobs))

    jobs=[]
    for view in dynamic_views:
        path=c5/view.head.head_id/view.proxy_policy/"RESULT.json";prior=_complete(path)
        if prior is None:jobs.append((shared,project,c3,c4,c5,view,1))
        else:results.append(prior)
    results.extend(_run_parallel(run_c5_view,jobs,min(n_jobs,8)))
    passed=sum(value.get("status")=="PASS" for value in results)
    summary={"status":"PASS" if passed==len(results) else "PASS_WITH_RETAINED_FAILURES","stage":"LEVEL_C_BASELINE_DEVELOPMENT",
             "results":len(results),"pass":passed,"failed_retained":len(results)-passed,"test_accessed":False}
    write_json(root/"SUMMARY.json",summary);return summary


def evaluate_level_c_baselines(shared:Path,project:Path,output:Path,n_jobs:int)->dict[str,Any]:
    root=output/"BASELINE_DEVELOPMENT";final=output/"BASELINE_PREDICTIONS"/"LEVEL_C_CONFIRMATION"
    c2=root/"C2";c3=root/"C3";c4=root/"C4";c5=root/"C5";jobs=[]
    input_views=_level_c_input_views(shared);dynamic_views=_level_c_dynamic_views(shared)
    for view in input_views:
        core=max(view.head.h_steps*8,view.head.w_steps)
        for split in ("test","ood"):
            if not (shared/"sample_ids"/view.relative_root/f"{split}.parquet").is_file():continue
            for model in ("PERSISTENCE","DPLS","XGBOOST","PARALLEL_HAMMERSTEIN","HAMMERSTEIN_WIENER"):
                jobs.append((shared,project,c2,c3,final/view.availability_scenario/view.proxy_policy,view,split,model,core))
    results=_run_parallel(_evaluate_baseline_job,jobs,n_jobs)
    jobs=[]
    for view in dynamic_views:
        core=max(view.head.h_steps*8,view.head.w_steps)
        for split in ("test","ood"):
            if not (shared/"sample_ids"/view.relative_root/f"{split}.parquet").is_file():continue
            for model in ("PERSISTENCE","AR","ARX","LINEAR_NARX"):
                jobs.append((shared,project,c2,c3,final/view.availability_scenario/view.proxy_policy,view,split,model,core))
    results.extend(_run_parallel(_evaluate_baseline_job,jobs,n_jobs))
    prism_jobs=[]
    for view in input_views:
        core=max(view.head.h_steps*8,view.head.w_steps)
        for split in ("test","ood"):
            if (shared/"sample_ids"/view.relative_root/f"{split}.parquet").is_file():prism_jobs.append((shared,project,c3,c4,c5,final/view.availability_scenario/view.proxy_policy,view,split,core))
    nested = run_parallel(
        _evaluate_prism_job,
        [(job,) for job in prism_jobs],
        min(n_jobs, 8),
        per_worker_gib=6.0,
        label="LEVEL_C_PRISM_EVALUATION",
    )
    for value in nested:
        results.extend(value)
    results=[_normalize_final_prediction(value) for value in results];passed=sum(value.get("status")=="PASS" for value in results)
    summary={"status":"PASS" if passed==len(results) else "COMPLETED_WITH_RETAINED_FAILURES","stage":"LEVEL_C_BASELINE_EVALUATION",
             "jobs":len(results),"pass":passed,"failed_retained":len(results)-passed}
    write_json(final/"SUMMARY.json",summary);return summary
