from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .cpu_data import ViewSpec, load_samples, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_config import V2Paths, load_frozen_config
from .v2_selection import one_se_select
from .v2_views import development_dynamic_views, development_input_views


def _read(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if value.get("status")!="PASS":raise RuntimeError(f"prerequisite not PASS: {path}")
    return value


def _loss(result:dict[str,Any],selected_key:str,field:str="fold_losses")->list[float]:
    losses=result[field];key=result[selected_key]
    if key not in losses:raise KeyError((key,list(losses)[:3]))
    return [float(value) for value in losses[key]]


def _materialize(samples:pd.DataFrame,source:Path,destination:Path,model:str)->dict[str,Any]:
    prediction=pd.read_parquet(source)[["base_origin_id","y_pred"]]
    frame=samples[["base_origin_id","view_sample_id","entity_id","origin","y_true"]].merge(prediction,on="base_origin_id",how="inner",validate="one_to_one")
    if len(frame)!=len(samples):raise RuntimeError(f"assembly candidate alignment mismatch: {model}")
    frame["model"]=model;frame["dtype"]="float64";destination.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(destination,index=False,compression="zstd")
    return {"path":str(destination),"sha256":sha256_file(destination),**regression_metrics(frame["y_true"].to_numpy(),frame["y_pred"].to_numpy())}


def _view_card(shared:Path,output:Path,view:ViewSpec,dynamic:bool)->dict[str,Any]:
    c=_read(output/"DEVELOPMENT"/"JOINT_BASIS"/view.head.head_id/view.proxy_policy/"RESULT.json")
    w=_read(output/"DEVELOPMENT"/"WIENER"/view.head.head_id/view.proxy_policy/"RESULT.json")
    candidates={};losses={}
    croot=output/"DEVELOPMENT"/"JOINT_BASIS"/view.head.head_id/view.proxy_policy
    candidates["K_COMPRESSED"]=output/c["component_files"]["K_COMPRESSED"]["path"]
    candidates["K_JOINT_BASIS"]=output/c["component_files"]["K_JOINT_BASIS"]["path"]
    losses["K_COMPRESSED"]=_loss(c,"selected",field="candidate_fold_losses") if False else c["candidate_fold_losses"][c["compressed_selection"]["selected"]]
    losses["K_JOINT_BASIS"]=c["candidate_fold_losses"][c["joint_selection"]["selected"]]
    candidates["K_JOINT_BASIS_W"]=output/w["prediction_path"]
    losses["K_JOINT_BASIS_W"]=w["fold_losses"][w["one_se"]["selected"]]
    availability=view.availability_scenario
    if dynamic:
        state=_read(output/"DEVELOPMENT"/"STATE_ONLY"/view.head.head_id/availability/"RESULT.json")
        a=_read(output/"DEVELOPMENT"/"RESIDUAL_STATE"/view.head.head_id/availability/view.proxy_policy/"RESULT.json")
        candidates["A_ONLY"]=output/state["prediction_path"]
        losses["A_ONLY"]=state["fold_losses"][state["one_se"]["selected"]]
        candidates["K_JOINT_BASIS_A"]=output/a["identity_w_a"]["prediction_path"]
        losses["K_JOINT_BASIS_A"]=a["identity_w_a"]["fold_losses"][a["identity_w_a"]["one_se"]["selected"]]
        candidates["K_JOINT_BASIS_W_A"]=output/a["prediction_path"]
        losses["K_JOINT_BASIS_W_A"]=a["fold_losses"][a["one_se"]["selected"]]
        if c["selected_family"]=="SPARSE_PAIRWISE_ANOVA_MAX3":
            candidates["K_PAIRWISE_W_A"]=output/a["prediction_path"];losses["K_PAIRWISE_W_A"]=losses["K_JOINT_BASIS_W_A"]
    order=["A_ONLY","K_COMPRESSED","K_JOINT_BASIS","K_JOINT_BASIS_W","K_JOINT_BASIS_A","K_JOINT_BASIS_W_A","K_PAIRWISE_W_A"]
    selection=one_se_select(losses,lambda value:(order.index(value),),minimum_usable_folds=3)
    selected=selection.selected;samples=load_samples(shared,view,"validation")
    root=output/"ASSEMBLY_CARDS"/view.head.head_id/view.information_set/availability/view.proxy_policy
    files={}
    for model,source in candidates.items():
        files[model]=_materialize(samples,source,root/f"{model}.validation.parquet",model)
        files[model]["path"]=str((root/f"{model}.validation.parquet").relative_to(output))
    card={"status":"PASS","stage":"V6_ASSEMBLY_SELECTION","dataset":view.head.dataset,"task":view.head.task_id,"target_head":view.head.head_id,
          "information_set":view.information_set,"availability_scenario":availability,"proxy_policy":view.proxy_policy,"selected_assembly":selected,
          "one_se":selection.to_json(),"candidate_fold_losses":losses,"candidate_predictions":files,"selected_prediction":files[selected],"test_accessed":False}
    write_json(root/"ASSEMBLY_CARD.json",card);return card


def run_v6_assembly(shared:Path,project:Path,output:Path)->dict[str,Any]:
    load_frozen_config(project)
    summary_path=output/"ASSEMBLY_CARDS"/"SUMMARY.json"
    if summary_path.is_file():
        prior=json.loads(summary_path.read_text())
        card_paths=sorted((output/"ASSEMBLY_CARDS").rglob("ASSEMBLY_CARD.json"))
        if (
            prior.get("status")=="PASS"
            and prior.get("test_accessed") is False
            and len(card_paths)==int(prior.get("cards",-1))
            and all(json.loads(path.read_text()).get("status")=="PASS" for path in card_paths)
        ):
            return prior
    cards=[]
    for view in development_input_views(shared):cards.append(_view_card(shared,output,view,False))
    for view in development_dynamic_views(shared):cards.append(_view_card(shared,output,view,True))
    summary={"status":"PASS","stage":"V6_ASSEMBLY_SELECTION","cards":len(cards),"selection_counts":{},"test_accessed":False}
    for card in cards:summary["selection_counts"][card["selected_assembly"]]=summary["selection_counts"].get(card["selected_assembly"],0)+1
    write_json(output/"ASSEMBLY_CARDS"/"SUMMARY.json",summary);return summary


def freeze_g3(paths:V2Paths)->dict[str,Any]:
    config=load_frozen_config(paths.project)
    audits=[]
    baseline_summary=paths.output/"BASELINE_DEVELOPMENT"/"SUMMARY.json"
    if not baseline_summary.is_file() or json.loads(baseline_summary.read_text()).get("status") not in {"PASS","PASS_WITH_RETAINED_FAILURES"}:
        raise RuntimeError("Level C baseline development must complete before G3")
    for root_name in ("DEVELOPMENT","ASSEMBLY_CARDS","BASELINE_DEVELOPMENT"):
        for path in sorted((paths.output/root_name).rglob("*.json")):
            audits.append({"path":str(path.relative_to(paths.output)),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    cards=[]
    for path in sorted((paths.output/"ASSEMBLY_CARDS").rglob("ASSEMBLY_CARD.json")):
        card=json.loads(path.read_text());cards.append({key:card[key] for key in ("target_head","information_set","availability_scenario","proxy_policy","selected_assembly")})
    commit=subprocess.check_output(["git","-C",str(paths.project.parent),"rev-parse","HEAD"],text=True).strip()
    inheritance=json.loads((paths.output/"FREEZE"/"DATA_INHERITANCE_AUDIT.json").read_text())
    manifest={"status":"PASS","stage":"G3_FINAL_FREEZE","protocol_id":config["protocol_id"],"code_commit":commit,
              "theory_sha256":sha256_file(paths.plan/"PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md"),
              "protocol_sha256":sha256_file(paths.plan/"PRISM_V2_MODULAR_CPU_BENCHMARK_PROTOCOL.md"),
              "config_sha256":sha256_file(paths.config_path),"c1_inheritance_audit_sha256":sha256_file(paths.output/"FREEZE"/"DATA_INHERITANCE_AUDIT.json"),
              "c6_summary_sha256":next(item["observed"] for item in inheritance["checks"] if item["name"]=="C6_V2_SUMMARY"),
              "development_results":audits,"assembly_cards":cards,"test_accessed":False}
    write_json(paths.output/"FREEZE"/"V2_FINAL_FREEZE_MANIFEST.json",manifest);return manifest
