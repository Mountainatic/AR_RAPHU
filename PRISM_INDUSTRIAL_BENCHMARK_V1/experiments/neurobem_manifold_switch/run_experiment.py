from __future__ import annotations

import argparse
import multiprocessing as mp
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import NeuroBEMSource
from .model_bank import BankEntry, ModelBank
from .monitor import ManifoldTemplate, calibrate, component_scores, residual_score
from .prism_adapter import FrozenPrismAdapter, fit_local_adapter, geometry_features
from .rollout import evaluate


_WORKER_CONTEXT = {}


def _evaluate_trajectory_index(index: int):
    context = _WORKER_CONTEXT
    trajectory = context["trajectories"][index]
    values = []
    for route in context["cfg"]["routes"]:
        for ablation in context["cfg"]["ablations"]:
            key = f"{trajectory.trajectory_id}|{route}|{ablation}"
            result_path = context["run"] / "trajectories" / (key.replace("|", "__") + ".json")
            if key in context["completed"] and result_path.exists():
                values.append((key, result_path, json.loads(result_path.read_text()), None))
                continue
            local_bank = ModelBank(list(context["bank"].entries))
            result, log = evaluate(
                trajectory.frame, route, context["global_adapter"], context["template"],
                local_bank, context["calibration"], context["cfg"], ablation,
            )
            result["trajectory"] = trajectory.trajectory_id
            values.append((key, result_path, result, log))
    return values


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def hash_file(path: Path) -> str:
    h = sha256(path.read_bytes()).hexdigest()
    return h


def capped_features(trajectories, cap: int) -> np.ndarray:
    values = []
    for item in trajectories:
        x = geometry_features(item.frame)
        if len(x) > cap:
            x = x[np.linspace(0, len(x) - 1, cap, dtype=np.int64)]
        values.append(x)
    return np.concatenate(values)


def fit_assets(source: NeuroBEMSource, cfg: dict, run: Path) -> tuple[FrozenPrismAdapter, ManifoldTemplate, ModelBank, dict, dict]:
    fit_names, calibration_names, split_audit = source.train_parent_holdout(float(cfg["train_parent_fit_fraction"]))
    train_names = fit_names
    fit_trajectories = source.load("train", fit_names)
    global_adapter = fit_local_adapter(
        fit_trajectories, "GLOBAL_PRISM_R2_TRAIN_ONLY", cfg["w_family"], int(cfg["row_cap_per_trajectory"]),
        cfg["ridge_grid"], float(cfg["max_condition"]), float(cfg["max_kkt"]), int(cfg["history"]),
    )
    global_adapter.save(run / "checkpoints" / "global_prism_r2_train_only.json")
    n = int(cfg["local_model_trajectories"])
    groups = [train_names[:n], train_names[-n:]]
    local_entries = []
    for index, names in enumerate(groups):
        trajectories = source.load("train", names)
        adapter = fit_local_adapter(
            trajectories, f"LOCAL_PRISM_{index}", cfg["w_family"], int(cfg["row_cap_per_trajectory"]),
            cfg["ridge_grid"], float(cfg["max_condition"]), float(cfg["max_kkt"]), int(cfg["history"]),
        )
        features = capped_features(trajectories, int(cfg["row_cap_per_trajectory"]))
        template = ManifoldTemplate.fit(features, int(cfg["geometry_rank"]))
        acceptance = float(np.quantile(template.projection_score(features), cfg["monitor_quantile"])) * float(cfg["known_acceptance_multiplier"])
        local_entries.append(BankEntry(adapter, template, acceptance))
        adapter.save(run / "checkpoints" / f"local_prism_{index}.json")

    template_count = min(int(cfg["template_trajectories"]), len(train_names))
    template_names = [train_names[i] for i in np.linspace(0, len(train_names) - 1, template_count, dtype=np.int64)]
    template_train = source.load("train", template_names)
    global_features = capped_features(template_train, int(cfg["row_cap_per_trajectory"]))
    global_template = ManifoldTemplate.fit(global_features, int(cfg["geometry_rank"]))
    validation = source.load("train", calibration_names)
    residual_values, geometry_values, residual_raw = [], [], []
    for item in validation:
        target, prediction, origins = global_adapter.one_step(cfg["routes"][0], item.frame)
        residual_raw.append(target - prediction)
    scale = np.std(np.concatenate(residual_raw), axis=0)
    for item in validation:
        target, prediction, origins = global_adapter.one_step(cfg["routes"][0], item.frame)
        residual = residual_score(target, prediction, scale)
        features = geometry_features(item.frame)[origins]
        projection = global_template.projection_score(features)
        tangent = global_template.tangent_score(features, int(cfg["tangent_window"]), int(cfg["tangent_rank"]))
        residual_values.append(residual)
        geometry_values.append(0.7 * projection + 0.3 * tangent)
    thresholds = calibrate(np.concatenate(residual_values), np.concatenate(geometry_values), float(cfg["monitor_quantile"]))
    calibration = {"residual_scale": scale.tolist(), "thresholds": thresholds, "source": "TRAIN_PARENT_CHRONOLOGICAL_HOLDOUT_ONLY_R2"}
    atomic_json(run / "checkpoints" / "monitor_calibration.json", calibration)
    atomic_json(run / "checkpoints" / "r2_data_contract_audit.json", split_audit)
    return global_adapter, global_template, ModelBank(local_entries), calibration, split_audit


def plot_log(path: Path, log: pd.DataFrame, result: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(log.step, log.velocity_error, label="velocity")
    axes[0].plot(log.step, log.body_rate_error, label="body rate")
    axes[0].legend(); axes[0].set_ylabel("state error")
    axes[1].plot(log.step, log.combined_score, label="combined")
    axes[1].plot(log.step, log.detector_threshold, "--", label="threshold")
    axes[1].legend(); axes[1].set_ylabel("monitor")
    codes, _ = pd.factorize(log.active_model_id)
    axes[2].step(log.step, codes, where="post"); axes[2].set_ylabel("model ID"); axes[2].set_xlabel("100-Hz step")
    for ax in axes:
        for key, color in (("t_alarm", "orange"), ("t_switch", "green"), ("t_diverge", "red")):
            if result.get(key) is not None: ax.axvline(result[key], color=color, alpha=.5)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--release-root", type=Path, required=True); p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--global-contracts", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--partition", choices=("calibration", "test"), default="calibration")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); p.add_argument("--resume", type=Path)
    args = p.parse_args(argv); cfg = load_config(args.config); cfg.setdefault("w_family", "IDENTITY_CORRECTION")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA_EXPLICITLY_REQUESTED_BUT_UNAVAILABLE")
    run = args.resume or args.output_root / (cfg["protocol_id"] + "_" + time.strftime("%Y%m%dT%H%M%S"))
    run.mkdir(parents=True, exist_ok=True); (run / "trajectories").mkdir(exist_ok=True); (run / "figures").mkdir(exist_ok=True)
    source = NeuroBEMSource(args.data_root, args.split_manifest, args.release_root)
    if not source.audit()["all_frozen_identities_present"]: raise RuntimeError("FROZEN_SPLIT_IDENTITY_MISSING")
    repo = Path(__file__).resolve().parents[3]
    metadata = {"git_commit": subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip(),
        "config": str(args.config), "config_sha256": hash_file(args.config), "seed": cfg["seed"], "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__, "hostname": socket.gethostname(), "platform": platform.platform(),
        "device": "cpu" if args.device in ("auto", "cpu") else "cuda", "dataset": source.audit(), "partition": args.partition,
        "test_tuning_prohibited": True, "prism_core_modified": False,
        "parent_global_contracts": str(args.global_contracts), "parent_global_contracts_sha256": hash_file(args.global_contracts),
        "r2_global_contract_refitted_from_train_only": True}
    atomic_json(run / "metadata.json", metadata); atomic_json(run / "config.json", cfg)
    global_adapter, template, bank, calibration, split_audit = fit_assets(source, cfg, run)
    completed_path = run / "checkpoints" / "completed.json"
    completed = set(json.loads(completed_path.read_text())["keys"]) if completed_path.exists() else set()
    rows = []
    if args.partition == "calibration":
        _, calibration_names, _ = source.train_parent_holdout(float(cfg["train_parent_fit_fraction"]))
        trajectories = source.load("train", calibration_names)
    else:
        trajectories = source.load("test")
    workers = max(1, min(int(cfg["trajectory_workers"]), len(trajectories)))
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {"trajectories": trajectories, "cfg": cfg, "run": run, "completed": completed,
        "bank": bank, "global_adapter": global_adapter, "template": template, "calibration": calibration}
    multiprocessing_start = "serial"
    if workers == 1:
        evaluated = map(_evaluate_trajectory_index, range(len(trajectories)))
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("NEUROBEM_PROCESS_PARALLELISM_REQUIRES_LINUX_FORK")
        pool = mp.get_context("fork").Pool(processes=workers)
        evaluated = pool.imap(_evaluate_trajectory_index, range(len(trajectories)), chunksize=1)
        multiprocessing_start = "fork_cow"
    try:
        for trajectory_values in evaluated:
            for key, result_path, result, log in trajectory_values:
                if log is not None:
                    atomic_json(result_path, result); log.to_csv(result_path.with_suffix(".csv"), index=False)
                    if result["ablation"] in {"static", "combined_switch_reid"}: plot_log(run / "figures" / (result_path.stem + ".png"), log, result)
                    completed.add(key); atomic_json(completed_path, {"keys": sorted(completed)})
                rows.append(result)
    finally:
        if workers > 1:
            pool.close(); pool.join()
    _WORKER_CONTEXT = {}
    table = pd.DataFrame(rows)
    static_map = {(row.trajectory, row.route): row.t_diverge for row in table.itertuples() if row.ablation == "static"}
    table["t_diverge_static"] = [static_map[(row.trajectory, row.route)] for row in table.itertuples()]
    table["lead_time"] = [None if row.t_alarm is None or row.t_diverge_static is None else row.t_diverge_static - row.t_alarm for row in table.itertuples()]
    table["static_diverged"] = table.t_diverge_static.notna()
    table["divergence_avoided_vs_static"] = table.static_diverged & ~table.diverged
    table.to_csv(run / "per_trajectory.csv", index=False)
    aggregate = []
    for (route, ablation), part in table.groupby(["route", "ablation"]):
        lead = pd.to_numeric(part.lead_time, errors="coerce")
        aggregate.append({"route": route, "ablation": ablation, "trajectories": len(part), "divergence_rate": float(part.diverged.mean()),
            "alarm_rate": float(part.t_alarm.notna().mean()), "positive_lead_fraction": float((lead > 0).mean()), "median_lead_time": None if lead.dropna().empty else float(lead.median()),
            "mean_velocity_error": float(part.rollout_velocity_error.mean()), "mean_attitude_error": float(part.rollout_attitude_error.mean()),
            "static_divergence_rate": float(part.static_diverged.mean()), "divergence_avoided_count": int(part.divergence_avoided_vs_static.sum()),
            "false_alarm_total": int(((part.t_alarm.notna()) & ~part.static_diverged).sum()), "false_switch_total": int(((part.t_switch.notna()) & ~part.static_diverged).sum()), "new_models_created": int(part.new_models_created.sum())})
    summary = {"status": "COMPLETED", "partition": args.partition, "trajectory_count": len(trajectories), "rows": aggregate,
        "hypothesis": "t_alarm < t_diverge_static", "test_accessed": args.partition == "test", "test_used_for_tuning": False,
        "historical_exact_training_divergence_retained": True, "r2_data_contract": split_audit, "run_dir": str(run),
        "trajectory_workers": workers, "parallelism": multiprocessing_start, "blas_threads_per_worker": 1}
    atomic_json(run / "summary.json", summary); pd.DataFrame(aggregate).to_csv(run / "ablation_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
