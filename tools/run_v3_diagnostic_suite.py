#!/usr/bin/env python3
"""Schedule frozen D1--D6 jobs and apply only preregistered decisions."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.diagnostics.config import load_diagnostic_config  # noqa: E402


RESULT_ROOT = PROJECT_ROOT / "results" / "v3_diagnostics"
JOB_SCRIPT = PROJECT_ROOT / "tools" / "run_v3_diagnostic_job.py"
ORDER = ("D1", "D2", "D3", "D4", "D5", "D6")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_summary(
    experiment: str, seed: int, variant: str, *, horizon: int = 1
) -> dict[str, Any]:
    root = RESULT_ROOT / experiment
    if experiment == "D3":
        root = root / f"horizon_{horizon}"
    path = root / f"seed_{seed}" / variant / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def job_command(job: tuple[str, str, int, int], device: str) -> list[str]:
    experiment, variant, seed, horizon = job
    return [
        sys.executable,
        str(JOB_SCRIPT),
        "--experiment",
        experiment,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--device",
        device,
    ]


def run_one(
    job: tuple[str, str, int, int], device: str
) -> tuple[tuple[str, str, int, int], int, str]:
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"
    result = subprocess.run(
        job_command(job, device),
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return job, result.returncode, result.stdout


def run_pool(
    jobs: list[tuple[str, str, int, int]],
    *,
    device: str,
    workers: int,
) -> tuple[
    list[tuple[str, str, int, int]],
    list[tuple[str, str, int, int]],
    bool,
]:
    pending = list(jobs)
    completed: list[tuple[str, str, int, int]] = []
    failed: list[tuple[str, str, int, int]] = []
    oom = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active: dict[concurrent.futures.Future, tuple[str, str, int, int]] = {}
        while pending and len(active) < workers:
            job = pending.pop(0)
            active[pool.submit(run_one, job, device)] = job
        while active:
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job = active.pop(future)
                _, returncode, output = future.result()
                print(output, end="")
                if returncode == 0:
                    completed.append(job)
                else:
                    failed.append(job)
                    if "CUDA out of memory" in output or "CUDA OOM" in output:
                        oom = True
            if not oom:
                while pending and len(active) < workers:
                    job = pending.pop(0)
                    active[pool.submit(run_one, job, device)] = job
    return completed, failed + pending, oom


def execute_jobs(
    jobs: list[tuple[str, str, int, int]],
    *,
    device: str,
    config: dict[str, Any],
) -> None:
    workers = config["runtime"]["gpu_workers"] if device == "cuda" else 1
    _, unfinished, oom = run_pool(jobs, device=device, workers=workers)
    if oom and device == "cuda":
        print(
            "CUDA OOM detected; retrying only unfinished jobs with "
            f"{config['runtime']['oom_fallback_workers']} workers."
        )
        _, unfinished, second_oom = run_pool(
            unfinished,
            device=device,
            workers=config["runtime"]["oom_fallback_workers"],
        )
        if second_oom:
            raise RuntimeError("CUDA OOM persisted after the single allowed fallback.")
    if unfinished:
        raise RuntimeError(f"Diagnostic jobs failed: {unfinished}")


def jobs_for(experiment: str, config: dict[str, Any]):
    seeds = (
        config["common"]["d6_seeds"]
        if experiment == "D6"
        else config["common"]["seeds"]
    )
    variants = {
        "D1": config["D1"]["variants"],
        "D2": config["D2"]["variants"],
        "D3": config["D3"]["variants"],
        "D4": config["D4"]["variants"],
        "D5": ["gate_path"],
        "D6": ["gradient_timeline"],
    }[experiment]
    return [
        (experiment, variant, seed, config["common"]["primary_horizon"])
        for seed in seeds
        for variant in variants
    ]


def aggregate_d1(config: dict[str, Any]) -> dict[str, Any]:
    oracle_pass = 0
    learned_pass = 0
    rows = []
    for seed in config["common"]["seeds"]:
        oracle = read_summary("D1", seed, "rank2_oracle_q")
        learned = read_summary(
            "D1", seed, "rank2_learned_q_truth_init"
        )
        oracle_ok = (
            oracle["validation_r2"] >= 0.95
            and oracle["mean_surface_nrmse"] <= 0.20
            and oracle["mean_surface_correlation"] >= 0.90
        )
        learned_ok = (
            learned["validation_rmse_scaled"]
            <= 1.10 * oracle["validation_rmse_scaled"]
            and learned["mean_w1"] <= 3.0
            and learned["mean_surface_nrmse"] <= 0.25
        )
        oracle_pass += int(oracle_ok)
        learned_pass += int(learned_ok)
        rows.append(
            {
                "seed": seed,
                "oracle_pass": oracle_ok,
                "learned_q_pass": learned_ok,
            }
        )
    status = (
        "D1_CAPACITY_FAIL"
        if oracle_pass < 4
        else (
            "D1_LAG_OPTIMIZATION_FAIL"
            if learned_pass < 4
            else "D1_CAPACITY_PASS"
        )
    )
    return {
        "status": status,
        "oracle_pass_seed_count": oracle_pass,
        "learned_q_pass_seed_count": learned_pass,
        "seeds": rows,
    }


def aggregate_d2(config: dict[str, Any]) -> dict[str, Any]:
    confirmed = 0
    rows = []
    for seed in config["common"]["seeds"]:
        rank1 = read_summary("D2", seed, "rank1_free_q")
        rank2 = read_summary("D2", seed, "rank2_free_q")
        rmse_gain = (
            rank1["validation_rmse_scaled"]
            - rank2["validation_rmse_scaled"]
        ) / rank1["validation_rmse_scaled"]
        surface_gain = (
            rank1["mean_surface_nrmse"] - rank2["mean_surface_nrmse"]
        ) / rank1["mean_surface_nrmse"]
        seed_confirmed = surface_gain >= 0.30 and (
            rmse_gain >= 0.05 or abs(rmse_gain) < 0.05
        )
        confirmed += int(seed_confirmed)
        rows.append(
            {
                "seed": seed,
                "g_rmse": rmse_gain,
                "g_surface": surface_gain,
                "blind_spot": seed_confirmed,
            }
        )
    return {
        "status": (
            "D2_RANK1_BLIND_SPOT_CONFIRMED"
            if confirmed >= 4
            else "D2_RANK1_BLIND_SPOT_NOT_CONFIRMED"
        ),
        "confirmed_seed_count": confirmed,
        "seeds": rows,
    }


def d3_requires_extension(config: dict[str, Any]) -> bool:
    failures = sum(
        read_summary("D3", seed, "residual_rank2_free_q", horizon=1)[
            "validation_innovation_r2"
        ]
        < config["D3"]["conditional_extension_trigger"][
            "innovation_r2_threshold"
        ]
        for seed in config["common"]["seeds"]
    )
    return (
        failures
        >= config["D3"]["conditional_extension_trigger"][
            "minimum_failed_seed_count"
        ]
    )


def aggregate_d3(config: dict[str, Any], *, extended: bool) -> dict[str, Any]:
    h1_values = [
        read_summary("D3", seed, "residual_rank2_free_q", horizon=1)[
            "validation_innovation_r2"
        ]
        for seed in config["common"]["seeds"]
    ]
    h1_median = sorted(h1_values)[len(h1_values) // 2]
    longer: dict[int, float] = {}
    if extended:
        for horizon in config["common"]["conditional_horizons"]:
            values = [
                read_summary(
                    "D3",
                    seed,
                    "residual_rank2_free_q",
                    horizon=horizon,
                )["validation_innovation_r2"]
                for seed in config["common"]["seeds"]
            ]
            longer[horizon] = sorted(values)[len(values) // 2]
    threshold = config["D3"]["conditional_extension_trigger"][
        "innovation_r2_threshold"
    ]
    if h1_median >= threshold:
        status = "D3_X_INFORMATION_REMAINS_AT_H1"
    elif any(value >= threshold for value in longer.values()):
        status = "D3_X_INFORMATION_EMERGES_AT_LONGER_HORIZON"
    else:
        status = "D3_AR_MEDIATES_MOST_X_INFORMATION"
    return {
        "status": status,
        "h1_median_innovation_r2": h1_median,
        "conditional_extension_run": extended,
        "longer_horizon_median_innovation_r2": longer,
    }


def aggregate_d4(config: dict[str, Any]) -> dict[str, Any]:
    shortcut_count = 0
    rows = []
    for seed in config["common"]["seeds"]:
        simultaneous = read_summary("D4", seed, "simultaneous")
        x_first = read_summary("D4", seed, "x_first")
        energy_gain = (
            x_first["true_support_energy_fraction"]
            - simultaneous["true_support_energy_fraction"]
        )
        response_improvement = (
            simultaneous["mean_active_response_nrmse"]
            - x_first["mean_active_response_nrmse"]
        ) / simultaneous["mean_active_response_nrmse"]
        rmse_ok = (
            x_first["validation_rmse_scaled"]
            <= 1.02 * simultaneous["validation_rmse_scaled"]
        )
        shortcut = (
            energy_gain >= 0.20
            and response_improvement >= 0.20
            and rmse_ok
        )
        shortcut_count += int(shortcut)
        rows.append(
            {
                "seed": seed,
                "energy_fraction_gain": energy_gain,
                "response_nrmse_improvement": response_improvement,
                "validation_rmse_within_2_percent": rmse_ok,
                "shortcut": shortcut,
            }
        )
    return {
        "status": (
            "D4_AR_SHORTCUT_CONFIRMED"
            if shortcut_count >= 4
            else "D4_TRAINING_ORDER_NOT_PRIMARY"
        ),
        "shortcut_seed_count": shortcut_count,
        "seeds": rows,
    }


def aggregate_d5(config: dict[str, Any]) -> dict[str, Any]:
    successes = sum(
        read_summary("D5", seed, "gate_path")[
            "path_contains_recoverable_support"
        ]
        for seed in config["common"]["seeds"]
    )
    return {
        "status": (
            "D5_SCALE_NORMALIZED_GATE_PATH_SUCCESS"
            if successes >= 4
            else "D5_GATE_PATH_STILL_NOT_SEPARABLE"
        ),
        "success_seed_count": successes,
    }


def aggregate_d6(config: dict[str, Any]) -> dict[str, Any]:
    summaries = [
        read_summary("D6", seed, "gradient_timeline")
        for seed in config["common"]["d6_seeds"]
    ]
    starvation = sum(
        summary["starved_true_variable_count"] >= 2 for summary in summaries
    )
    collapse = sum(summary["proximal_collapse"] for summary in summaries)
    labels = []
    if starvation >= 2:
        labels.append("D6_GRADIENT_STARVATION_CONFIRMED")
    else:
        labels.append("D6_NOT_A_GRADIENT_MAGNITUDE_PROBLEM")
    if collapse >= 2:
        labels.append("D6_PROXIMAL_COLLAPSE_CONFIRMED")
    return {
        "status": "+".join(labels),
        "starvation_seed_count": starvation,
        "proximal_collapse_seed_count": collapse,
    }


AGGREGATORS = {
    "D1": aggregate_d1,
    "D2": aggregate_d2,
    "D4": aggregate_d4,
    "D5": aggregate_d5,
    "D6": aggregate_d6,
}


def save_aggregate(experiment: str, payload: dict[str, Any]) -> None:
    atomic_json(RESULT_ROOT / experiment / "aggregate_summary.json", payload)


def load_aggregate(experiment: str) -> dict[str, Any]:
    return json.loads(
        (RESULT_ROOT / experiment / "aggregate_summary.json").read_text(
            encoding="utf-8"
        )
    )


def write_final_decision() -> None:
    statuses = {
        experiment: load_aggregate(experiment)["status"]
        for experiment in ORDER
    }
    primary = "NO_FROZEN_FAILURE_CLASS_TRIGGERED"
    secondary: list[str] = []
    changes: list[str] = []
    if statuses["D1"] == "D1_CAPACITY_FAIL":
        primary = "RANK2_MODEL_CAPACITY_OR_IMPLEMENTATION"
        changes.append("FIX_RANK2_MODEL_BEFORE_ANY_SCREENING_CHANGE")
    elif (
        statuses["D1"] == "D1_CAPACITY_PASS"
        and statuses["D2"] == "D2_RANK1_BLIND_SPOT_CONFIRMED"
    ):
        primary = "SCHEME_A_RANK1_HARD_GATE"
        changes.append("ADD_SCHEME_B_RESCUE_FOR_A_REJECTED_VARIABLES")
    elif statuses["D3"] == "D3_AR_MEDIATES_MOST_X_INFORMATION":
        primary = "CONDITIONAL_INFORMATION_LIMIT"
        changes.append("SEPARATE_PREDICTIVE_SUPPORT_FROM_GENERATIVE_SUPPORT")
    elif statuses["D4"] == "D4_AR_SHORTCUT_CONFIRMED":
        primary = "JOINT_OPTIMIZATION_SHORTCUT"
        changes.append("X_FIRST_OR_RESIDUALIZED_CURRICULUM")
    if statuses["D5"] == "D5_SCALE_NORMALIZED_GATE_PATH_SUCCESS":
        secondary.append("GROUP_PROX_SCALE_MISMATCH")
        changes.append("NORMALIZED_SCALAR_GATE_SELECTION")
    if "D6_GRADIENT_STARVATION_CONFIRMED" in statuses["D6"]:
        secondary.append("GRADIENT_STARVATION")
        changes.append("DELAYED_SPARSITY_AND_BRANCHWISE_OPTIMIZATION")
    if not changes:
        changes.append("NO_ARCHITECTURE_CHANGE_SUPPORTED")

    rows = [
        {"experiment": experiment, "status": statuses[experiment]}
        for experiment in ORDER
    ]
    csv_path = RESULT_ROOT / "diagnostic_summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "status"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    fields = {
        **{f"{experiment}_STATUS": statuses[experiment] for experiment in ORDER},
        "PRIMARY_FAILURE_CLASS": primary,
        "SECONDARY_FAILURE_CLASS": (
            "+".join(secondary) if secondary else "NONE"
        ),
        "SUPPORTED_NEXT_ARCHITECTURE_CHANGE": "+".join(changes),
        "UNSUPPORTED_CHANGES": (
            "V2_RERUN+PREREGISTRATION_CHANGE+UNMAPPED_ARCHITECTURE_CHANGE"
        ),
    }
    decision = "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"
    temporary = RESULT_ROOT / "DIAGNOSTIC_DECISION.md.tmp"
    temporary.write_text(decision, encoding="utf-8")
    os.replace(temporary, RESULT_ROOT / "DIAGNOSTIC_DECISION.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--experiment", choices=ORDER)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_diagnostic_config()
    experiments = ORDER if args.all else (args.experiment,)
    for experiment in experiments:
        execute_jobs(
            jobs_for(experiment, config),
            device=args.device,
            config=config,
        )
        if experiment == "D3":
            extended = d3_requires_extension(config)
            if extended:
                conditional_jobs = [
                    ("D3", variant, seed, horizon)
                    for horizon in config["common"]["conditional_horizons"]
                    for seed in config["common"]["seeds"]
                    for variant in config["D3"]["variants"]
                ]
                execute_jobs(
                    conditional_jobs,
                    device=args.device,
                    config=config,
                )
            aggregate = aggregate_d3(config, extended=extended)
        else:
            aggregate = AGGREGATORS[experiment](config)
        save_aggregate(experiment, aggregate)
        print(f"{experiment}: {aggregate['status']}")
    if args.all:
        write_final_decision()


if __name__ == "__main__":
    main()
