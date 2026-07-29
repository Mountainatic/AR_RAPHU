#!/usr/bin/env python3
"""Run the frozen CZ R3 stages with one active ORSS CUDA task."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch

from ar_raphu.cz_real.orss_r3 import (
    evaluate_frozen_configuration,
    rank_profile_configuration,
    run_development_task,
)
from ar_raphu.cz_real.protocol import (
    DIRECT_HORIZONS,
    LX_GRID,
    LY_GRID,
    load_furnace_a,
)
from ar_raphu.cz_real.r3_history import history_complexity_key
from ar_raphu.orss.diagnostics import write_json
from ar_raphu.orss.penalties import PenaltyWeights


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _furnace_a(raw_dir: Path, filename: str) -> Path:
    requested = raw_dir / filename
    if requested.exists():
        return requested
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _history_specs(config: dict[str, object]):
    return [
        (horizon, L_x, L_y)
        for horizon in config["horizons"]
        for L_x in config["history"]["Lx"]
        for L_y in config["history"]["Ly"]
    ]


def _task_path(output: Path, horizon: int, L_x: int, L_y: int) -> Path:
    return (
        output
        / "R3A"
        / "candidates"
        / f"h_{horizon:03d}"
        / f"Lx_{L_x:03d}_Ly_{L_y:03d}"
        / "task_result.json"
    )


def _aggregate_history(output: Path, config: dict[str, object]) -> dict[str, object]:
    selections: dict[str, object] = {}
    flattened: list[dict[str, object]] = []
    one_se_payload: dict[str, object] = {}
    for horizon in config["horizons"]:
        rows: list[dict[str, object]] = []
        for L_x in config["history"]["Lx"]:
            for L_y in config["history"]["Ly"]:
                path = _task_path(output, int(horizon), int(L_x), int(L_y))
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["status"] != "COMPLETED":
                    raise RuntimeError(f"R3 history task failed: {path}")
                rows.append(row)
                flattened.append(
                    {
                        "horizon": int(horizon),
                        "L_x": int(L_x),
                        "L_y": int(L_y),
                        "validation_MSE_mean": float(row["validation_loss"]),
                        "validation_MSE_SE": float(row["validation_se"]),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                    }
                )
        minimum = min(
            rows,
            key=lambda row: (
                float(row["validation_loss"]),
                history_complexity_key(
                    int(row["history"]["L_x"]), int(row["history"]["L_y"])
                ),
            ),
        )
        threshold = float(minimum["validation_loss"]) + float(
            minimum["validation_se"]
        )
        eligible = [
            row for row in rows if float(row["validation_loss"]) <= threshold
        ]
        selected = min(
            eligible,
            key=lambda row: history_complexity_key(
                int(row["history"]["L_x"]), int(row["history"]["L_y"])
            ),
        )
        selections[str(horizon)] = {
            "minimum_mean_configuration": minimum["history"],
            "one_se_threshold": threshold,
            "selected_history": selected["history"],
            "selected_validation_MSE_mean": selected["validation_loss"],
            "selected_validation_MSE_SE": selected["validation_se"],
            "selected_penalty": selected["penalty"]["selection"]["selected"][
                "normalized_weights"
            ],
            "history_selected_at_grid_edge": (
                int(selected["history"]["L_x"])
                in {min(LX_GRID), max(LX_GRID)}
                or int(selected["history"]["L_y"])
                in {min(LY_GRID), max(LY_GRID)}
            ),
        }
        one_se_payload[str(horizon)] = {
            "threshold": threshold,
            "eligible_histories": [row["history"] for row in eligible],
        }
    result = {
        "schema": "CZ_R3A_NATIVE_HISTORY_SELECTION_ORSS_V1",
        "status": "COMPLETED",
        "selection_unit": "PER_DIRECT_HORIZON",
        "one_standard_error_rule": True,
        "complexity_key": ["Lx+Ly", "Lx*Ly", "max(Lx,Ly)", "Lx", "Ly"],
        "anchor": {
            "M_tau": int(config["history"]["anchor_Mtau"]),
            "M_x": int(config["history"]["anchor_Mx"]),
            "CONTINUATION_SCALE_COEFFICIENT": float(
                config["history"]["continuation_scale_coefficient"]
            ),
        },
        "selections": selections,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
        "next_stage": "R3B_RESOLUTION_AND_PENALTY",
    }
    write_json(output / "R3A" / "history_selection.json", result)
    write_json(
        output / "R3A" / "history_one_se_set.json", one_se_payload
    )
    try:
        import pandas as pd

        frame = pd.DataFrame(flattened)
        frame.to_parquet(output / "R3A" / "history_candidates.parquet", index=False)
    except Exception as error:
        write_json(
            output / "R3A" / "history_candidates.json",
            {"rows": flattened, "parquet_error": repr(error)},
        )
    return result


def _run_history(
    *,
    config: dict[str, object],
    config_path: Path,
    data,
    output: Path,
    device: torch.device,
    resume: bool,
) -> None:
    specifications = _history_specs(config)
    started = time.perf_counter()
    source_commit = _source_commit()
    config_hash = _hash_file(config_path)
    for index, (horizon, L_x, L_y) in enumerate(specifications, start=1):
        destination = _task_path(output, horizon, L_x, L_y)
        if resume and destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            identity = existing.get("checkpoint_identity", {})
            if (
                existing.get("status") == "COMPLETED"
                and identity.get("config_hash") == config_hash
                and identity.get("data_hash") == data.source_sha256
                and identity.get("source_commit") == source_commit
                and identity.get("solver_version") == "ORSS_V1"
            ):
                print(
                    f"R3A {index}/{len(specifications)} resumed "
                    f"h={horizon} Lx={L_x} Ly={L_y}",
                    flush=True,
                )
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = run_development_task(
            data.inputs,
            data.target,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=int(config["history"]["anchor_Mtau"]),
            M_x=int(config["history"]["anchor_Mx"]),
            c_rho=float(config["history"]["continuation_scale_coefficient"]),
            device=device,
            primary_dtype=torch.float32,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            positive_grid_points=int(config["penalty"]["positive_grid_points"]),
            maximum_edge_expansions=int(
                config["penalty"]["maximum_edge_expansions"]
            ),
            rb_tolerance=float(
                config["reduced_basis"]["residual_tolerance_development"]
            ),
            rb_maximum_dimension=int(
                config["reduced_basis"]["maximum_dimension"]
            ),
            krylov_tolerance=float(
                config["krylov"]["relative_tolerance_development"]
            ),
            krylov_maximum_iterations=int(
                config["krylov"]["maximum_iterations"]
            ),
        )
        payload["checkpoint_identity"] = {
            "dataset": "CZ_REAL_V1",
            "furnace": "A",
            "fold": "ALL_DEVELOPMENT",
            "horizon": horizon,
            "Lx": L_x,
            "Ly": L_y,
            "Mtau": int(config["history"]["anchor_Mtau"]),
            "Mx": int(config["history"]["anchor_Mx"]),
            "solver": "ORSS",
            "source_commit": source_commit,
            "config_hash": config_hash,
            "data_hash": data.source_sha256,
            "solver_version": "ORSS_V1",
        }
        write_json(destination, payload)
        marker = destination.parent / (
            "DONE" if payload["status"] == "COMPLETED" else "FAILED"
        )
        marker.write_text(payload["status"] + "\n", encoding="utf-8")
        print(
            f"R3A {index}/{len(specifications)} h={horizon} "
            f"Lx={L_x} Ly={L_y} status={payload['status']} "
            f"loss={float(payload['validation_loss']):.8g} "
            f"elapsed={float(payload['elapsed_seconds']):.1f}s",
            flush=True,
        )
        if payload["status"] != "COMPLETED":
            raise RuntimeError(f"R3A_TASK_FAILED:{destination}")
    selection = _aggregate_history(output, config)
    write_json(
        output / "R3A" / "history_runtime_profile.json",
        {
            "status": "COMPLETED",
            "one_active_gpu_task": True,
            "completed_tasks": len(specifications),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_json(
        output / "R3A" / "R3A_STATUS.json",
        {
            "stage": "R3A_NATIVE_HISTORY_SELECTION",
            "status": "COMPLETED",
            "H2_NATIVE_COMPLETE": True,
            "next_stage": selection["next_stage"],
        },
    )


def _resolution_task_path(
    output: Path, horizon: int, M_tau: int, M_x: int
) -> Path:
    return (
        output
        / "R3B"
        / "candidates"
        / f"h_{horizon:03d}"
        / f"Mtau_{M_tau:03d}_Mx_{M_x:03d}"
        / "task_result.json"
    )


def _run_resolution(
    *,
    config: dict[str, object],
    config_path: Path,
    data,
    output: Path,
    device: torch.device,
    resume: bool,
) -> None:
    history_path = output / "R3A" / "history_selection.json"
    if not history_path.exists():
        raise RuntimeError("H2_NATIVE_HISTORY_NOT_FROZEN")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    source_commit = _source_commit()
    config_hash = _hash_file(config_path)
    specifications = [
        (int(horizon), int(M_tau), int(M_x))
        for horizon in config["horizons"]
        for M_tau in config["resolution"]["Mtau"]
        for M_x in config["resolution"]["Mx"]
    ]
    started = time.perf_counter()
    for index, (horizon, M_tau, M_x) in enumerate(specifications, start=1):
        selected_history = history["selections"][str(horizon)][
            "selected_history"
        ]
        L_x = int(selected_history["L_x"])
        L_y = int(selected_history["L_y"])
        destination = _resolution_task_path(
            output, horizon, M_tau, M_x
        )
        if resume and destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            identity = existing.get("checkpoint_identity", {})
            if (
                existing.get("status") == "COMPLETED"
                and identity.get("source_commit") == source_commit
                and identity.get("config_hash") == config_hash
                and identity.get("data_hash") == data.source_sha256
            ):
                print(
                    f"R3B {index}/{len(specifications)} resumed "
                    f"h={horizon} Mtau={M_tau} Mx={M_x}",
                    flush=True,
                )
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = run_development_task(
            data.inputs,
            data.target,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=M_tau,
            M_x=M_x,
            c_rho=float(config["continuation_scale_coefficient"]),
            device=device,
            primary_dtype=torch.float32,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            positive_grid_points=int(config["penalty"]["positive_grid_points"]),
            maximum_edge_expansions=int(
                config["penalty"]["maximum_edge_expansions"]
            ),
            rb_tolerance=float(
                config["reduced_basis"]["residual_tolerance_development"]
            ),
            rb_maximum_dimension=int(
                config["reduced_basis"]["maximum_dimension"]
            ),
            krylov_tolerance=float(
                config["krylov"]["relative_tolerance_development"]
            ),
            krylov_maximum_iterations=int(
                config["krylov"]["maximum_iterations"]
            ),
        )
        payload["checkpoint_identity"] = {
            "dataset": "CZ_REAL_V1",
            "furnace": "A",
            "fold": "ALL_DEVELOPMENT",
            "horizon": horizon,
            "Lx": L_x,
            "Ly": L_y,
            "Mtau": M_tau,
            "Mx": M_x,
            "solver": "ORSS",
            "source_commit": source_commit,
            "config_hash": config_hash,
            "data_hash": data.source_sha256,
            "solver_version": "ORSS_V1",
        }
        write_json(destination, payload)
        (destination.parent / (
            "DONE" if payload["status"] == "COMPLETED" else "FAILED"
        )).write_text(payload["status"] + "\n", encoding="utf-8")
        print(
            f"R3B {index}/{len(specifications)} h={horizon} "
            f"Mtau={M_tau} Mx={M_x} status={payload['status']} "
            f"loss={float(payload['validation_loss']):.8g} "
            f"elapsed={float(payload['elapsed_seconds']):.1f}s",
            flush=True,
        )
        if payload["status"] != "COMPLETED":
            raise RuntimeError(f"R3B_TASK_FAILED:{destination}")

    selections: dict[str, object] = {}
    penalty_search: dict[str, object] = {}
    all_gates = []
    for horizon in config["horizons"]:
        rows = [
            json.loads(
                _resolution_task_path(
                    output, int(horizon), int(M_tau), int(M_x)
                ).read_text(encoding="utf-8")
            )
            for M_tau in config["resolution"]["Mtau"]
            for M_x in config["resolution"]["Mx"]
        ]
        minimum = min(
            rows,
            key=lambda row: (
                float(row["validation_loss"]),
                int(row["resolution"]["M_tau"])
                * int(row["resolution"]["M_x"]),
                int(row["resolution"]["M_tau"]),
                int(row["resolution"]["M_x"]),
            ),
        )
        threshold = float(minimum["validation_loss"]) + float(
            minimum["validation_se"]
        )
        eligible = [
            row for row in rows if float(row["validation_loss"]) <= threshold
        ]
        selected = min(
            eligible,
            key=lambda row: (
                int(row["resolution"]["M_tau"])
                * int(row["resolution"]["M_x"]),
                int(row["resolution"]["M_tau"]),
                int(row["resolution"]["M_x"]),
            ),
        )
        finest = max(
            rows,
            key=lambda row: (
                int(row["resolution"]["M_tau"])
                * int(row["resolution"]["M_x"]),
                int(row["resolution"]["M_tau"]),
                int(row["resolution"]["M_x"]),
            ),
        )
        relative_to_finest = abs(
            float(selected["validation_loss"])
            - float(finest["validation_loss"])
        ) / max(float(finest["validation_loss"]), 1.0e-15)
        higher = [
            row
            for row in rows
            if int(row["resolution"]["M_tau"])
            >= int(selected["resolution"]["M_tau"])
            and int(row["resolution"]["M_x"])
            >= int(selected["resolution"]["M_x"])
        ]
        lepski = all(
            abs(
                float(row["validation_loss"])
                - float(selected["validation_loss"])
            )
            <= float(config["resolution"]["lepski_multiplier"])
            * (
                float(row["validation_se"])
                + float(selected["validation_se"])
            )
            for row in higher
        )
        gates = {
            "REPRESENTATION_GATE_PASS": relative_to_finest
            <= float(
                config["resolution"][
                    "representation_gate_primary_relative_error"
                ]
            ),
            "LEPSKI_PASS": lepski,
            "PENALTY_INTERVAL_CERTIFIED": (
                selected["penalty"]["interval_status"]
                == "PENALTY_INTERVAL_CERTIFIED"
            ),
            "RB_RESIDUAL_CERTIFIED": bool(
                selected["RB_RESIDUAL_CERTIFIED"]
            ),
            "FINAL_KKT_PASS": bool(selected["FINAL_KKT_PASS"]),
        }
        all_gates.append(all(gates.values()))
        selections[str(horizon)] = {
            "selected_history": selected["history"],
            "minimum_mean_resolution": minimum["resolution"],
            "selected_resolution": selected["resolution"],
            "selected_penalty": selected["penalty"]["selection"]["selected"][
                "normalized_weights"
            ],
            "one_se_threshold": threshold,
            "eligible_resolutions": [
                row["resolution"] for row in eligible
            ],
            "relative_validation_loss_to_finest": relative_to_finest,
            "gates": gates,
        }
        penalty_search[str(horizon)] = {
            "resolution": selected["resolution"],
            "selection": selected["penalty"]["selection"],
            "interval_history": selected["penalty"]["interval_history"],
        }
    status = "COMPLETED" if all(all_gates) else "FAILED"
    result = {
        "schema": "CZ_R3B_RESOLUTION_SELECTION_ORSS_V1",
        "status": status,
        "selections": selections,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }
    write_json(output / "R3B" / "resolution_selection.json", result)
    write_json(output / "R3B" / "penalty_search.json", penalty_search)
    write_json(
        output / "R3B" / "R3B_STATUS.json",
        {
            "stage": "R3B_RESOLUTION_AND_PENALTY",
            "status": status,
            "R3_RESOLUTION_COMPLETE": status == "COMPLETED",
            "elapsed_seconds": time.perf_counter() - started,
            "next_stage": (
                "R3C_CONTINUATION_SELECTION"
                if status == "COMPLETED"
                else "HARD_STOP"
            ),
        },
    )
    if status != "COMPLETED":
        raise RuntimeError("R3B_GATE_FAILED")


def _run_continuation(
    *,
    config: dict[str, object],
    config_path: Path,
    data,
    output: Path,
    device: torch.device,
    resume: bool,
) -> None:
    resolution_path = output / "R3B" / "resolution_selection.json"
    if not resolution_path.exists():
        raise RuntimeError("R3_RESOLUTION_NOT_FROZEN")
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    if resolution.get("status") != "COMPLETED":
        raise RuntimeError("R3_RESOLUTION_NOT_COMPLETE")
    source_commit = _source_commit()
    config_hash = _hash_file(config_path)
    root = output / "R3C"
    started = time.perf_counter()
    selections: dict[str, object] = {}
    first_exit: dict[str, object] = {}
    usage_rows: list[dict[str, object]] = []
    free_run_rows: dict[str, object] = {}
    for horizon in config["horizons"]:
        frozen = resolution["selections"][str(horizon)]
        L_x = int(frozen["selected_history"]["L_x"])
        L_y = int(frozen["selected_history"]["L_y"])
        M_tau = int(frozen["selected_resolution"]["M_tau"])
        M_x = int(frozen["selected_resolution"]["M_x"])
        weights = PenaltyWeights(**frozen["selected_penalty"])
        candidates: list[dict[str, object]] = []
        for c_rho in config["continuation_candidates"]:
            destination = (
                root
                / "candidates"
                / f"h_{int(horizon):03d}"
                / f"c_rho_{float(c_rho):g}.json"
            )
            payload: dict[str, object]
            if resume and destination.exists():
                existing = json.loads(destination.read_text(encoding="utf-8"))
                identity = existing.get("checkpoint_identity", {})
                if (
                    existing.get("status") == "COMPLETED"
                    and identity.get("source_commit") == source_commit
                    and identity.get("config_hash") == config_hash
                    and identity.get("data_hash") == data.source_sha256
                ):
                    payload = existing
                else:
                    payload = {}
            else:
                payload = {}
            if not payload:
                payload = evaluate_frozen_configuration(
                    data.inputs,
                    data.target,
                    horizon=int(horizon),
                    L_x=L_x,
                    L_y=L_y,
                    M_tau=M_tau,
                    M_x=M_x,
                    normalized_weights=weights,
                    c_rho=float(c_rho),
                    device=device,
                    chunk_time=int(config["cuda"]["operator_chunk_time"]),
                    maximum_iterations=int(
                        config["krylov"]["maximum_iterations"]
                    ),
                )
                payload["CONTINUATION_SCALE_COEFFICIENT"] = float(c_rho)
                payload["checkpoint_identity"] = {
                    "dataset": "CZ_REAL_V1",
                    "furnace": "A",
                    "horizon": int(horizon),
                    "Lx": L_x,
                    "Ly": L_y,
                    "Mtau": M_tau,
                    "Mx": M_x,
                    "lambda_tau": weights.lag,
                    "lambda_x": weights.amplitude,
                    "lambda_0": weights.ridge,
                    "continuation_c_rho": float(c_rho),
                    "solver": "ORSS",
                    "source_commit": source_commit,
                    "config_hash": config_hash,
                    "data_hash": data.source_sha256,
                    "solver_version": "ORSS_V1",
                }
                write_json(destination, payload)
            candidates.append(payload)
            print(
                f"R3C h={horizon} c_rho={float(c_rho):g} "
                f"status={payload['status']} "
                f"free_MSE={float(payload['free_run_MSE_mean']):.8g}",
                flush=True,
            )
            if payload["status"] != "COMPLETED":
                raise RuntimeError(f"R3C_TASK_FAILED:{destination}")
        selected = min(
            candidates,
            key=lambda row: (
                not bool(row["finite_complete_free_run"]),
                float(row["free_run_MSE_mean"]),
                float(row["maximum_normalized_extrapolation_distance"]),
                float(row["continuation_usage_fraction"]),
                float(row["CONTINUATION_SCALE_COEFFICIENT"]),
            ),
        )
        selections[str(horizon)] = {
            "history": {"L_x": L_x, "L_y": L_y},
            "resolution": {"M_tau": M_tau, "M_x": M_x},
            "penalty": frozen["selected_penalty"],
            "selected_CONTINUATION_SCALE_COEFFICIENT": selected[
                "CONTINUATION_SCALE_COEFFICIENT"
            ],
            "selection_order": config["selection_order"],
            "free_run_MSE_mean": selected["free_run_MSE_mean"],
            "teacher_forced_MSE_mean": selected["teacher_forced_MSE_mean"],
        }
        first_exit[str(horizon)] = {
            str(row["CONTINUATION_SCALE_COEFFICIENT"]): [
                {
                    "fold": fold["fold"],
                    "first_exit_target_index": fold["first_exit_target_index"],
                }
                for fold in row["folds"]
            ]
            for row in candidates
        }
        free_run_rows[str(horizon)] = candidates
        for row in candidates:
            usage_rows.append(
                {
                    "horizon": int(horizon),
                    "c_rho": float(
                        row["CONTINUATION_SCALE_COEFFICIENT"]
                    ),
                    "usage_fraction": float(
                        row["continuation_usage_fraction"]
                    ),
                    "maximum_normalized_distance": float(
                        row["maximum_normalized_extrapolation_distance"]
                    ),
                }
            )
    result = {
        "schema": "CZ_R3C_CONTINUATION_SELECTION_ORSS_V1",
        "status": "COMPLETED",
        "selections": selections,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }
    write_json(root / "continuation_selection.json", result)
    write_json(root / "first_exit_audit.json", first_exit)
    write_json(root / "free_run_metrics.json", free_run_rows)
    try:
        import pandas as pd

        pd.DataFrame(usage_rows).to_csv(
            root / "continuation_usage.csv", index=False
        )
    except Exception:
        write_json(root / "continuation_usage.json", usage_rows)
    write_json(
        root / "R3C_STATUS.json",
        {
            "stage": "R3C_CONTINUATION_SELECTION",
            "status": "COMPLETED",
            "R3_CONTINUATION_COMPLETE": True,
            "elapsed_seconds": time.perf_counter() - started,
            "next_stage": "R3D_RANK_PROFILE",
        },
    )


def _run_rank(
    *,
    config: dict[str, object],
    config_path: Path,
    data,
    output: Path,
    device: torch.device,
    resume: bool,
) -> None:
    continuation_path = output / "R3C" / "continuation_selection.json"
    if not continuation_path.exists():
        raise RuntimeError("R3_CONTINUATION_NOT_FROZEN")
    continuation = json.loads(
        continuation_path.read_text(encoding="utf-8")
    )
    source_commit = _source_commit()
    config_hash = _hash_file(config_path)
    root = output / "R3D"
    profiles: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for horizon in config["horizons"]:
        frozen = continuation["selections"][str(horizon)]
        destination = root / f"h_{int(horizon):03d}" / "rank_result.json"
        payload: dict[str, object] = {}
        if resume and destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            identity = existing.get("checkpoint_identity", {})
            if (
                existing.get("status") == "COMPLETED"
                and identity.get("source_commit") == source_commit
                and identity.get("config_hash") == config_hash
                and identity.get("data_hash") == data.source_sha256
            ):
                payload = existing
        if not payload:
            payload = rank_profile_configuration(
                data.inputs,
                data.target,
                horizon=int(horizon),
                L_x=int(frozen["history"]["L_x"]),
                L_y=int(frozen["history"]["L_y"]),
                M_tau=int(frozen["resolution"]["M_tau"]),
                M_x=int(frozen["resolution"]["M_x"]),
                normalized_weights=PenaltyWeights(**frozen["penalty"]),
                c_rho=float(
                    frozen["selected_CONTINUATION_SCALE_COEFFICIENT"]
                ),
                rank_budgets=[
                    float(value) for value in config["rank_budgets"]
                ],
                device=device,
                chunk_time=int(config["cuda"]["operator_chunk_time"]),
                maximum_iterations=int(
                    config["krylov"]["maximum_iterations"]
                ),
            )
            payload["checkpoint_identity"] = {
                "dataset": "CZ_REAL_V1",
                "furnace": "A",
                "horizon": int(horizon),
                "Lx": int(frozen["history"]["L_x"]),
                "Ly": int(frozen["history"]["L_y"]),
                "Mtau": int(frozen["resolution"]["M_tau"]),
                "Mx": int(frozen["resolution"]["M_x"]),
                "continuation_c_rho": float(
                    frozen["selected_CONTINUATION_SCALE_COEFFICIENT"]
                ),
                "solver": "ORSS",
                "source_commit": source_commit,
                "config_hash": config_hash,
                "data_hash": data.source_sha256,
                "solver_version": "ORSS_V1",
            }
            write_json(destination, payload)
        if payload["status"] != "COMPLETED":
            raise RuntimeError(f"R3D_TASK_FAILED:{destination}")
        profiles[str(horizon)] = payload
        for fold in payload["folds"]:
            for row in fold["rank_curve"]:
                csv_rows.append(
                    {
                        "horizon": int(horizon),
                        "fold": int(fold["fold"]),
                        **row,
                    }
                )
        print(
            f"R3D h={horizon} predictive_rank="
            f"{payload['predictive_rank_by_budget']}",
            flush=True,
        )
    write_json(
        root / "rank_selection.json",
        {
            "schema": "CZ_R3D_RANK_PROFILE_ORSS_V1",
            "status": "COMPLETED",
            "profiles": {
                horizon: {
                    "predictive_rank_by_budget": row[
                        "predictive_rank_by_budget"
                    ],
                    "structural_rank_status": row[
                        "structural_rank_status"
                    ],
                }
                for horizon, row in profiles.items()
            },
            "furnace_A_confirmation_accessed": False,
            "furnace_B_access_count": 0,
        },
    )
    try:
        import pandas as pd

        pd.DataFrame(csv_rows).to_csv(root / "rank_profile.csv", index=False)
    except Exception:
        write_json(root / "rank_profile.json", csv_rows)
    write_json(
        root / "R3D_STATUS.json",
        {
            "stage": "R3D_RANK_PROFILE",
            "status": "COMPLETED",
            "R3_RANK_COMPLETE": True,
            "elapsed_seconds": time.perf_counter() - started,
            "next_stage": "R4_BASELINE_MATRIX",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("history", "resolution-penalty", "continuation", "rank"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=("orss",), default="orss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-task", action="store_true")
    parser.add_argument("--furnace-a-only", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--output", default="results/cz_real_data/complete_5090"
    )
    args = parser.parse_args()
    if not args.furnace_a_only:
        raise RuntimeError("FURNACE_B_ACCESSED_BEFORE_FREEZE")
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("REAL_CUDA_DISPATCH_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = bool(
        config["cuda"]["allow_tf32_operator"]
    )
    raw_dir = Path(args.raw_dir)
    data = load_furnace_a(
        _furnace_a(raw_dir, str(config["data"]["furnace_a_filename"]))
    )
    output = Path(args.output)
    if args.stage == "history":
        _run_history(
            config=config,
            config_path=config_path,
            data=data,
            output=output,
            device=device,
            resume=args.resume,
        )
    elif args.stage == "resolution-penalty":
        _run_resolution(
            config=config,
            config_path=config_path,
            data=data,
            output=output,
            device=device,
            resume=args.resume,
        )
    elif args.stage == "continuation":
        _run_continuation(
            config=config,
            config_path=config_path,
            data=data,
            output=output,
            device=device,
            resume=args.resume,
        )
    elif args.stage == "rank":
        _run_rank(
            config=config,
            config_path=config_path,
            data=data,
            output=output,
            device=device,
            resume=args.resume,
        )
    else:
        raise RuntimeError(f"STAGE_IMPLEMENTATION_PENDING:{args.stage}")


if __name__ == "__main__":
    main()
