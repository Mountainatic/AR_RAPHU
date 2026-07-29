#!/usr/bin/env python3
"""Run the frozen CZ FAST GO/NO-GO identifiability audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ar_raphu.cz_fast_audit.conditional_gram import conditional_gram_audit
from ar_raphu.cz_fast_audit.decision import RuntimeGate, decide_go_nogo
from ar_raphu.cz_fast_audit.fast_coarse_xar import coarse_xar_audit
from ar_raphu.cz_fast_audit.fast_linear import linear_increment_audit
from ar_raphu.cz_fast_audit.lag_correlation import lag_correlation_audit
from ar_raphu.cz_fast_audit.qk_stability import qk_stability_audit
from ar_raphu.cz_fast_audit.report import build_markdown_report
from ar_raphu.cz_fast_audit.residualization import (
    FAST_TASKS,
    conditional_energy_audit,
)
from ar_raphu.cz_real.protocol import (
    FURNACE_A_SHA256,
    PRIMARY_INPUTS,
    file_sha256,
    load_furnace_a,
)

DEFAULT_OUTPUT = ROOT / "results" / "cz_real_data" / "fast_go_nogo_v1"
DEFAULT_RAW = Path("/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fallback_fields: tuple[str, ...] = ("status",),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = list(fallback_fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def _validate_config(config: dict[str, object]) -> None:
    expected_tasks = [
        (
            task.name,
            task.L_x,
            task.L_y,
            task.horizon,
            task.maximum_correlation_lag,
        )
        for task in FAST_TASKS
    ]
    actual_tasks = [
        (
            str(item["name"]),
            int(item["Lx"]),
            int(item["Ly"]),
            int(item["h"]),
            int(item["max_correlation_lag"]),
        )
        for item in config["history_tasks"]
    ]
    if actual_tasks != expected_tasks:
        raise ValueError("FAST_HISTORY_TASKS_DO_NOT_MATCH_FROZEN_PLAN")
    if config["folds"] != [
        {"train": [0.0, 0.5], "validation": [0.5, 0.6]},
        {"train": [0.0, 0.7], "validation": [0.7, 0.8]},
    ]:
        raise ValueError("FAST_FOLDS_DO_NOT_MATCH_FROZEN_PLAN")
    if not bool(config["data"]["furnace_a_only"]):
        raise ValueError("FURNACE_A_ONLY_REQUIRED")
    if int(config["data"]["furnace_b_access_count_required"]) != 0:
        raise ValueError("FURNACE_B_ACCESS_REQUIREMENT_MUST_BE_ZERO")


def _resolve_furnace_a(raw_dir: Path) -> Path:
    # Exact allowlist only. Never glob or hash the Furnace-B source.
    candidates = (
        raw_dir / "实验数据1(3).xlsx",
        raw_dir / "实验数据1.xlsx",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            "FURNACE_A_EXACT_ALLOWLIST_RESOLUTION_FAILED:"
            + ",".join(str(path) for path in existing)
        )
    if file_sha256(existing[0]) != FURNACE_A_SHA256:
        raise ValueError("FURNACE_A_SHA256_MISMATCH")
    return existing[0]


def _stage(
    name: str,
    *,
    profile: dict[str, object],
    runtime_gate: RuntimeGate,
    maximum_minutes: float,
    operation: Callable[[], Any],
) -> Any:
    runtime_gate.check(name)
    started = time.perf_counter()
    print(f"[{name}] START", flush=True)
    result = operation()
    elapsed = time.perf_counter() - started
    profile["stages"][name] = {
        "status": "COMPLETED",
        "elapsed_seconds": elapsed,
        "maximum_seconds": float(maximum_minutes) * 60.0,
    }
    if elapsed > float(maximum_minutes) * 60.0:
        raise TimeoutError(
            f"FAST_AUDIT_RUNTIME_GATE_FAILED:{name}:{elapsed:.3f}s"
        )
    runtime_gate.check(name)
    print(f"[{name}] COMPLETED {elapsed:.3f}s", flush=True)
    return result


def _base_status(
    *,
    decision: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    return {
        "status": decision["status"],
        "furnace_b_access_count": 0,
        "runtime_seconds": elapsed,
        "positive_increment_horizons": decision[
            "positive_increment_horizons"
        ],
        "conditional_energy_summary": decision[
            "conditional_energy_summary"
        ],
        "conditional_gram_summary": decision[
            "conditional_gram_summary"
        ],
        "q_stability_summary": decision["q_stability_summary"],
        "k_low_order_summary": decision["k_low_order_summary"],
        "increment_summary": decision.get("increment_summary", {}),
        "next_allowed_stage": decision["next_allowed_stage"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "cz_real_data"
        / "cz_fast_go_nogo_v1.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--furnace-a-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    if not args.furnace_a_only:
        raise SystemExit("--furnace-a-only is mandatory")
    if not args.device.startswith("cuda"):
        raise SystemExit("The frozen FAST audit requires --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_UNAVAILABLE")

    device = "cuda:0" if args.device == "cuda" else args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_status_path = output / "CZ_FAST_GO_NOGO_STATUS.json"
    if args.resume and final_status_path.exists():
        existing = json.loads(final_status_path.read_text(encoding="utf-8"))
        if existing.get("status") in {
            "GO_FULL_CZ_IDENTIFICATION",
            "GO_PREDICTION_ONLY",
            "GO_PARTIAL_K",
            "NO_GO_FULL_KERNEL",
        }:
            print(
                f"RESUME_FINAL_STATUS={existing['status']}", flush=True
            )
            return 0

    for name in ("FAST_A", "FAST_B", "FAST_C", "FAST_D", "FAST_E", "FAST_F"):
        (output / name).mkdir(exist_ok=True)
    _write_json(
        output / "furnace_b_access_audit.json",
        {
            "furnace_b_access_count": 0,
            "required": 0,
            "status": "FURNACE_B_LOCKED_AND_NOT_ACCESSED",
        },
    )

    started_wall = time.time()
    runtime = config["runtime"]
    runtime_gate = RuntimeGate.start(float(runtime["max_total_minutes"]))
    profile: dict[str, object] = {
        "schema": config["schema"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device)),
        "solver": runtime["solver"],
        "primary_dtype": runtime["primary_dtype"],
        "allow_tf32": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "environment_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "stages": {},
    }
    stage_caps = runtime["stage_max_minutes"]
    energy_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    gram_rows: list[dict[str, object]] = []
    gram_summary: dict[str, object] = {}
    linear_rows: list[dict[str, object]] = []
    coarse_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    q_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []
    source_hash = "UNAVAILABLE"
    error: str | None = None

    try:
        furnace_a_path = _resolve_furnace_a(args.raw_dir)
        furnace = load_furnace_a(furnace_a_path)
        source_hash = furnace.source_sha256
        development_stop = int(
            np.floor(
                len(furnace.target)
                * float(config["data"]["development_fraction"])
            )
        )
        x = np.ascontiguousarray(
            furnace.inputs[:development_stop], dtype=np.float64
        )
        y = np.ascontiguousarray(
            furnace.target[:development_stop], dtype=np.float64
        )
        input_names = tuple(PRIMARY_INPUTS)
        profile["data"] = {
            "source_file": furnace_a_path.name,
            "source_sha256": source_hash,
            "source_sheet": furnace.source_sheet,
            "total_samples": len(furnace.target),
            "development_samples_used": development_stop,
            "furnace_b_access_count": 0,
        }

        energy_rows, energy_selection = _stage(
            "FAST_A",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_A"],
            operation=lambda: conditional_energy_audit(
                x,
                y,
                input_names=input_names,
                ridge_grid=config["conditional_energy"]["ridge_grid"],
                bootstrap_replicates=int(
                    config["bootstrap"]["development_replicates"]
                ),
                random_seed=int(config["bootstrap"]["random_seed"]),
            ),
        )
        _write_csv(
            output / "FAST_A" / "FAST_A_CONDITIONAL_ENERGY.csv",
            energy_rows,
        )
        _write_json(
            output / "FAST_A" / "FAST_A_SELECTION.json",
            energy_selection,
        )

        lag_rows, lag_summary = _stage(
            "FAST_B",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_B"],
            operation=lambda: lag_correlation_audit(
                x,
                y,
                input_names=input_names,
                ridge_grid=config["conditional_energy"]["ridge_grid"],
                bootstrap_replicates=int(
                    config["bootstrap"]["development_replicates"]
                ),
                random_seed=int(config["bootstrap"]["random_seed"]) + 1,
            ),
        )
        _write_csv(
            output
            / "FAST_B"
            / "FAST_B_AR_RESIDUAL_LAG_CORRELATION.csv",
            lag_rows,
        )
        _write_json(
            output / "FAST_B" / "FAST_B_SELECTION.json", lag_summary
        )

        gram_rows, gram_summary = _stage(
            "FAST_C",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_C"],
            operation=lambda: conditional_gram_audit(
                x,
                y,
                input_names=input_names,
                energy_selection=energy_selection,
                lag_basis_count=int(config["coarse_xar"]["Mtau"]),
                amplitude_basis_count=int(config["coarse_xar"]["Mx"]),
                top_eigenvalues=int(
                    config["conditional_gram"]["top_eigenvalues"]
                ),
                coercivity_ratios=config["conditional_gram"][
                    "coercivity_ratios"
                ],
            ),
        )
        _write_csv(
            output
            / "FAST_C"
            / "FAST_C_CONDITIONAL_GRAM_SPECTRUM.csv",
            gram_rows,
        )
        _write_json(
            output / "FAST_C" / "FAST_C_SCHUR_SUMMARY.json",
            gram_summary,
        )

        linear_rows, linear_selection = _stage(
            "FAST_D",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_D"],
            operation=lambda: linear_increment_audit(
                x,
                y,
                ridge_grid=config["linear"]["ridge_grid"],
                device=device,
            ),
        )
        _write_csv(
            output / "FAST_D" / "FAST_D_LINEAR_INCREMENT.csv",
            linear_rows,
        )
        _write_json(
            output / "FAST_D" / "FAST_D_SELECTION.json",
            linear_selection,
        )

        (
            coarse_rows,
            contribution_rows,
            coarse_profile,
            models,
        ) = _stage(
            "FAST_E",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_E"],
            operation=lambda: coarse_xar_audit(
                x,
                y,
                penalty_path=config["coarse_xar"]["penalty_path"],
                lag_basis_count=int(config["coarse_xar"]["Mtau"]),
                amplitude_basis_count=int(config["coarse_xar"]["Mx"]),
                continuation_scale=float(
                    config["coarse_xar"]["continuation_c_rho"]
                ),
                linear_rows=linear_rows,
                device=device,
            ),
        )
        _write_csv(
            output / "FAST_E" / "FAST_E_COARSE_XAR_RESULTS.csv",
            coarse_rows,
        )
        _write_csv(
            output
            / "FAST_E"
            / "FAST_E_CONTRIBUTION_DECOMPOSITION.csv",
            contribution_rows,
        )
        _write_json(
            output / "FAST_E" / "FAST_E_RUNTIME_PROFILE.json",
            coarse_profile,
        )

        q_rows, k_rows, fast_f_summary = _stage(
            "FAST_F_G",
            profile=profile,
            runtime_gate=runtime_gate,
            maximum_minutes=stage_caps["FAST_F_G"],
            operation=lambda: qk_stability_audit(
                x,
                y,
                input_names=input_names,
                coarse_rows=coarse_rows,
                models=models,
                lag_basis_count=int(config["coarse_xar"]["Mtau"]),
                amplitude_basis_count=int(config["coarse_xar"]["Mx"]),
                continuation_scale=float(
                    config["coarse_xar"]["continuation_c_rho"]
                ),
                q_high=float(config["gates"]["q_stability_high"]),
                q_moderate=float(
                    config["gates"]["q_stability_moderate"]
                ),
                k_mode_correlation_threshold=float(
                    config["gates"]["k_mode_correlation_threshold"]
                ),
            ),
        )
        _write_csv(
            output / "FAST_F" / "FAST_F_Q_STABILITY.csv", q_rows
        )
        _write_csv(
            output / "FAST_F" / "FAST_F_K_LOW_ORDER_STABILITY.csv",
            k_rows,
        )
        _write_json(
            output / "FAST_F" / "FAST_F_SUMMARY.json", fast_f_summary
        )
        decision = decide_go_nogo(
            conditional_energy_rows=energy_rows,
            conditional_gram_summary=gram_summary,
            coarse_rows=coarse_rows,
            q_rows=q_rows,
            k_rows=k_rows,
            gates=config["gates"],
            complete=True,
        )
    except Exception as exc:  # final status is mandatory on every failure path
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        profile["failure"] = {
            "error": error,
            "traceback": traceback.format_exc(),
        }
        decision = decide_go_nogo(
            conditional_energy_rows=energy_rows,
            conditional_gram_summary=gram_summary,
            coarse_rows=coarse_rows,
            q_rows=q_rows,
            k_rows=k_rows,
            gates=config["gates"],
            complete=False,
        )

    elapsed = time.time() - started_wall
    profile["completed_utc"] = datetime.now(timezone.utc).isoformat()
    profile["elapsed_seconds"] = elapsed
    profile["status"] = decision["status"]
    if error is not None:
        profile["status_detail"] = "FAST_AUDIT_RUNTIME_GATE_FAILED" if (
            "FAST_AUDIT_RUNTIME_GATE_FAILED" in error
        ) else "FAST_AUDIT_EXECUTION_FAILED"
    status = _base_status(decision=decision, elapsed=elapsed)
    if error is not None:
        status["failure"] = error

    # Materialize every mandatory path even if a runtime gate interrupts a stage.
    required_csvs = (
        (
            output / "FAST_A" / "FAST_A_CONDITIONAL_ENERGY.csv",
            energy_rows,
        ),
        (
            output
            / "FAST_B"
            / "FAST_B_AR_RESIDUAL_LAG_CORRELATION.csv",
            lag_rows,
        ),
        (
            output
            / "FAST_C"
            / "FAST_C_CONDITIONAL_GRAM_SPECTRUM.csv",
            gram_rows,
        ),
        (
            output / "FAST_D" / "FAST_D_LINEAR_INCREMENT.csv",
            linear_rows,
        ),
        (
            output / "FAST_E" / "FAST_E_COARSE_XAR_RESULTS.csv",
            coarse_rows,
        ),
        (
            output
            / "FAST_E"
            / "FAST_E_CONTRIBUTION_DECOMPOSITION.csv",
            contribution_rows,
        ),
        (
            output / "FAST_F" / "FAST_F_Q_STABILITY.csv",
            q_rows,
        ),
        (
            output / "FAST_F" / "FAST_F_K_LOW_ORDER_STABILITY.csv",
            k_rows,
        ),
    )
    for path, rows in required_csvs:
        if not path.exists():
            _write_csv(path, rows)
    schur_path = output / "FAST_C" / "FAST_C_SCHUR_SUMMARY.json"
    if not schur_path.exists():
        _write_json(schur_path, gram_summary)

    _write_json(output / "runtime_profile.json", profile)
    _write_json(final_status_path, status)
    _write_text(
        output / "CZ_FAST_GO_NOGO_REPORT.md",
        build_markdown_report(
            status=status,
            conditional_energy_rows=energy_rows,
            lag_rows=lag_rows,
            gram_summary=gram_summary,
            linear_rows=linear_rows,
            coarse_rows=coarse_rows,
            q_rows=q_rows,
            k_rows=k_rows,
            runtime_profile=profile,
            source_sha256=source_hash,
        ),
    )
    print(f"FINAL_STATUS={status['status']}", flush=True)
    print(f"OUTPUT_DIR={output}", flush=True)
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
