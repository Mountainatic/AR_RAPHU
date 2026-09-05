from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "PRISM_V211_STAGEWISE_ABLATION_HYBRID_HW_20260904_R3"
MINIMUM_FREE_GIB = 15.0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def _require_space(path: Path) -> None:
    free = _free_gib(path)
    if free < MINIMUM_FREE_GIB:
        raise RuntimeError(f"STOP_DATA_DISK_FREE_GIB_{free:.3f}_BELOW_{MINIMUM_FREE_GIB}")


def _pass_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "PASS" and value.get("protocol_id") == PROTOCOL_ID


def _run(command: list[str], log_path: Path, env: dict[str, str], data_root: Path) -> None:
    _require_space(data_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc()}] RUN {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def _runner_command(
    unit: dict[str, Any],
    stage: str,
    run_root: Path,
    development_workers: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "run_stagewise_ablation_hybrid.py"),
        stage,
        "--project",
        str(PROJECT),
        "--shared",
        str(unit["shared"]),
        "--selection-run-root",
        str(unit["selection_run_root"]),
        "--head-id",
        str(unit["head_id"]),
        "--information-set",
        str(unit["information_set"]),
        "--availability-scenario",
        str(unit["availability_scenario"]),
        "--proxy-policy",
        str(unit["proxy_policy"]),
    ]
    if stage == "development":
        command.extend(
            ["--workers", str(development_workers), "--per-worker-gib", "4"]
        )
    if stage in {"checkpoint", "infer"}:
        command.extend(["--checkpoint-root", str(unit["checkpoint_root"])])
    if stage == "infer":
        command.extend(["--destination-run-root", str(unit["prediction_run_root"])])
    return command


def _units(args: argparse.Namespace, representative_shared: Path) -> list[dict[str, Any]]:
    run_root = args.run_root.resolve()
    public_root = args.public_root.resolve()
    tep_root = args.tep_root.resolve()
    cz_root = args.cz_root.resolve()
    definitions = [
        ("TEP H0", "tep", "TEP_G_NOWCAST_H0__H0__W1", tep_root / "tep_nowcast" / "shared", run_root / "selection" / "tep_h0", "proxy_excluded", None, True, True),
        ("TEP H0 input-only", "tep", "TEP_G_NOWCAST_H0__H0__W1", tep_root / "tep_nowcast" / "shared", run_root / "selection" / "tep_h0", "proxy_excluded", None, False, False),
        ("TEP H0 maturity-5", "tep", "TEP_G_NOWCAST_H0__H0__W1", tep_root / "tep_nowcast" / "shared", run_root / "selection" / "tep_h0", "proxy_excluded", None, False, False),
        ("Debutanizer", "debutanizer", "DEB_C4__H5__W1", public_root / "shared", public_root, "primary", None, False, True),
        ("SRU H2S", "sru", "SRU_H2S_REP_H1__H1__W1", representative_shared, run_root / "selection" / "sru_h2s", "primary", None, True, True),
        ("SRU SO2", "sru", "SRU_SO2_REP_H1__H1__W1", representative_shared, run_root / "selection" / "sru_so2", "primary", None, True, True),
        ("PMSM", "pmsm", "PMSM_PM5__H600__W60", public_root / "shared", public_root, "proxy_excluded", None, False, True),
        ("MetroPT P60", "metropt", "METRO_P60__H6__W1", public_root / "shared", public_root, "proxy_excluded", None, False, True),
        ("MetroPT Oil20", "metropt", "METRO_OIL20__H120__W12", public_root / "shared", public_root, "primary", None, False, True),
        ("CZ Rod1→Rod2", "cz_czochralski", "CZ_DIAM_RAW2S_CURRENT_L256_H4", cz_root / "h4" / "shared" / "Rod_1_to_Rod_2", cz_root / "h4" / "directions" / "Rod_1_to_Rod_2", "primary", "Rod_1_to_Rod_2", False, True),
        ("CZ Rod2→Rod1", "cz_czochralski", "CZ_DIAM_RAW2S_CURRENT_L256_H4", cz_root / "h4" / "shared" / "Rod_2_to_Rod_1", cz_root / "h4" / "directions" / "Rod_2_to_Rod_1", "primary", "Rod_2_to_Rod_1", False, True),
    ]
    units: list[dict[str, Any]] = []
    for index, (label, dataset, head, shared, selection, proxy, direction, develop, primary) in enumerate(definitions):
        slug = f"{index:02d}_{head}" + (f"_{direction}" if direction else "")
        private = dataset == "cz_czochralski"
        prediction_parent = run_root / ("private_cz_predictions" if private else "predictions")
        units.append(
            {
                "label": label,
                "dataset": dataset,
                "head_id": head,
                "shared": shared,
                "selection_run_root": selection,
                "information_set": "input_only" if label.endswith("input-only") else "dynamic",
                "availability_scenario": "analyzer_maturity_5_steps" if label.endswith("maturity-5") else "record_time",
                "proxy_policy": proxy,
                "direction": direction,
                "develop": develop,
                "primary": primary,
                "checkpoint_root": run_root / "checkpoints" / slug,
                "prediction_run_root": prediction_parent / slug,
                "privacy": "PRIVATE_AGGREGATE_ONLY" if private else "PUBLIC",
            }
        )
    return units


def _report_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in unit.items()
        if key not in {"shared", "checkpoint_root", "develop"}
    }


def _unit_groups(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Keep the three TEP views ordered while exposing independent units."""
    tep_labels = {"TEP H0", "TEP H0 input-only", "TEP H0 maturity-5"}
    tep_group = [unit for unit in units if str(unit["label"]) in tep_labels]
    if [str(unit["label"]) for unit in tep_group] != [
        "TEP H0",
        "TEP H0 input-only",
        "TEP H0 maturity-5",
    ]:
        raise RuntimeError("unexpected TEP unit order")
    return [tep_group] + [
        [unit] for unit in units if str(unit["label"]) not in tep_labels
    ]


def _run_unit(
    unit: dict[str, Any],
    *,
    run_root: Path,
    data_root: Path,
    env: dict[str, str],
    development_workers: int,
    status: dict[str, Any],
    status_path: Path,
    status_lock: threading.Lock,
    bootstrap_locks: dict[Path, threading.Lock],
) -> None:
    label = str(unit["label"])
    with status_lock:
        status["stages"][label] = "RUNNING"
        _write_json(status_path, status)
    try:
        if unit["develop"]:
            _run(
                _runner_command(
                    unit, "development", run_root, development_workers
                ),
                run_root / "logs" / f"{label}.development.log",
                env,
                data_root,
            )
        if unit["information_set"] == "dynamic":
            selection_root = Path(unit["selection_run_root"]).resolve()
            with bootstrap_locks[selection_root]:
                _run(
                    _runner_command(
                        unit, "freeze-bootstrap", run_root, development_workers
                    ),
                    run_root / "logs" / f"{label}.freeze-bootstrap.log",
                    env,
                    data_root,
                )
        checkpoint_marker = (
            unit["checkpoint_root"] / "STAGEWISE_CHECKPOINT_MANIFEST.json"
        )
        if not _pass_marker(checkpoint_marker):
            _run(
                _runner_command(
                    unit, "checkpoint", run_root, development_workers
                ),
                run_root / "logs" / f"{label}.checkpoint.log",
                env,
                data_root,
            )
        inference_env = env.copy()
        inference_env["PRISM_FORMAL_INFERENCE_ONLY"] = "1"
        inference_marker = (
            unit["prediction_run_root"] / "STAGEWISE_INFERENCE_COMPLETE.json"
        )
        if not _pass_marker(inference_marker):
            _run(
                _runner_command(unit, "infer", run_root, development_workers),
                run_root / "logs" / f"{label}.infer.log",
                inference_env,
                data_root,
            )
    except Exception:
        with status_lock:
            status["stages"][label] = "FAILED"
            _write_json(status_path, status)
        raise
    with status_lock:
        status["stages"][label] = "PASS"
        _write_json(status_path, status)


def _run_group(
    group: list[dict[str, Any]],
    **unit_kwargs: Any,
) -> None:
    for unit in group:
        _run_unit(unit, **unit_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe AutoDL hybrid-h/w stagewise run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--tep-root", type=Path, required=True)
    parser.add_argument("--cz-root", type=Path, required=True)
    parser.add_argument("--parallel-units", type=int, default=1)
    parser.add_argument("--development-workers", type=int, default=8)
    args = parser.parse_args()
    if args.parallel_units < 1:
        parser.error("--parallel-units must be at least 1")
    if args.development_workers < 1:
        parser.error("--development-workers must be at least 1")

    run_root = args.run_root.resolve()
    data_root = run_root.parent
    status_path = run_root / "logs" / "STAGEWISE_RUN_STATUS.json"
    status: dict[str, Any] = {
        "status": "WAITING_FOR_EXISTING_JOB" if args.wait_pid else "RUNNING",
        "protocol_id": PROTOCOL_ID,
        "started_utc": _utc(),
        "wait_pid": args.wait_pid,
        "minimum_free_gib": MINIMUM_FREE_GIB,
        "parallel_units": args.parallel_units,
        "development_workers_per_unit": args.development_workers,
        "stages": {},
        "test_accessed": False,
        "ood_accessed": False,
    }
    _write_json(status_path, status)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["AR_RAPHU_RUNTIME_MANAGER"] = "uv"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    try:
        while args.wait_pid and _pid_alive(args.wait_pid):
            status["last_wait_check_utc"] = _utc()
            _write_json(status_path, status)
            time.sleep(30)
        status["status"] = "RUNNING"
        status["existing_job_finished_utc"] = _utc()
        _write_json(status_path, status)
        _require_space(data_root)

        representative_shared = run_root / "reconstructed_representative_c1" / "shared"
        if not (representative_shared / "TASK_REGISTRY.json").is_file():
            if representative_shared.exists():
                raise RuntimeError("incomplete representative C1 exists; refusing overwrite")
            _run(
                [
                    sys.executable,
                    str(PROJECT / "scripts" / "build_shared_data.py"),
                    "--raw-root",
                    str(args.raw_root.resolve()),
                    "--registry-root",
                    str(args.registry_root.resolve()),
                    "--config",
                    str(PROJECT / "configs" / "representative_horizon_stage1_tep_sru_c1_tasks.json"),
                    "--output",
                    str(representative_shared),
                ],
                run_root / "logs" / "C1_RECONSTRUCTION.log",
                env,
                data_root,
            )
        _run(
            [
                sys.executable,
                str(PROJECT / "scripts" / "validate_shared_data.py"),
                "--shared",
                str(representative_shared),
                "--output",
                str(run_root / "logs" / "C1_RECONSTRUCTION_VALIDATION.json"),
            ],
            run_root / "logs" / "C1_RECONSTRUCTION_VALIDATION.log",
            env,
            data_root,
        )
        status["stages"]["C1_RECONSTRUCTION"] = "PASS"
        _write_json(status_path, status)

        units = _units(args, representative_shared)
        status_lock = threading.Lock()
        bootstrap_locks = {
            Path(unit["selection_run_root"]).resolve(): threading.Lock()
            for unit in units
        }
        groups = _unit_groups(units)
        with ThreadPoolExecutor(max_workers=args.parallel_units) as executor:
            futures = {
                executor.submit(
                    _run_group,
                    group,
                    run_root=run_root,
                    data_root=data_root,
                    env=env,
                    development_workers=args.development_workers,
                    status=status,
                    status_path=status_path,
                    status_lock=status_lock,
                    bootstrap_locks=bootstrap_locks,
                ): [str(unit["label"]) for unit in group]
                for group in groups
            }
            errors: list[str] = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
            if errors:
                raise RuntimeError("parallel unit failures: " + "; ".join(errors))

        manifest_path = run_root / "freeze" / "STAGEWISE_REPORT_INPUT_MANIFEST.json"
        report_manifest = {
            "status": "FROZEN_BEFORE_REPORTING",
            "protocol_id": PROTOCOL_ID,
            "bootstrap_replicates": 500,
            "orchestration": {
                "parallel_units": args.parallel_units,
                "development_workers_per_unit": args.development_workers,
                "threads_per_process": 1,
            },
            "units": [_report_unit(unit) for unit in units],
        }
        _write_json(manifest_path, report_manifest)
        _run(
            [
                sys.executable,
                str(PROJECT / "scripts" / "report_stagewise_ablation_hybrid.py"),
                "--manifest",
                str(manifest_path),
                "--output",
                str(run_root / "public_aggregate_report"),
            ],
            run_root / "logs" / "REPORT.log",
            env,
            data_root,
        )
        private_root = run_root / "private_cz_predictions"
        if private_root.exists():
            for path in private_root.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            private_root.chmod(0o700)
        status["status"] = "PASS"
        status["completed_utc"] = _utc()
        status["test_accessed"] = True
        status["ood_accessed"] = False
        status["public_aggregate_report"] = str(run_root / "public_aggregate_report")
        _write_json(status_path, status)
    except Exception as error:
        status["status"] = "FAILED"
        status["failed_utc"] = _utc()
        status["error"] = str(error)
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
