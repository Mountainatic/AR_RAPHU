from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PROJECT_ROOT))
if str(_SCRIPT_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PROJECT_ROOT / "src"))

import torch

from prism_benchmark.cpu_data import HeadSpec, ViewSpec, main_views
from prism_benchmark.cz_baselines import run_cz_baseline_development
from prism_benchmark.cz_extension import (
    H_STEPS,
    INPUT_COLUMNS,
    MODEL_PERIOD_SEC,
    TASK_ID,
    W0_STEPS,
    W_STEPS,
    build_all,
)
from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.neural3 import (
    MODEL_FAMILIES,
    select_candidate,
    shared_data_fingerprint,
)
from prism_benchmark.six_dataset_extension import (
    build_extension_common_support,
    freeze_extension,
)
from prism_benchmark.six_dataset_materialization import (
    materialize_extension,
    preflight_extension_materialization,
)
from prism_benchmark.six_dataset_package import (
    package_extension,
    write_extension_documents,
)
from prism_benchmark.six_dataset_reporting import report_extension
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_joint_stability import run_joint_stability_view
from prism_benchmark.cz_k_support import run_cz_k_channel
from prism_benchmark.v211_support import SUPPORT_CONTRACT
from prism_benchmark.v211_w import run_w_view


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/PRISM_V211_CZ_NEURAL3_SIX_DATASET_20260817"
)
PUBLIC_SHARED = Path(
    "/root/autodl-tmp/PRISM_V211_NATIVE_PUBLIC_ALL_20260815/shared"
)
PROJECT = Path(__file__).resolve().parents[1]
RAW_CZ_ROOT = Path(
    "/root/autodl-tmp/PRISM_DATASETS_V1/raw_sources/cz_czochralski"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_resource_environment() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    for name in (
        "PRISM_V211_K_INNER_WORKERS",
        "PRISM_V211_C_INNER_WORKERS",
        "PRISM_V211_W_INNER_WORKERS",
        "PRISM_V211_A_INNER_WORKERS",
    ):
        os.environ.setdefault(name, "1")


def _cz_head(direction: str) -> HeadSpec:
    return HeadSpec(
        head_id=TASK_ID,
        task_id=TASK_ID,
        dataset="cz_czochralski",
        target="crystal_diameter",
        cadence_seconds=MODEL_PERIOD_SEC,
        h_steps=H_STEPS,
        w_steps=W_STEPS,
        w0_steps=W0_STEPS,
        primary=True,
    )


def _cz_view(
    run_root: Path,
    direction: str,
    information_set: str,
) -> tuple[Path, ViewSpec]:
    shared = run_root / "shared" / direction
    return (
        shared,
        ViewSpec(
            _cz_head(direction),
            information_set,
            "record_time",
            "primary",
        ),
    )


def build_cz(run_root: Path) -> dict[str, Any]:
    raw = next(RAW_CZ_ROOT.glob("*.xlsx"), None)
    if raw is None:
        raise RuntimeError("STOP_CZ_RAW_FILE_MISSING")
    return build_all(raw, PROJECT, run_root)


def run_prism_direction(run_root: Path, direction: str) -> dict[str, Any]:
    shared = run_root / "shared" / direction
    direction_root = run_root / "directions" / direction
    paths = PublicAllPaths(PROJECT, shared, direction_root)
    input_view = _cz_view(run_root, direction, "input_only")[1]
    dynamic_view = _cz_view(run_root, direction, "dynamic")[1]
    output = paths.output
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"direction": direction, "status": "PASS"}
    for channel in INPUT_COLUMNS:
        result = run_cz_k_channel(
            shared,
            PROJECT,
            output,
            input_view,
            channel,
            "public_all",
        )
        results[f"K:{channel}"] = result
        if result.get("status") != "PASS":
            results["status"] = "COMPLETED_WITH_RETAINED_FAILURES"
    for stage, function, view in (
        ("C", run_c_view, input_view),
        ("W", run_w_view, input_view),
        ("A", run_a_view, dynamic_view),
    ):
        result = function(shared, PROJECT, output, view, "public_all")
        results[stage] = result
        if result.get("status") != "PASS":
            results["status"] = "COMPLETED_WITH_RETAINED_FAILURES"
    joint = run_joint_stability_view(
        shared,
        PROJECT,
        output,
        None,
        dynamic_view,
        "public_all",
    )
    results["JOINT"] = joint
    if joint.get("status") != "PASS":
        results["status"] = "COMPLETED_WITH_RETAINED_FAILURES"
    write_json(run_root / "logs" / f"PRISM_{direction}.json", results)
    gc.collect()
    return {
        "direction": direction,
        "status": results["status"],
        "prism_root": str(output),
        "direction_root": str(direction_root),
        "test_accessed": False,
        "ood_accessed": False,
    }


def run_prism(run_root: Path) -> dict[str, Any]:
    return {
        "status": "PASS",
        "directions": [
            run_prism_direction(run_root, "Rod_1_to_Rod_2"),
            run_prism_direction(run_root, "Rod_2_to_Rod_1"),
        ],
        "test_accessed": False,
        "ood_accessed": False,
    }


def run_baselines(run_root: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for direction in ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1"):
        shared = run_root / "shared" / direction
        paths = PublicAllPaths(
            PROJECT,
            shared,
            run_root / "directions" / direction,
        )
        summary = run_cz_baseline_development(
            shared,
            PROJECT,
            paths.output,
            _cz_view(run_root, direction, "input_only")[1],
            _cz_view(run_root, direction, "dynamic")[1],
        )
        summaries.append({"direction": direction, **summary})
        write_json(
            run_root / "logs" / f"BASELINES_{direction}.json",
            summaries[-1],
        )
        gc.collect()
    return {
        "status": (
            "PASS"
            if all(item["status"] != "FAILED" for item in summaries)
            else "FAILED"
        ),
        "directions": summaries,
        "test_accessed": False,
        "ood_accessed": False,
    }


def _public_views() -> list[ViewSpec]:
    input_only = main_views(PUBLIC_SHARED, "input_only")
    dynamic = main_views(PUBLIC_SHARED, "dynamic")
    return input_only + dynamic


def _neural_job_views(
    run_root: Path,
    scope: str,
) -> list[tuple[Path, ViewSpec]]:
    jobs: list[tuple[Path, ViewSpec]] = []
    if scope in {"public5", "all"}:
        jobs.extend((PUBLIC_SHARED, view) for view in _public_views())
    if scope in {"cz", "all"}:
        for direction in ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1"):
            shared = run_root / "shared" / direction
            jobs.extend(
                (
                    shared,
                    ViewSpec(
                        _cz_head(direction),
                        information_set,
                        "record_time",
                        "primary",
                    ),
                )
                for information_set in ("input_only", "dynamic")
            )
    return jobs


def _neural_worker(
    task: tuple[Path, ViewSpec, str, Path],
) -> dict[str, Any]:
    shared, view, model_name, destination = task
    _set_resource_environment()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "event": "NEURAL3_DEVELOPMENT_START",
                "pid": os.getpid(),
                "model": model_name,
                "view": view.relative_root.as_posix(),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = select_candidate(
        shared=shared,
        view=view,
        model_name=model_name,
        output=destination,
        device=device,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _reusable_selection(
    shared: Path,
    destination: Path,
    model_name: str,
    view: ViewSpec,
) -> dict[str, Any] | None:
    path = destination / model_name / view.relative_root / "SELECTION.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("status") == "PASS"
        and value.get("model") == model_name
        and value.get("view") == view.relative_root.as_posix()
        and value.get("support_contract") == SUPPORT_CONTRACT
        and value.get("test_accessed") is False
        and value.get("ood_accessed") is False
        and value.get("data_support_fingerprint")
        == shared_data_fingerprint(shared, view)
    ):
        return value
    return None
def _run_neural_development_parallel(
    run_root: Path,
    scope: str,
    workers: int,
) -> dict[str, Any]:
    jobs = _neural_job_views(run_root, scope)
    results: list[dict[str, Any]] = []
    pending: list[tuple[Path, ViewSpec, str, Path]] = []
    reused = 0
    started = time.time()
    for index, (shared, view) in enumerate(jobs, start=1):
        for model_name in MODEL_FAMILIES:
            destination = run_root / "results" / "NEURAL3"
            if shared == PUBLIC_SHARED:
                destination = destination / "PUBLIC5"
            else:
                destination = destination / "CZ" / shared.name
            existing = _reusable_selection(
                shared, destination, model_name, view
            )
            if existing is not None:
                reused += 1
                results.append(existing)
                print(
                    json.dumps(
                        {
                            "event": "NEURAL3_DEVELOPMENT_REUSE",
                            "job": index,
                            "jobs": len(jobs),
                            "model": model_name,
                            "view": view.relative_root.as_posix(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                pending.append((shared, view, model_name, destination))
    requested_workers = max(1, int(workers))
    effective_workers = min(requested_workers, len(pending)) if pending else 0
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_neural_worker, task) for task in pending]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "event": "NEURAL3_DEVELOPMENT_DONE",
                            "model": result.get("model"),
                            "view": result.get("view"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(key=lambda value: (value.get("view", ""), value.get("model", "")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audit = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAILED",
        "scope": scope,
        "jobs": len(results),
        "scheduled_jobs": len(pending),
        "reused_jobs": reused,
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "elapsed_seconds": time.time() - started,
        "device": device,
        "test_metrics_used_for_selection": False,
        "historical_metrics_used_for_selection": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(run_root / "logs" / f"NEURAL3_DEVELOPMENT_{scope}.json", audit)
    return audit




def run_neural_development(
    run_root: Path,
    scope: str,
    workers: int,
) -> dict[str, Any]:
    _set_resource_environment()
    return _run_neural_development_parallel(run_root, scope, workers)


def environment_audit(run_root: Path) -> dict[str, Any]:
    _set_resource_environment()
    value: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_version": torch.version.cuda,
        "test_accessed": False,
        "ood_accessed": False,
    }
    try:
        value["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except OSError as error:
        value["nvidia_smi_error"] = str(error)
    write_json(run_root / "logs" / "NEURAL3_ENVIRONMENT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "build",
            "environment",
            "prism",
            "baselines",
            "neural-dev",
            "common-support",
            "freeze",
            "preflight",
            "final",
            "report",
            "documents",
            "package",
            "status",
        ),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--scope", choices=("public5", "cz", "all"), default="all"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PRISM_NEURAL_WORKERS", "8")),
    )
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--generating-commit")
    parser.add_argument("--reporting-commit")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    public_root = (
        None if args.public_root is None else args.public_root.resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)
    _set_resource_environment()
    if args.stage == "build":
        print(json.dumps(build_cz(run_root), ensure_ascii=False))
    elif args.stage == "environment":
        print(json.dumps(environment_audit(run_root), ensure_ascii=False))
    elif args.stage == "prism":
        print(json.dumps(run_prism(run_root), ensure_ascii=False))
    elif args.stage == "baselines":
        print(json.dumps(run_baselines(run_root), ensure_ascii=False))
    elif args.stage == "neural-dev":
        print(
            json.dumps(
                run_neural_development(run_root, args.scope, args.workers),
                ensure_ascii=False,
            )
        )
    elif args.stage == "common-support":
        print(
            json.dumps(
                build_extension_common_support(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                ),
                ensure_ascii=False,
            )
        )
    elif args.stage == "freeze":
        print(
            json.dumps(
                freeze_extension(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                    generating_commit=args.generating_commit,
                ),
                ensure_ascii=False,
            )
        )
    elif args.stage == "preflight":
        print(
            json.dumps(
                preflight_extension_materialization(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                ),
                ensure_ascii=False,
            )
        )
    elif args.stage == "final":
        print(
            json.dumps(
                materialize_extension(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                ),
                ensure_ascii=False,
            )
        )
    elif args.stage == "report":
        print(
            json.dumps(
                report_extension(run_root, public_root=public_root),
                ensure_ascii=False,
            )
        )
    elif args.stage == "documents":
        print(
            json.dumps(
                write_extension_documents(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                    generating_commit=args.generating_commit,
                    reporting_commit=args.reporting_commit,
                ),
                ensure_ascii=False,
            )
        )
    elif args.stage == "package":
        print(
            json.dumps(
                package_extension(
                    run_root,
                    PROJECT,
                    public_root=public_root,
                    reporting_commit=args.reporting_commit,
                ),
                ensure_ascii=False,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "logs": sorted(str(path) for path in (run_root / "logs").glob("*.json")),
                    "results": sorted(str(path) for path in (run_root / "results").glob("**/SELECTION.json")),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
