from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .cpu_data import ViewSpec, sha256_file
from .stage0 import write_json
from .v211_public_all_baselines import (
    COMPLETED_STATUSES,
    DYNAMIC_MODELS,
    NATIVE_INPUT_MODELS,
    SUCCESSFUL_COMPLETION_STATUSES,
    SIMPLE_MODELS,
    STATIC_INPUT_MODELS,
    run_ar_job,
    run_arx_job,
    run_dpls_job,
    run_hammerstein_job,
    run_n4sid_job,
    run_narx_job,
    run_simple_job,
    run_static_input_job,
)
from .v211_support import SUPPORT_CONTRACT


BaselineFunction = Callable[..., dict[str, Any]]


def _result_path(
    root: Path,
    family: str,
    model: str,
    view: ViewSpec,
) -> Path:
    return (
        root
        / family
        / "PREDICTIONS"
        / model
        / view.relative_root
        / "RESULT.json"
    )


def _reusable(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("status") in SUCCESSFUL_COMPLETION_STATUSES
        and value.get("support_contract") == SUPPORT_CONTRACT
        and value.get("test_accessed") is False
    ):
        return value
    return None


def _run_or_reuse(
    *,
    root: Path,
    family: str,
    model: str,
    view: ViewSpec,
    function: BaselineFunction,
    arguments: tuple[Any, ...],
) -> tuple[dict[str, Any], bool]:
    previous = _reusable(_result_path(root, family, model, view))
    if previous is not None:
        return previous, True
    return function(*arguments), False


def run_cz_baseline_development(
    shared: Path,
    project: Path,
    output: Path,
    input_view: ViewSpec,
    dynamic_view: ViewSpec,
) -> dict[str, Any]:
    root = output / "BASELINE_DEVELOPMENT"
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    reused = 0

    for model in SIMPLE_MODELS:
        for view in (input_view, dynamic_view):
            result, was_reused = _run_or_reuse(
                root=root,
                family="C2",
                model=model,
                view=view,
                function=run_simple_job,
                arguments=(shared, project, root, view, model),
            )
            results.append(result)
            reused += int(was_reused)

    for model in STATIC_INPUT_MODELS:
        result, was_reused = _run_or_reuse(
            root=root,
            family="C2",
            model=model,
            view=input_view,
            function=run_static_input_job,
            arguments=(shared, project, root, input_view, model),
        )
        results.append(result)
        reused += int(was_reused)

    result, was_reused = _run_or_reuse(
        root=root,
        family="C2",
        model="DPLS",
        view=input_view,
        function=run_dpls_job,
        arguments=(shared, project, root, input_view),
    )
    results.append(result)
    reused += int(was_reused)

    for model in ("PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"):
        result, was_reused = _run_or_reuse(
            root=root,
            family="C3",
            model=model,
            view=input_view,
            function=run_hammerstein_job,
            arguments=(shared, project, root, input_view, model),
        )
        results.append(result)
        reused += int(was_reused)

    dynamic_jobs: tuple[tuple[str, BaselineFunction], ...] = (
        ("AR", run_ar_job),
        ("ARX", run_arx_job),
        ("LINEAR_NARX", run_narx_job),
        ("N4SID", run_n4sid_job),
    )
    for model, function in dynamic_jobs:
        result, was_reused = _run_or_reuse(
            root=root,
            family="C3",
            model=model,
            view=dynamic_view,
            function=function,
            arguments=(shared, project, root, dynamic_view),
        )
        results.append(result)
        reused += int(was_reused)

    statuses = pd.Series(
        [str(item.get("status")) for item in results],
        dtype="string",
    ).value_counts()
    pd.DataFrame(
        [
            {
                key: value
                for key, value in item.items()
                if key not in {"selection", "traceback"}
            }
            for item in results
        ]
    ).sort_values(
        ["information_set", "model"],
        na_position="last",
    ).to_csv(root / "CZ_BASELINE_DEVELOPMENT.csv", index=False)
    fully_completed = all(
        item.get("status") in COMPLETED_STATUSES for item in results
    )
    all_successful = all(
        item.get("status")
        in {
            "PASS",
            "NOT_RUN_IMPLEMENTATION_ABSENT",
            "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        }
        for item in results
    )
    summary = {
        "status": (
            "PASS"
            if all_successful
            else "COMPLETED_WITH_RETAINED_FAILURES"
            if fully_completed
            else "FAILED"
        ),
        "stage": "D3_CZ_CPU_BASELINE_DEVELOPMENT",
        "support_contract": SUPPORT_CONTRACT,
        "dataset": input_view.head.dataset,
        "task_id": input_view.head.task_id,
        "jobs": len(results),
        "reused_jobs": reused,
        "status_counts": {
            str(key): int(value) for key, value in statuses.items()
        },
        "registered_models": sorted(
            {
                *SIMPLE_MODELS,
                *STATIC_INPUT_MODELS,
                *NATIVE_INPUT_MODELS,
                *DYNAMIC_MODELS,
            }
        ),
        "candidate_native_fitting": True,
        "common_validation_scoring": True,
        "cpu_model_freeze_sha256": sha256_file(
            project / "configs/cpu_model_freeze_v1.json"
        ),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(root / "SUMMARY.json", summary)
    return summary
