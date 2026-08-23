from __future__ import annotations

from pathlib import Path

from .cpu_data import ViewSpec, input_columns
from .v2_views import development_dynamic_views, development_input_views
from .v211_representative_stage1_config import PRIMARY_TASKS


def _sorted(views: list[ViewSpec]) -> list[ViewSpec]:
    return sorted(
        views,
        key=lambda view: (
            view.head.dataset,
            view.head.head_id,
            view.availability_scenario,
            view.proxy_policy,
        ),
    )


def _validate(views: list[ViewSpec], information_set: str) -> None:
    if not views:
        raise RuntimeError(f"no representative Stage-1 {information_set} views")
    if any(not view.head.primary for view in views):
        raise RuntimeError("non-primary head escaped representative Stage-1 scope")
    observed = {view.head.task_id for view in views}
    if observed != PRIMARY_TASKS:
        raise RuntimeError(
            "representative Stage-1 task mismatch: "
            f"missing={sorted(PRIMARY_TASKS - observed)} "
            f"extra={sorted(observed - PRIMARY_TASKS)}"
        )
    if any(view.information_set != information_set for view in views):
        raise RuntimeError("representative Stage-1 information-set mismatch")


def representative_stage1_input_views(shared: Path) -> list[ViewSpec]:
    views = _sorted(
        [
            view
            for view in development_input_views(shared)
            if view.head.task_id in PRIMARY_TASKS
        ]
    )
    _validate(views, "input_only")
    if len(views) != 3:
        raise RuntimeError(f"representative Stage-1 input view count: {len(views)} != 3")
    return views


def representative_stage1_dynamic_views(shared: Path) -> list[ViewSpec]:
    views = _sorted(
        [
            view
            for view in development_dynamic_views(shared)
            if view.head.task_id in PRIMARY_TASKS
        ]
    )
    _validate(views, "dynamic")
    if len(views) != 4:
        raise RuntimeError(f"representative Stage-1 dynamic view count: {len(views)} != 4")
    return views


def representative_stage1_k_jobs(shared: Path) -> list[tuple[ViewSpec, str]]:
    jobs = [
        (view, channel)
        for view in representative_stage1_input_views(shared)
        for channel in input_columns(
            shared, view.head.task_id, view.proxy_policy
        )
    ]
    if len(jobs) != 57:
        raise RuntimeError(
            f"representative Stage-1 K job count: {len(jobs)} != 57"
        )
    return jobs
