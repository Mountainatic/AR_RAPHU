from __future__ import annotations

from pathlib import Path

from .cpu_data import ViewSpec, input_columns
from .v2_views import development_dynamic_views, development_input_views
from .v211_public_all_config import PRIMARY_TASKS


EXPECTED_INPUT_VIEWS = 9
EXPECTED_DYNAMIC_VIEWS = 11
EXPECTED_K_CHANNEL_JOBS = 124


def _validate_primary_scope(views: list[ViewSpec], information_set: str) -> None:
    if not views:
        raise RuntimeError(f"no public-all {information_set} views were selected")
    if any(not view.head.primary for view in views):
        raise RuntimeError("non-primary head escaped public-all view selection")
    observed = {view.head.task_id for view in views}
    if observed != PRIMARY_TASKS:
        raise RuntimeError(
            f"public-all primary task mismatch: missing={sorted(PRIMARY_TASKS - observed)} "
            f"extra={sorted(observed - PRIMARY_TASKS)}"
        )
    if any(view.information_set != information_set for view in views):
        raise RuntimeError("public-all information-set mismatch")


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


def _primary_views(views: list[ViewSpec]) -> list[ViewSpec]:
    return [view for view in views if view.head.primary]


def public_all_input_views(shared: Path) -> list[ViewSpec]:
    views = _sorted(_primary_views(development_input_views(shared)))
    _validate_primary_scope(views, "input_only")
    if len(views) != EXPECTED_INPUT_VIEWS:
        raise RuntimeError(
            f"public-all input view count mismatch: {len(views)} != {EXPECTED_INPUT_VIEWS}"
        )
    return views


def public_all_dynamic_views(shared: Path) -> list[ViewSpec]:
    views = _sorted(_primary_views(development_dynamic_views(shared)))
    _validate_primary_scope(views, "dynamic")
    if len(views) != EXPECTED_DYNAMIC_VIEWS:
        raise RuntimeError(
            f"public-all dynamic view count mismatch: {len(views)} != {EXPECTED_DYNAMIC_VIEWS}"
        )
    return views


def public_all_k_jobs(shared: Path) -> list[tuple[ViewSpec, str]]:
    jobs = [
        (view, channel)
        for view in public_all_input_views(shared)
        for channel in input_columns(shared, view.head.task_id, view.proxy_policy)
    ]
    if len(jobs) != EXPECTED_K_CHANNEL_JOBS:
        raise RuntimeError(
            f"public-all K job count mismatch: {len(jobs)} != {EXPECTED_K_CHANNEL_JOBS}"
        )
    return jobs
