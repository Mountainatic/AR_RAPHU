from __future__ import annotations

from pathlib import Path

from .cpu_data import ViewSpec
from .v2_views import registered_views
from .v211_metro_config import ACTIVE_DATASET, ACTIVE_HEAD


def metro_p60_views(shared: Path, information_set: str) -> list[ViewSpec]:
    if information_set not in {"input_only", "dynamic"}:
        raise ValueError(f"unsupported information set: {information_set}")
    views = [
        view
        for view in registered_views(shared, information_set)
        if view.head.dataset == ACTIVE_DATASET and view.head.head_id == ACTIVE_HEAD
    ]
    if not views:
        raise RuntimeError(f"registered Metro-P60 view is missing: {information_set}")
    assert_only_metro_p60(views)
    return sorted(
        views,
        key=lambda view: (
            view.head.head_id,
            view.availability_scenario,
            view.proxy_policy,
        ),
    )


def metro_p60_input_views(shared: Path) -> list[ViewSpec]:
    return metro_p60_views(shared, "input_only")


def metro_p60_dynamic_views(shared: Path) -> list[ViewSpec]:
    return metro_p60_views(shared, "dynamic")


def assert_only_metro_p60(views: list[ViewSpec]) -> None:
    if not views:
        raise RuntimeError("no Metro-P60 views were selected")
    if any(view.head.dataset != ACTIVE_DATASET for view in views):
        raise RuntimeError("non-Metro dataset escaped the audit view filter")
    if any(view.head.head_id != ACTIVE_HEAD for view in views):
        raise RuntimeError("unregistered Metro head escaped the audit view filter")
