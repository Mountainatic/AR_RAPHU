from __future__ import annotations

from pathlib import Path

from .cpu_data import ViewSpec
from .v2_views import registered_views
from .v21_config import ACTIVE_DATASET, ACTIVE_HEADS


def sru_views(shared: Path, information_set: str) -> list[ViewSpec]:
    if information_set not in {"input_only", "dynamic"}:
        raise ValueError(f"unsupported information set: {information_set}")
    views = [
        view
        for view in registered_views(shared, information_set)
        if view.head.dataset == ACTIVE_DATASET and view.head.head_id in ACTIVE_HEADS
    ]
    found = {view.head.head_id for view in views}
    if found != ACTIVE_HEADS:
        missing = sorted(ACTIVE_HEADS - found)
        raise RuntimeError(f"registered SRU views are missing: {missing}")
    return sorted(
        views,
        key=lambda view: (
            view.head.head_id,
            view.availability_scenario,
            view.proxy_policy,
        ),
    )


def sru_input_views(shared: Path) -> list[ViewSpec]:
    return sru_views(shared, "input_only")


def sru_dynamic_views(shared: Path) -> list[ViewSpec]:
    return sru_views(shared, "dynamic")


def assert_only_sru(views: list[ViewSpec]) -> None:
    if not views:
        raise RuntimeError("no SRU views were selected")
    if any(view.head.dataset != ACTIVE_DATASET for view in views):
        raise RuntimeError("non-SRU view escaped v2.1 filtering")
    if any(view.head.head_id not in ACTIVE_HEADS for view in views):
        raise RuntimeError("unregistered SRU head escaped v2.1 filtering")
