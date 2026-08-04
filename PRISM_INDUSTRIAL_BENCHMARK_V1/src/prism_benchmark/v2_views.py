from __future__ import annotations

from pathlib import Path

from .cpu_data import HeadSpec, ViewSpec, load_heads, primary_proxy_policy


def all_heads(shared: Path) -> dict[str, HeadSpec]:
    return {head.head_id: head for head in load_heads(shared, primary_only=False)}


def registered_views(shared: Path, information_set: str) -> list[ViewSpec]:
    heads = all_heads(shared)
    result: list[ViewSpec] = []
    root = shared / "sample_ids"
    for head_root in sorted(root.iterdir()):
        head = heads.get(head_root.name)
        if head is None:
            continue
        information_root = head_root / information_set
        if not information_root.is_dir():
            continue
        for availability_root in sorted(information_root.iterdir()):
            for proxy_root in sorted(availability_root.iterdir()):
                if (proxy_root / "train.parquet").is_file() and (proxy_root / "validation.parquet").is_file():
                    result.append(ViewSpec(head, information_set, availability_root.name, proxy_root.name))
    return result


def state_development_views(shared: Path) -> list[ViewSpec]:
    """State has no U/X; retain availability variants but deduplicate proxy variants."""
    views = registered_views(shared, "dynamic")
    chosen: dict[tuple[str, str], ViewSpec] = {}
    for view in views:
        key = (view.head.head_id, view.availability_scenario)
        primary = primary_proxy_policy(shared, view.head.task_id)
        if key not in chosen or view.proxy_policy == primary:
            chosen[key] = view
    return sorted(chosen.values(), key=lambda value: (value.head.head_id, value.availability_scenario))


def evaluation_level(view: ViewSpec, shared: Path) -> str | None:
    primary_proxy = primary_proxy_policy(shared, view.head.task_id)
    if view.head.primary and view.availability_scenario == "record_time" and view.proxy_policy == primary_proxy:
        return "LEVEL_B_PRIMARY_EXPLORATORY"
    if not view.head.primary and view.availability_scenario == "record_time" and view.proxy_policy == primary_proxy:
        return "LEVEL_C_CONFIRMATION"
    if view.head.primary and view.head.dataset == "tep" and view.availability_scenario == "delay_5_steps" and view.proxy_policy == primary_proxy:
        return "LEVEL_C_CONFIRMATION"
    if view.head.primary and view.head.dataset == "debutanizer" and view.availability_scenario == "delay_10_steps" and view.proxy_policy == primary_proxy:
        return "LEVEL_C_CONFIRMATION"
    if view.head.primary and view.head.task_id in {"PMSM_PM5", "METRO_P60"} and view.availability_scenario == "record_time" and view.proxy_policy == "full_sensor_secondary":
        return "LEVEL_C_CONFIRMATION"
    return None


def development_input_views(shared: Path) -> list[ViewSpec]:
    return [view for view in registered_views(shared, "input_only") if evaluation_level(view, shared) is not None]


def development_dynamic_views(shared: Path) -> list[ViewSpec]:
    return [view for view in registered_views(shared, "dynamic") if evaluation_level(view, shared) is not None]

