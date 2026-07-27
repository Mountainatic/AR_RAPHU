"""Machine-readable PB1 development preregistration checks."""

from __future__ import annotations

from typing import Any


def development_preregistration_gaps(config: dict[str, Any]) -> list[str]:
    """Return choices that materially affect PB1 results but remain unspecified."""

    gaps: list[str] = []
    dataset = config.get("dataset", {})
    task = config.get("task", {})
    selection = config.get("selection", {})
    models = set(config.get("models", []))

    if dataset.get("development_split", {}).get("status") != "FROZEN":
        gaps.append("dataset.development_split.status")
    if not task.get("xar_history_selection"):
        gaps.append("task.xar_history_selection")
    if "ar_ridge" in models or "arx_ridge" in models:
        if not config.get("baselines", {}).get("ridge_weight_grid"):
            gaps.append("baselines.ridge_weight_grid")
    if "narx_mlp" in models:
        narx = config.get("baselines", {}).get("narx_mlp", {})
        for field in (
            "hidden_width_grid",
            "depth_grid",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "seeds",
        ):
            if field not in narx:
                gaps.append(f"baselines.narx_mlp.{field}")
    spectral_models = {
        "rank1_ar_raphu",
        "fixed_rank2_ar_raphu",
        "full_spectral_ar_raphu",
        "adaptive_spectral_ar_raphu",
    }
    if models & spectral_models:
        regularization = selection.get("regularization_grid", {})
        for field in (
            "lag_smoothness",
            "amplitude_smoothness",
            "ridge_weight",
        ):
            if not regularization.get(field):
                gaps.append(f"selection.regularization_grid.{field}")
    if "adaptive_spectral_ar_raphu" in models:
        bootstrap = config.get("bootstrap", {})
        for field in ("block_length", "development_replicates", "seed"):
            if field not in bootstrap:
                gaps.append(f"bootstrap.{field}")
    if dataset.get("id") == "whpn":
        sensitivity = selection.get("loss_sensitivity", [])
        if "huber" in sensitivity and "huber_delta" not in selection:
            gaps.append("selection.huber_delta")
    return sorted(set(gaps))


def development_preflight_status(config: dict[str, Any]) -> str:
    return (
        "READY_FOR_DEVELOPMENT"
        if not development_preregistration_gaps(config)
        else "BLOCKED_BY_MISSING_PREREGISTRATION"
    )
