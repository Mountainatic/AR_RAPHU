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
    history = task.get("xar_history_selection", {})
    if not history:
        gaps.append("task.xar_history_selection")
    else:
        for lane in (
            "H1_baseline_faithful",
            "H2_native_model",
            "H3_shared_history_fairness",
        ):
            if lane not in history:
                gaps.append(f"task.xar_history_selection.{lane}")
    literature = config.get("literature_profiles", {})
    for field in ("path", "paper_profile", "companion_code_commit"):
        if not literature.get(field):
            gaps.append(f"literature_profiles.{field}")
    if literature.get("use_companion_internal_split") is not False:
        gaps.append("literature_profiles.use_companion_internal_split_false")
    baselines = config.get("baselines", {})
    if "arx_qr" in models:
        arx = baselines.get("arx_primary", {})
        if arx.get("solver") != "PIVOTED_QR_OR_SVD_MINIMUM_NORM":
            gaps.append("baselines.arx_primary.solver")
        if arx.get("scientific_ridge") != 0.0:
            gaps.append("baselines.arx_primary.scientific_ridge_zero")
    if "mlpnarx_champneys2024" in models:
        narx = baselines.get("mlpnarx_primary", {})
        required = {
            "profile": "MLPNARX_CHAMPNEYS2024",
            "hidden_layers": 1,
            "activation": "tanh",
            "widths": [2, 5, 7, 10],
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "iterations": 20000,
            "early_stopping": False,
            "initializations": 5,
            "history": "ARX_AIC_SELECTED",
            "scaling": "TRAIN_ONLY_MINMAX_MINUS1_PLUS1",
            "selection": "VALIDATION_AIC",
        }
        for field, expected in required.items():
            if narx.get(field) != expected:
                gaps.append(f"baselines.mlpnarx_primary.{field}")
    spectral_models = {
        "rank1_ar_raphu",
        "fixed_rank2_ar_raphu",
        "full_spectral_ar_raphu",
        "adaptive_spectral_ar_raphu",
    }
    if models & spectral_models:
        regularization = selection.get("spectral_penalty", {})
        required = {
            "normalization": (
                "POSITIVE_GENERALIZED_EIGENVALUE_MEDIAN_RELATIVE_TO_TRAIN_GRAM"
            ),
            "interval": "AUTOMATIC_SHRINKAGE_COVERAGE",
            "grid_points_per_axis": 7,
            "boundary_expansions_max": 2,
            "one_se_tie": "LOWEST_EFFECTIVE_DF",
        }
        for field, expected in required.items():
            if regularization.get(field) != expected:
                gaps.append(f"selection.spectral_penalty.{field}")
        for field in (
            "near_unpenalized_shrinkage",
            "near_zero_shrinkage",
            "risk",
        ):
            if field not in regularization:
                gaps.append(f"selection.spectral_penalty.{field}")
    if "adaptive_spectral_ar_raphu" in models:
        bootstrap = config.get("bootstrap", {})
        if bootstrap.get("development_replicates") != 250:
            gaps.append("bootstrap.development_replicates_250")
        if bootstrap.get("confirmation_replicates") != 1000:
            gaps.append("bootstrap.confirmation_replicates_1000")
        if bootstrap.get("shared_fixed_block_length") is not None:
            gaps.append("bootstrap.shared_fixed_block_length_must_be_null")
        for field in ("primary_unit", "within_record_block"):
            if not bootstrap.get(field):
                gaps.append(f"bootstrap.{field}")
    if dataset.get("id") == "whpn":
        if selection.get("primary_loss") != "MSE":
            gaps.append("selection.primary_loss_mse")
        sensitivity = selection.get("loss_sensitivity", {})
        if sensitivity.get("role") != "APPENDIX_ONLY":
            gaps.append("selection.loss_sensitivity.role_appendix_only")
        if sensitivity.get("participates_in_primary_selection") is not False:
            gaps.append(
                "selection.loss_sensitivity.not_in_primary_selection"
            )
        if config.get("dataset", {}).get("preserve_process_noise") is not True:
            gaps.append("dataset.preserve_process_noise")
    solver = config.get("solver", {})
    if solver.get("numerical_jitter_is_scientific_ridge") is not False:
        gaps.append("solver.numerical_jitter_is_scientific_ridge_false")
    return sorted(set(gaps))


def development_preflight_status(config: dict[str, Any]) -> str:
    return (
        "READY_FOR_DEVELOPMENT"
        if not development_preregistration_gaps(config)
        else "BLOCKED_BY_MISSING_PREREGISTRATION"
    )
