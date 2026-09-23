"""Authority-preserving pure-K checkpoint derivation and replay for CZ.

The authoritative representative checkpoint starts at K+C.  This adapter does
not fit a new estimator.  It extracts the best active K channel selected by the
authority development run, reuses the frozen K physical contracts, and writes
a separate portable inference checkpoint.  The adapter intentionally imports
the authority feature and prediction functions rather than copying them.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec
from .portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    assert_inference_only,
    checkpoint_key,
    load_portable_checkpoint,
    write_portable_checkpoint,
)
from .stage0 import write_json
from .v211_public_all_closure import common_support_record
from .v211_public_all_baselines import SupportRequirement
from .v211_public_all_config import PublicAllPaths
from .v211_support import support_id_hash
from .v211_c import BEST_ACTIVE_K


PURE_K_MODEL = "PRISM_V2_1_1_K"
PURE_K_CODEC = "PRISM_AUTHORITY_PURE_K_PIPELINE_V1"


@contextmanager
def _checkpoint_read_context() -> Iterator[None]:
    previous = os.environ.get(INFERENCE_ONLY_ENV)
    os.environ[INFERENCE_ONLY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(INFERENCE_ONLY_ENV, None)
        else:
            os.environ[INFERENCE_ONLY_ENV] = previous


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _authority_checkpoint_dir(root: Path, view: ViewSpec) -> Path:
    return root / "prism" / checkpoint_key(
        view.head.dataset,
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
    )


def pure_k_checkpoint_dir(root: Path, view: ViewSpec) -> Path:
    return root / "pure_k" / checkpoint_key(
        view.head.dataset,
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
    )


def _c_result_path(paths: PublicAllPaths, view: ViewSpec) -> Path:
    return (
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )


def pure_k_contract_from_c_result(
    c_result: Mapping[str, Any],
    physical_channels: list[str],
    source_c_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the authority K parent contract used immediately before C."""

    active = [str(value) for value in c_result.get("active_channels", [])]
    if not active:
        if str(c_result.get("selected_family")) != "K_EXACT_ZERO":
            raise RuntimeError("STOP_PURE_K_EMPTY_ACTIVE_SET_WITHOUT_K_EXACT_ZERO")
        source = dict(source_c_contract or {})
        if source.get("family") != "K_EXACT_ZERO" or "intercept" not in source:
            raise RuntimeError("STOP_PURE_K_EXACT_ZERO_INTERCEPT_NOT_FROZEN")
        return {
            "family": "K_EXACT_ZERO",
            "intercept": float(source["intercept"]),
            "coefficient": [],
            "parameter_count": 1,
        }

    channel = str(c_result.get("best_active_k_channel", ""))
    if not channel or channel not in active or channel not in physical_channels:
        raise RuntimeError("STOP_PURE_K_BEST_ACTIVE_CHANNEL_NOT_FROZEN")
    return {
        "family": BEST_ACTIVE_K,
        "channel": channel,
        "intercept": 0.0,
        "coefficient": [1.0],
        "parameter_count": 1,
    }


def predict_pure_k_from_compressed(
    compressed: np.ndarray,
    contract: Mapping[str, Any],
    physical_channels: list[str],
) -> np.ndarray:
    matrix = np.asarray(compressed, dtype=np.float64)
    family = str(contract["family"])
    if family == "K_EXACT_ZERO":
        return np.full(len(matrix), float(contract["intercept"]), dtype=np.float64)
    if family != BEST_ACTIVE_K:
        raise RuntimeError(f"STOP_UNSUPPORTED_PURE_K_CONTRACT:{family}")
    channel = str(contract["channel"])
    if channel not in physical_channels:
        raise RuntimeError("STOP_PURE_K_CHANNEL_ABSENT_FROM_PHYSICAL_MATRIX")
    return matrix[:, physical_channels.index(channel)].copy()


def _common_development(paths: PublicAllPaths, view: ViewSpec) -> pd.DataFrame:
    from .v211_public_all_materialization import _development

    record = common_support_record(paths, view)
    requirements = [SupportRequirement(**item) for item in record.get("requirements", [])]
    return _development(paths.shared, view, requirements or [SupportRequirement()])


def derive_pure_k_checkpoint_for_view(
    paths: PublicAllPaths,
    view: ViewSpec,
    authority_checkpoint_root: Path,
    pure_k_root: Path,
) -> dict[str, Any]:
    """Derive a no-refit pure-K checkpoint from the sealed authority checkpoint."""

    source = _authority_checkpoint_dir(authority_checkpoint_root, view)
    with _checkpoint_read_context():
        state, arrays, manifest = load_portable_checkpoint(source)
    if state.get("codec") != "PRISM_KCWA_JOINT_PIPELINE":
        raise RuntimeError("STOP_PURE_K_SOURCE_CODEC_MISMATCH")
    c_result = _read_json(_c_result_path(paths, view))
    if c_result.get("status") != "PASS":
        raise RuntimeError("STOP_PURE_K_C_DEVELOPMENT_NOT_PASS")
    physical = dict(state["physical"])
    channels = [str(value) for value in physical.get("channels", [])]
    contract = pure_k_contract_from_c_result(
        c_result, channels, dict(state.get("c_contract", {}))
    )
    replay = predict_pure_k_from_compressed(arrays["compressed"], contract, channels)
    metadata = {
        "codec": PURE_K_CODEC,
        "artifact_type": "DETERMINISTIC_DERIVATION_FROM_SEALED_AUTHORITY_CHECKPOINT",
        "family": "PRISM",
        "model": PURE_K_MODEL,
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "physical": physical,
        "k_contract": contract,
        "fit_partition": state["fit_partition"],
        "fit_rows": int(state["fit_rows"]),
        "fit_support_hash": str(state["fit_support_hash"]),
        "feature_order": channels,
        "source_authority_checkpoint": str(source),
        "source_authority_checkpoint_hash": str(manifest["checkpoint_hash"]),
        "source_selection_hash": str(state["selection_hash"]),
        "refit_performed": False,
        "test_metrics_used": False,
        "reload_prediction_tolerance": 1e-10,
    }
    destination = pure_k_checkpoint_dir(pure_k_root, view)
    if destination.exists():
        with _checkpoint_read_context():
            existing_state, existing_arrays, existing_manifest = load_portable_checkpoint(
                destination
            )
        bindings_match = (
            existing_state.get("codec") == PURE_K_CODEC
            and existing_state.get("source_authority_checkpoint_hash")
            == manifest["checkpoint_hash"]
            and existing_state.get("source_selection_hash") == state["selection_hash"]
            and existing_state.get("k_contract") == contract
            and existing_state.get("target_head") == view.head.head_id
            and existing_state.get("information_set") == view.information_set
            and existing_state.get("refit_performed") is False
        )
        prediction_error = float(
            np.max(
                np.abs(existing_arrays["pure_k_prediction"] - replay), initial=0.0
            )
        )
        if not bindings_match or prediction_error > 1e-10:
            raise RuntimeError("STOP_EXISTING_PURE_K_CHECKPOINT_BINDING_MISMATCH")
        return {
            "status": "PASS",
            "model": PURE_K_MODEL,
            "checkpoint_dir": str(destination),
            "checkpoint_hash": str(existing_manifest["checkpoint_hash"]),
            "source_checkpoint_hash": str(manifest["checkpoint_hash"]),
            "fit_rows": int(state["fit_rows"]),
            "fit_support_hash": str(state["fit_support_hash"]),
            "refit_performed": False,
            "test_accessed": False,
            "reused_verified_checkpoint": True,
            "maximum_absolute_replay_error": prediction_error,
        }
    derived = write_portable_checkpoint(
        destination,
        metadata,
        {
            "compressed": np.asarray(arrays["compressed"], dtype=np.float64),
            "pure_k_prediction": replay,
        },
    )
    return {
        "status": "PASS",
        "model": PURE_K_MODEL,
        "checkpoint_dir": str(destination),
        "checkpoint_hash": str(derived["checkpoint_hash"]),
        "source_checkpoint_hash": str(manifest["checkpoint_hash"]),
        "fit_rows": int(state["fit_rows"]),
        "fit_support_hash": str(state["fit_support_hash"]),
        "refit_performed": False,
        "test_accessed": False,
        "reused_verified_checkpoint": False,
    }


def verify_pure_k_checkpoint_reload(checkpoint: Path) -> dict[str, Any]:
    with _checkpoint_read_context():
        state, arrays, manifest = load_portable_checkpoint(checkpoint)
    if state.get("codec") != PURE_K_CODEC or state.get("refit_performed") is not False:
        raise RuntimeError("STOP_INVALID_PURE_K_CHECKPOINT_STATE")
    observed = predict_pure_k_from_compressed(
        arrays["compressed"], state["k_contract"], list(state["physical"]["channels"])
    )
    maximum = float(
        np.max(np.abs(observed - arrays["pure_k_prediction"]), initial=0.0)
    )
    tolerance = float(state.get("reload_prediction_tolerance", 1e-10))
    if maximum > tolerance:
        raise RuntimeError("STOP_PURE_K_CHECKPOINT_RELOAD_MISMATCH")
    return {
        "status": "PASS",
        "checkpoint_hash": str(manifest["checkpoint_hash"]),
        "maximum_absolute_prediction_error": maximum,
        "tolerance": tolerance,
    }


def pure_k_oof_identity_certificate(
    paths: PublicAllPaths,
    view: ViewSpec,
    pure_k_root: Path,
) -> dict[str, Any]:
    """Certify that C selected its own nested-OOF K parent unchanged.

    C has its own outer-fold K refits, so those predictions must not be compared
    with the final full-development checkpoint or the K stage's different fold
    system.  The sealed C selector contract and its selected OOF artifact are
    the authoritative evidence for the development identity route.
    """

    from .level_reconstruction import support_hash

    checkpoint = pure_k_checkpoint_dir(pure_k_root, view)
    with _checkpoint_read_context():
        state, _, manifest = load_portable_checkpoint(checkpoint)
    c_result = _read_json(_c_result_path(paths, view))
    selection = dict(c_result.get("family_selection") or {})
    identity_checks = {
        "routing_zero_identity": c_result.get("routing_status") == "ZERO_IDENTITY",
        "selection_status": c_result.get("selection_status") == "C_ZERO_IDENTITY",
        "selected_family_is_k_parent": c_result.get("selected_family") == BEST_ACTIVE_K,
        "selector_inactive": selection.get("active") is False,
        "selector_route_zero_identity": selection.get("routing_status") == "ZERO_IDENTITY",
        "selector_identity_is_k_parent": selection.get("identity") == BEST_ACTIVE_K,
        "final_candidate_is_identity": selection.get("final_selected_candidate")
        == BEST_ACTIVE_K,
    }
    if not all(identity_checks.values()):
        raise RuntimeError("STOP_PURE_K_OOF_IDENTITY_REQUIRES_C_ZERO_IDENTITY")
    if (
        state.get("k_contract", {}).get("family") != BEST_ACTIVE_K
        or state.get("k_contract", {}).get("channel")
        != c_result.get("best_active_k_channel")
    ):
        raise RuntimeError("STOP_PURE_K_FINAL_CONTRACT_NOT_BOUND_TO_C_PARENT")
    c_path = (
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "SELECTED_OOF.parquet"
    )
    c_frame = pd.read_parquet(c_path)
    required = {"base_origin_id", "oof_fold", "y_true", "y_pred"}
    if not required.issubset(c_frame.columns):
        raise RuntimeError("STOP_C_SELECTED_OOF_SCHEMA_MISMATCH")
    observed_losses = [
        float(np.mean(np.square(group["y_true"] - group["y_pred"]), dtype=np.float64))
        for _, group in c_frame.groupby("oof_fold", sort=True)
    ]
    parent_losses = [float(value) for value in selection["parent_outer_fold_losses"]]
    final_losses = [float(value) for value in selection["final_selected_fold_losses"]]
    best_k_losses = [float(value) for value in c_result["best_active_k_fold_losses"]]
    if not (len(observed_losses) == len(parent_losses) == len(final_losses) == len(best_k_losses)):
        raise RuntimeError("STOP_PURE_K_C_IDENTITY_FOLD_COUNT_MISMATCH")
    maximum = float(
        np.max(
            np.abs(
                np.asarray(
                    [*np.subtract(observed_losses, final_losses),
                     *np.subtract(parent_losses, final_losses),
                     *np.subtract(best_k_losses, final_losses)],
                    dtype=np.float64,
                )
            ),
            initial=0.0,
        )
    )
    if maximum > 1e-15:
        raise RuntimeError("STOP_PURE_K_C_IDENTITY_FOLD_LOSS_MISMATCH")
    return {
        "status": "PASS",
        "model": PURE_K_MODEL,
        "rows": int(len(c_frame)),
        "folds": int(len(observed_losses)),
        "validation_base_origin_order_hash": support_hash(
            c_frame["base_origin_id"].astype(str).tolist()
        ),
        "fit_support_hash": str(state["fit_support_hash"]),
        "checkpoint_hash": str(manifest["checkpoint_hash"]),
        "c_routing_status": str(c_result["routing_status"]),
        "c_selected_family": str(c_result["selected_family"]),
        "identity_checks": identity_checks,
        "observed_oof_fold_losses": observed_losses,
        "maximum_absolute_fold_loss_error": maximum,
        "tolerance": 1e-15,
        "refit_performed": False,
        "test_accessed": False,
    }


def predict_pure_k_checkpoint_for_view(
    paths: PublicAllPaths,
    view: ViewSpec,
    pure_k_root: Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Inference-only replay of the derived pure-K checkpoint."""

    assert_inference_only()
    from .representative_prism_checkpoints import (
        _predict_c,
        _predict_physical_features,
        _write_model,
    )
    from .v211_public_all_baseline_materialization import _common_test

    started = time.time()
    checkpoint = pure_k_checkpoint_dir(pure_k_root, view)
    state, _, manifest = load_portable_checkpoint(checkpoint)
    samples = _common_test(paths, view, split)
    matrices = _predict_physical_features(
        paths, view, samples, split, state["physical"]
    )
    prediction = _predict_c(
        matrices, state["k_contract"], list(state["physical"]["channels"])
    )
    record = _write_model(
        paths,
        view,
        samples,
        PURE_K_MODEL,
        prediction,
        int(state["k_contract"].get("parameter_count", 0)),
        checkpoint,
        str(manifest["checkpoint_hash"]),
        str(state["fit_support_hash"]),
        started,
        split,
    )
    write_json(
        paths.output
        / split.upper()
        / "PURE_K_INFERENCE_RESULT.json",
        {
            "status": "PASS",
            "model": record,
            "inference_only": True,
            "refit_performed": False,
        },
    )
    return record


def pure_k_formal_identity_certificate(
    paths: PublicAllPaths,
    view: ViewSpec,
    pure_k_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay pure K and prove equality to sealed formal K+C when C is identity.

    The sealed authority run is read-only.  The newly materialized pure-K
    prediction is written below ``output_root`` and is never used for model
    selection.
    """

    from .level_reconstruction import metric_bundle_delta_and_level, support_hash
    from .representative_prism_checkpoints import (
        _current_levels,
        _predict_c,
        _predict_physical_features,
    )
    from .v211_public_all_baseline_materialization import _common_test
    from .v211_public_all_materialization import _prediction_frame, _prediction_root

    c_result = _read_json(_c_result_path(paths, view))
    if c_result.get("routing_status") != "ZERO_IDENTITY":
        raise RuntimeError("STOP_PURE_K_FORMAL_IDENTITY_REQUIRES_C_ZERO_IDENTITY")
    checkpoint = pure_k_checkpoint_dir(pure_k_root, view)
    with _checkpoint_read_context():
        state, _, manifest = load_portable_checkpoint(checkpoint)
        samples = _common_test(paths, view, "test")
        matrices = _predict_physical_features(
            paths, view, samples, "test", state["physical"]
        )
        prediction = _predict_c(
            matrices, state["k_contract"], list(state["physical"]["channels"])
        )

    model = f"{PURE_K_MODEL}_DYNAMIC" if view.information_set == "dynamic" else PURE_K_MODEL
    frame = _prediction_frame(samples, view, model, prediction, 1)
    frame["split"] = "test"
    destination = output_root / "test_predictions" / view.relative_root / f"{model}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")

    parent_model = (
        "PRISM_V2_1_1_K_C_DYNAMIC"
        if view.information_set == "dynamic"
        else "PRISM_V2_1_1_K_C"
    )
    parent_path = _prediction_root(paths, "test") / view.relative_root / f"{parent_model}.parquet"
    parent = pd.read_parquet(parent_path)
    merged = frame[["base_origin_id", "y_true", "y_pred"]].rename(
        columns={"y_true": "pure_y_true", "y_pred": "pure_y_pred"}
    ).merge(
        parent[["base_origin_id", "y_true", "y_pred"]].rename(
            columns={"y_true": "parent_y_true", "y_pred": "parent_y_pred"}
        ),
        on="base_origin_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame) or len(merged) != len(parent):
        raise RuntimeError("STOP_PURE_K_FORMAL_SUPPORT_MISMATCH")
    truth_error = float(
        np.max(
            np.abs(
                merged["pure_y_true"].to_numpy(dtype=np.float64)
                - merged["parent_y_true"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    prediction_error = float(
        np.max(
            np.abs(
                merged["pure_y_pred"].to_numpy(dtype=np.float64)
                - merged["parent_y_pred"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    if truth_error != 0.0 or prediction_error > 1e-10:
        raise RuntimeError("STOP_PURE_K_FORMAL_PARENT_IDENTITY_MISMATCH")
    metric = metric_bundle_delta_and_level(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
        _current_levels(paths, view, samples, "test"),
    )
    metric.pop("future_level_true")
    metric.pop("future_level_pred")
    return {
        "status": "PASS",
        "model": model,
        "parent_model": parent_model,
        "rows": int(len(frame)),
        "sample_id_order_hash": support_hash(frame["sample_id"].astype(str).tolist()),
        "scoring_support_hash": support_id_hash(samples),
        "fit_support_hash": str(state["fit_support_hash"]),
        "checkpoint_hash": str(manifest["checkpoint_hash"]),
        "prediction_path": str(destination),
        "sealed_parent_prediction_path": str(parent_path),
        "maximum_absolute_truth_error": truth_error,
        "maximum_absolute_prediction_error": prediction_error,
        "tolerance": 1e-10,
        "c_routing_status": "ZERO_IDENTITY",
        "refit_performed": False,
        "test_metrics_used_for_selection": False,
        **metric,
    }
