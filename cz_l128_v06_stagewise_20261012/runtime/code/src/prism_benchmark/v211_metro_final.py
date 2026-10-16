from __future__ import annotations

import gc
import json
import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cpu_data import BaseAccessor, ViewSpec, sha256_file
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_c import _ridge_fit, fit_physical_features
from .v2_config import load_frozen_config
from .v2_k import _cap, profile_values
from .v2_runtime import release_process_memory, run_parallel
from .v2_urysohn import basis_from_metadata, predict_contract
from .v211_a import (
    EXACT_ZERO,
    fit_mature_residual_ar,
    mature_residual_features,
    predict_mature_residual_ar,
)
from .v211_joint import (
    J_K,
    J_KA,
    J_KW,
    J_KWA,
    fit_joint_candidate,
    joint_w_basis,
    predict_joint_candidate,
)
from .v211_k import load_active_channels
from .v211_support import (
    SUPPORT_CONTRACT,
    apply_assembly_support,
    load_native_samples,
    support_id_hash,
)
from .v211_metro_config import (
    MetroV211Paths,
    effective_worker_count,
    load_metro_config,
    require_metro_test_freeze,
)
from .v211_metro_contracts import stable_candidate_id
from .v211_metro_views import metro_p60_dynamic_views
from .v211_w import (
    IDENTITY,
    MONOTONE,
    NATURAL_CUBIC,
    _ispline_fixed,
    fit_w_correction,
    predict_w_correction,
)
from .v211_joint_stability_config import (
    CHANNEL_COMPRESSED,
    FULL_BASIS,
    JOINT_ESTIMATOR_SEMANTICS,
)
from .v211_joint_stability import fit_joint_candidate_stability


PF_CANDIDATES = ("KC", "KCW", "KCA", "KCWA", "PF_SELECTED")
JOINT_CANDIDATES = (J_K, J_KW, J_KA, J_KWA, "J_SELECTED")


def _formal_candidate_names(formal_routes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    routes = set(formal_routes)
    if "PHYSICS_FIRST" not in routes:
        raise RuntimeError("M7 requires a frozen PHYSICS_FIRST route")
    return (*PF_CANDIDATES, *JOINT_CANDIDATES) if "JOINT" in routes else PF_CANDIDATES


@dataclass
class MetricAccumulator:
    rows: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    y_sum: float = 0.0
    y_squared_sum: float = 0.0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        y = np.asarray(target, dtype=np.float64)
        p = np.asarray(prediction, dtype=np.float64)
        error = y - p
        self.rows += len(y)
        self.squared_error += float(error @ error)
        self.absolute_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.y_sum += float(np.sum(y, dtype=np.float64))
        self.y_squared_sum += float(y @ y)

    def to_json(self) -> dict[str, Any]:
        mse = self.squared_error / max(1, self.rows)
        denominator = self.y_squared_sum - self.y_sum * self.y_sum / max(1, self.rows)
        return {
            "rows": self.rows,
            "mse": mse,
            "rmse": math.sqrt(max(0.0, mse)),
            "mae": self.absolute_error / max(1, self.rows),
            "r2": 1.0 - self.squared_error / denominator if denominator > 0 else float("nan"),
        }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"development prerequisite is not PASS: {path}")
    return value


def _result_path(output: Path, stage: str, view: ViewSpec) -> Path:
    if stage in {"C", "W"}:
        return output / "DEVELOPMENT" / stage / view.head.head_id / view.proxy_policy / "RESULT.json"
    return (
        output
        / "DEVELOPMENT"
        / stage
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )


def _development_prerequisites(
    paths: MetroV211Paths,
    view: ViewSpec,
    formal_routes: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Load only estimators that M6 explicitly froze for formal materialization."""
    result = {
        stage: _read(_result_path(paths.output, stage, view))
        for stage in ("C", "W", "A")
    }
    if "JOINT" in set(formal_routes):
        result["JOINT"] = _read(_result_path(paths.output, "JOINT", view))
    return result


def _row_id_hash(frame: pd.DataFrame) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in frame["base_origin_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_input_model(
    paths: MetroV211Paths,
    view: ViewSpec,
    fit: pd.DataFrame,
    active: list[dict[str, Any]],
    c_result: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    features = fit_physical_features(
        paths.shared,
        view,
        fit,
        fit,
        active,
        dict(v2),
        fit_split="validation",
        evaluation_split="validation",
    )
    target = fit["y_true"].to_numpy(dtype=np.float64)
    family = str(c_result["selected_family"])
    if family == "BEST_ACTIVE_K_CHANNEL":
        channel = str(c_result["best_active_k_channel"])
        index = features["channels"].index(channel)
        prediction = features["compressed_train"][:, index].copy()
        c_contract = {
            "family": family,
            "channel": channel,
            "intercept": 0.0,
            "coefficient": [1.0],
        }
    elif family == "K_EXACT_ZERO" or not active:
        intercept = float(np.mean(target, dtype=np.float64))
        prediction = np.full(len(fit), intercept, dtype=np.float64)
        c_contract = {"family": "K_EXACT_ZERO", "intercept": intercept, "coefficient": []}
    else:
        key = "joint" if family == "ADDITIVE_JOINT_BASIS" else "compressed"
        prediction, c_contract = _ridge_fit(
            features[f"{key}_train"],
            target,
            features[f"{key}_train"],
            float(c_result["selected_alpha"]),
        )
        c_contract = {"family": family, **c_contract}
    return {
        "channels": list(features["channels"]),
        "channel_contracts": features["channel_contracts"],
        "global_joint_columns": features.get("global_joint_columns", []),
        "c_contract": c_contract,
        "fit_prediction": prediction,
        "compressed_fit": features["compressed_train"],
        "joint_fit": features["joint_train"],
    }


def _predict_input_blocks(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    model: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    compressed = []
    joint = []
    for item in model["channel_contracts"]:
        values, _ = profile_values(
            accessor,
            samples,
            str(item["channel"]),
            tuple(int(value) for value in item["profile"]),
            int(item["m_tau"]),
        )
        k_contract = item["k_contract"]
        compressed.append(predict_contract(values, k_contract))
        basis = basis_from_metadata(k_contract["basis"])
        raw = basis.transform(values).reshape(len(samples), -1)
        selected = np.asarray(item["joint_columns"], dtype=np.int64)
        joint.append(raw[:, selected])
    compressed_matrix = (
        np.column_stack(compressed)
        if compressed
        else np.empty((len(samples), 0), dtype=np.float64)
    )
    joint_matrix = (
        np.concatenate(joint, axis=1)
        if joint
        else np.empty((len(samples), 0), dtype=np.float64)
    )
    global_columns = np.asarray(model.get("global_joint_columns", []), dtype=np.int64)
    if len(global_columns) and joint_matrix.shape[1] != len(global_columns):
        joint_matrix = joint_matrix[:, global_columns]
    return {"compressed": compressed_matrix, "joint": joint_matrix}


def _predict_c(blocks: Mapping[str, np.ndarray], contract: Mapping[str, Any], channels: list[str]) -> np.ndarray:
    family = str(contract["family"])
    rows = len(next(iter(blocks.values())))
    if family == "K_EXACT_ZERO":
        return np.full(rows, float(contract["intercept"]), dtype=np.float64)
    if family == "BEST_ACTIVE_K_CHANNEL":
        return blocks["compressed"][:, channels.index(str(contract["channel"]))].copy()
    key = "joint" if family == "ADDITIVE_JOINT_BASIS" else "compressed"
    matrix = blocks[key]
    return (
        (matrix - np.asarray(contract["mean"], dtype=np.float64))
        / np.asarray(contract["scale"], dtype=np.float64)
    ) @ np.asarray(contract["coefficient"], dtype=np.float64) + float(contract["intercept"])


def _partition_predictions(
    paths: MetroV211Paths,
    view: ViewSpec,
    model: Mapping[str, Any],
    pf_w_contract: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    chunk_rows: int,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for split, frame in frames.items():
        accessor = BaseAccessor(paths.shared, view.head.dataset, split, model["channels"])
        kc_chunks = []
        kcw_chunks = []
        for start in range(0, len(frame), chunk_rows):
            chunk = frame.iloc[start : start + chunk_rows]
            blocks = _predict_input_blocks(accessor, chunk, model)
            kc = _predict_c(blocks, model["c_contract"], model["channels"])
            correction = predict_w_correction(kc, dict(pf_w_contract))
            kc_chunks.append(kc)
            kcw_chunks.append(kc + correction)
            del blocks, kc, correction
        result[split] = {
            "KC": np.concatenate(kc_chunks) if kc_chunks else np.empty(0),
            "KCW": np.concatenate(kcw_chunks) if kcw_chunks else np.empty(0),
        }
        del accessor
        release_process_memory()
    return result


def _residual_source(
    frames: Mapping[str, pd.DataFrame],
    physical: Mapping[str, Mapping[str, np.ndarray]],
    candidate: str,
) -> pd.DataFrame:
    items = []
    for split, frame in frames.items():
        item = frame[["entity_id", "origin"]].copy()
        item["residual"] = frame["y_true"].to_numpy(dtype=np.float64) - physical[split][candidate]
        items.append(item)
    return pd.concat(items, ignore_index=True).sort_values(["entity_id", "origin"]).reset_index(drop=True)


def _fit_a_contract(
    view: ViewSpec,
    fit: pd.DataFrame,
    residual_target: np.ndarray,
    residual_source: pd.DataFrame,
    a_selection: Mapping[str, Any],
    v2: Mapping[str, Any],
    upstream: np.ndarray,
) -> dict[str, Any]:
    selected = a_selection["a_contract"]
    if selected["family"] == EXACT_ZERO:
        return {"family": EXACT_ZERO, "parameter_count": 0, "profile": selected.get("profile")}
    delta, history = (int(value) for value in selected["profile"])
    residual_mean = float(np.mean(residual_target, dtype=np.float64))
    features, coverage, maturity = mature_residual_features(
        fit,
        residual_source,
        h_steps=view.head.h_steps,
        w_steps=view.head.w_steps,
        delta=delta,
        history=history,
        maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
        residual_mean=residual_mean,
    )
    _, contract = fit_mature_residual_ar(
        features,
        residual_target,
        features,
        alpha=float(selected["alpha"]),
        mu=float(selected["soft_overlap_mu"]),
        upstream_predictions=upstream,
    )
    contract.update(
        {
            "profile": [delta, history],
            "residual_mean": residual_mean,
            "maturity_audit": maturity,
            "fit_coverage": coverage,
        }
    )
    return contract


def _predict_a(
    view: ViewSpec,
    samples: pd.DataFrame,
    source: pd.DataFrame,
    contract: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> np.ndarray:
    if contract["family"] == EXACT_ZERO:
        return np.zeros(len(samples), dtype=np.float64)
    delta, history = (int(value) for value in contract["profile"])
    features, _, _ = mature_residual_features(
        samples,
        source,
        h_steps=view.head.h_steps,
        w_steps=view.head.w_steps,
        delta=delta,
        history=history,
        maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
        residual_mean=float(contract["residual_mean"]),
    )
    return predict_mature_residual_ar(features, dict(contract))


def _predict_joint_w_design(latent: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    values = (
        np.asarray(latent, dtype=np.float64) - float(metadata["mean"])
    ) / float(metadata["scale"])
    knots = np.asarray(metadata["knots"], dtype=np.float64)
    if metadata["family"] == MONOTONE:
        return _ispline_fixed(
            values,
            knots,
            float(metadata["train_min"]),
            float(metadata["train_max"]),
        )
    if metadata["family"] == NATURAL_CUBIC:
        return natural_cubic_columns(values, knots)[:, 2:]
    raise ValueError(f"unsupported Joint W basis family: {metadata['family']}")


def _candidate_ids(
    view: ViewSpec,
    selected_pf: str,
    selected_joint: str | None,
    development_decision_sha256: str,
    *,
    formal_routes: list[str] | tuple[str, ...] = ("PHYSICS_FIRST", "JOINT"),
) -> dict[str, str]:
    common = {
        "view": view.relative_root.as_posix(),
        "development_decision_sha256": development_decision_sha256,
    }
    result: dict[str, str] = {
        name: stable_candidate_id("FINAL", {**common, "candidate": name})
        for name in PF_CANDIDATES[:-1]
    }
    result["PF_SELECTED"] = result[selected_pf]
    if "JOINT" in set(formal_routes):
        if selected_joint is None:
            raise RuntimeError("frozen Joint route lacks a selected candidate")
        result.update(
            {
                name: stable_candidate_id("FINAL", {**common, "candidate": name})
                for name in JOINT_CANDIDATES[:-1]
            }
        )
        result["J_SELECTED"] = result[selected_joint]
    return result


def _writer_frame(
    samples: pd.DataFrame,
    view: ViewSpec,
    label: str,
    candidate_id: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = samples[
        [
            "view_sample_id",
            "base_origin_id",
            "split",
            "entity_id",
            "origin",
            "latest_available_target_index",
            "y_true",
        ]
    ].copy().rename(columns={"view_sample_id": "sample_id"})
    frame["view"] = view.relative_root.as_posix()
    frame["candidate_id"] = candidate_id
    frame["candidate"] = label
    frame["y_pred"] = np.asarray(prediction, dtype=np.float64)
    frame["prediction_available"] = np.isfinite(frame["y_pred"].to_numpy())
    return frame


def materialize_view(
    paths: MetroV211Paths,
    view: ViewSpec,
    development_decision_sha256: str,
    formal_routes: list[str] | tuple[str, ...],
    frozen_candidate_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_metro_config(paths.project)
    v2 = load_frozen_config(paths.project)
    prerequisites = _development_prerequisites(paths, view, formal_routes)
    c_result = prerequisites["C"]
    w_result = prerequisites["W"]
    a_result = prerequisites["A"]
    joint_result = prerequisites.get("JOINT")
    include_joint = joint_result is not None
    active = [
        item
        for item in load_active_channels(paths.output, view)
        if item.get("channel") in set(c_result["active_channels"])
    ]
    anchor_frames = {
        split: load_native_samples(paths.shared, view, split)
        for split in ("train", "validation", "test", "ood")
    }
    frames = {
        split: apply_assembly_support(frame, active)
        for split, frame in anchor_frames.items()
    }
    development = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    development = development.sort_values(["entity_id", "origin"]).reset_index(drop=True)
    fit = _cap(development, int(v2["row_caps"]["joint_physical_fit"])).reset_index(drop=True)
    model = _fit_input_model(paths, view, fit, active, c_result, v2)
    fit_kc = np.asarray(model["fit_prediction"], dtype=np.float64)
    pf_w_selected = w_result["w_contract"]
    pf_w_ablation = w_result["pf_ablation_w_candidate"]
    fit_w_ablation, pf_w_ablation_contract = fit_w_correction(
        fit_kc,
        fit["y_true"].to_numpy(dtype=np.float64) - fit_kc,
        fit_kc,
        family=str(pf_w_ablation["family"]),
        knot_count=int(pf_w_ablation["knot_count"]),
        smoothness=float(pf_w_ablation["smoothness"]),
        mu=float(pf_w_ablation["soft_overlap_mu"]),
        upstream_predictions=model["compressed_fit"],
        direction=int(pf_w_ablation["direction"]),
    )
    if pf_w_selected["family"] == IDENTITY:
        pf_w_selected_contract = {
            "family": IDENTITY,
            "coefficient": [],
            "parameter_count": 0,
            "numerical_certificate": {"status": "EXACT_ZERO"},
        }
    else:
        pf_w_selected_contract = dict(pf_w_ablation_contract)
    fit_kcw = fit_kc + fit_w_ablation
    chunk_rows = int(config["resource"]["prediction_chunk_rows"])
    physical = _partition_predictions(
        paths, view, model, pf_w_ablation_contract, frames, chunk_rows
    )
    kc_source = _residual_source(frames, physical, "KC")
    kcw_source = _residual_source(frames, physical, "KCW")
    y_fit = fit["y_true"].to_numpy(dtype=np.float64)
    kca_contract = _fit_a_contract(
        view,
        fit,
        y_fit - fit_kc,
        kc_source,
        a_result,
        v2,
        model["compressed_fit"],
    )
    kcwa_contract = _fit_a_contract(
        view,
        fit,
        y_fit - fit_kcw,
        kcw_source,
        a_result,
        v2,
        np.column_stack([model["compressed_fit"], fit_w_ablation]),
    )
    joint_contracts: dict[str, Any] = {}
    joint_w_metadata: dict[str, Any] | None = None
    selected_joint: str | None = None
    delta = history = None
    if include_joint:
        joint_w_contract = w_result["joint_w_basis_contract"]
        joint_w_fit, _, joint_w_metadata = joint_w_basis(
            fit_kc, fit_kc, joint_w_contract
        )
        target_accessor = BaseAccessor(
            paths.shared, view.head.dataset, "validation", [view.head.target]
        )
        delta, history = (int(value) for value in joint_result["ar_profile"])
        joint_a_fit = target_accessor.target_state(
            fit, view.head.target, delta, history
        )
        for route in (J_K, J_KW, J_KA, J_KWA):
            selected = joint_result["route_local_selected"][route]
            semantics = joint_result.get("joint_estimator_semantics")
            if semantics == JOINT_ESTIMATOR_SEMANTICS:
                representation = str(selected["k_representation"])
                if representation == CHANNEL_COMPRESSED:
                    k_fit = model["compressed_fit"]
                elif representation == FULL_BASIS:
                    k_fit = model["joint_fit"]
                else:
                    raise RuntimeError(
                        f"unsupported frozen v2.2 K representation: {representation}"
                    )
                _, contract, _ = fit_joint_candidate_stability(
                    {"K": k_fit, "W": joint_w_fit, "A": joint_a_fit},
                    y_fit,
                    {"K": k_fit, "W": joint_w_fit, "A": joint_a_fit},
                    candidate=route,
                    k_representation=representation,
                    numerical_alpha=float(selected["numerical_alpha"]),
                    predictive_eta=float(selected["predictive_eta"]),
                    raw_k_support=tuple(model["channels"]),
                )
            elif semantics == "LEGACY_V211_JOINT":
                _, alpha, ratio_k, ratio_w = selected
                _, contract, _ = fit_joint_candidate(
                    {"K": model["joint_fit"], "W": joint_w_fit, "A": joint_a_fit},
                    y_fit,
                    {"K": model["joint_fit"], "W": joint_w_fit, "A": joint_a_fit},
                    candidate=route,
                    alpha=float(alpha),
                    k_over_a_ratio=float(ratio_k),
                    w_over_a_ratio=float(ratio_w),
                )
            else:
                raise RuntimeError("STOP_ESTIMATOR_SEMANTICS_UNBOUND")
            joint_contracts[route] = contract
        selected_joint = str(joint_result["final_selected_candidate"])
    selected_w_active = pf_w_selected["family"] != IDENTITY
    selected_a_active = a_result["a_contract"]["family"] != EXACT_ZERO
    selected_pf = (
        "KCWA"
        if selected_w_active and selected_a_active
        else "KCW"
        if selected_w_active
        else "KCA"
        if selected_a_active
        else "KC"
    )
    if str(a_result.get("pf_selected_route")) != selected_pf:
        raise RuntimeError("frozen PF selected route disagrees with development materialization")
    candidate_ids = _candidate_ids(
        view,
        selected_pf,
        selected_joint,
        development_decision_sha256,
        formal_routes=formal_routes,
    )
    if frozen_candidate_ids is not None:
        expected_names = set(_formal_candidate_names(formal_routes))
        if set(frozen_candidate_ids) != expected_names:
            raise RuntimeError("M7 frozen candidate-ID set differs from formal routes")
        candidate_ids = {name: str(frozen_candidate_ids[name]) for name in expected_names}
    output_root = (
        paths.output
        / "FINAL"
        / "predictions"
        / view.head.head_id
        / view.information_set
        / view.availability_scenario
        / view.proxy_policy
    )
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_paths: dict[str, dict[str, str]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for split in ("test", "ood"):
        frame = frames[split]
        accessor = BaseAccessor(
            paths.shared,
            view.head.dataset,
            split,
            [*model["channels"], view.head.target],
        )
        writers: dict[str, pq.ParquetWriter] = {}
        candidate_names = _formal_candidate_names(formal_routes)
        accumulators = {name: MetricAccumulator() for name in candidate_names}
        paths_by_name = {name: output_root / f"{split}_{name}.parquet" for name in accumulators}
        try:
            for start in range(0, len(frame), chunk_rows):
                chunk = frame.iloc[start : start + chunk_rows]
                blocks = _predict_input_blocks(accessor, chunk, model)
                kc = _predict_c(blocks, model["c_contract"], model["channels"])
                w = predict_w_correction(kc, pf_w_ablation_contract)
                kcw = kc + w
                kca = kc + _predict_a(view, chunk, kc_source, kca_contract, v2)
                kcwa = kcw + _predict_a(view, chunk, kcw_source, kcwa_contract, v2)
                predictions = {
                    "KC": kc,
                    "KCW": kcw,
                    "KCA": kca,
                    "KCWA": kcwa,
                    "PF_SELECTED": {
                        "KC": kc,
                        "KCW": kcw,
                        "KCA": kca,
                        "KCWA": kcwa,
                    }[selected_pf],
                }
                if include_joint:
                    joint_w_chunk = _predict_joint_w_design(kc, joint_w_metadata)
                    joint_a_chunk = accessor.target_state(
                        chunk, view.head.target, delta, history
                    )
                    joint_predictions = {
                        route: predict_joint_candidate(
                            {
                                "K": (
                                    blocks["compressed"]
                                    if joint_contracts[route].get("k_representation")
                                    == CHANNEL_COMPRESSED
                                    else blocks["joint"]
                                ),
                                "W": joint_w_chunk,
                                "A": joint_a_chunk,
                            },
                            joint_contracts[route],
                        )[0]
                        for route in (J_K, J_KW, J_KA, J_KWA)
                    }
                    predictions.update(joint_predictions)
                    predictions["J_SELECTED"] = joint_predictions[selected_joint]
                y = chunk["y_true"].to_numpy(dtype=np.float64)
                for name, prediction in predictions.items():
                    materialized = _writer_frame(chunk, view, name, candidate_ids[name], prediction)
                    table = pa.Table.from_pandas(materialized, preserve_index=False)
                    if name not in writers:
                        writers[name] = pq.ParquetWriter(
                            paths_by_name[name], table.schema, compression="zstd"
                        )
                    writers[name].write_table(table)
                    accumulators[name].update(y, prediction)
                del blocks, predictions
                gc.collect()
        finally:
            for writer in writers.values():
                writer.close()
        prediction_paths[split] = {
            name: str(path.relative_to(paths.output)) for name, path in paths_by_name.items()
        }
        for name, path in paths_by_name.items():
            metrics[f"{split}:{name}"] = {
                "split": split,
                "candidate": name,
                "candidate_id": candidate_ids[name],
                "prediction_path": str(path.relative_to(paths.output)),
                "prediction_sha256": sha256_file(path),
                **accumulators[name].to_json(),
            }
        del accessor
        release_process_memory()
    contract_root = paths.output / "FINAL" / "contracts" / view.proxy_policy
    contract_root.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "status": "PASS",
        "view": view.relative_root.as_posix(),
        "fit_row_cap": int(v2["row_caps"]["joint_physical_fit"]),
        "fit_rows": len(fit),
        "fit_row_id_sha256": _row_id_hash(fit),
        "sample_support_contract": SUPPORT_CONTRACT,
        "assembly_support_contract": c_result.get(
            "assembly_support_contract"
        ),
        "assembly_support_hashes": {
            split: support_id_hash(frame) for split, frame in frames.items()
        },
        "assembly_support_applied_before_selected_k_features": True,
        "prediction_chunk_rows": chunk_rows,
        "candidate_ids": candidate_ids,
        "formal_routes": list(formal_routes),
        "selected_pf": selected_pf,
        "selected_joint": selected_joint,
        "joint_status": (
            "JOINT_PREDICTIVE_VALIDATED"
            if include_joint
            else "NOT_APPLICABLE_NOT_FROZEN"
        ),
        "input_model": {
            "channels": model["channels"],
            "channel_contracts": model["channel_contracts"],
            "global_joint_columns": model["global_joint_columns"],
            "c_contract": model["c_contract"],
        },
        "pf_selected_w_contract": pf_w_selected_contract,
        "pf_ablation_w_contract": pf_w_ablation_contract,
        "pf_ablation_selection_eligible": False,
        "kca_contract": kca_contract,
        "kcwa_contract": kcwa_contract,
        "prediction_paths": prediction_paths,
        "metrics": list(metrics.values()),
        "test_accessed": True,
        "ood_accessed": True,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if include_joint:
        contract_payload.update(
            {
                "joint_w_basis_metadata": joint_w_metadata,
                "joint_contracts": joint_contracts,
            }
        )
    contract_path = contract_root / "FINAL_MODEL_CONTRACT.json"
    write_json(contract_path, contract_payload)
    return {
        "status": "PASS",
        "view": view.relative_root.as_posix(),
        "proxy_policy": view.proxy_policy,
        "contract_path": str(contract_path.relative_to(paths.output)),
        "contract_sha256": sha256_file(contract_path),
        "metrics": list(metrics.values()),
        "test_accessed": True,
        "ood_accessed": True,
        "wall_seconds": contract_payload["wall_seconds"],
        "peak_rss_kib": contract_payload["peak_rss_kib"],
    }


def run_m7(paths: MetroV211Paths) -> dict[str, Any]:
    manifest = require_metro_test_freeze(paths)
    formal_routes = list(manifest.get("formal_routes", ()))
    _formal_candidate_names(formal_routes)
    if "JOINT" not in formal_routes and manifest.get(
        "joint_formal_test_eligible"
    ) is not False:
        raise RuntimeError("PF-only freeze must explicitly exclude Joint from M7")
    freeze_sha256 = sha256_file(paths.development_freeze_path)
    development_decision_sha256 = manifest["development_decision_sha256"]
    first_access_timestamp = time.time()
    write_json(
        paths.test_access_audit_path,
        {
            "status": "LOCKBOX_ACCESS_STARTED",
            "stage": "M7_TEST_OOD_MATERIALIZATION",
            "freeze_sha256": freeze_sha256,
            "frozen_code_commit": manifest["code_commit"],
            "config_sha256": manifest["config_sha256"],
            "theory_sha256": manifest["canonical_theory_sha256"],
            "first_access_timestamp": first_access_timestamp,
            "test_accessed": False,
            "ood_accessed": False,
            "views": [],
        },
    )
    config = load_metro_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    frozen_ids = {
        str(item["view"]): item["candidate_ids"]
        for item in manifest["pending_materialization_candidate_ids"]
    }
    results = run_parallel(
        materialize_view,
        [
            (
                paths,
                view,
                development_decision_sha256,
                formal_routes,
                frozen_ids[view.relative_root.as_posix()],
            )
            for view in views
        ],
        effective_worker_count(config),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")),
        label="PRISM_V211_METRO_M7_TEST_OOD",
    )
    candidate_ids_match = True
    for item in results:
        contract = _read(paths.output / item["contract_path"])
        candidate_ids_match &= contract["candidate_ids"] == frozen_ids[contract["view"]]
    status = (
        "PASS"
        if all(item.get("status") == "PASS" for item in results)
        and candidate_ids_match
        else "FAILED"
    )
    audit = {
        "status": status,
        "stage": "M7_TEST_OOD_MATERIALIZATION",
        "freeze_sha256": freeze_sha256,
        "frozen_code_commit": manifest["code_commit"],
        "config_sha256": manifest["config_sha256"],
        "theory_sha256": manifest["canonical_theory_sha256"],
        "first_access_timestamp": first_access_timestamp,
        "views": results,
        "candidate_ids_match_m6_freeze": candidate_ids_match,
        "test_accessed": True,
        "ood_accessed": True,
    }
    write_json(paths.test_access_audit_path, audit)
    run_status = json.loads((paths.output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    run_status.update(
        {
            "status": status,
            "stage": "M7",
            "development_frozen": True,
            "test_accessed": True,
            "ood_accessed": True,
        }
    )
    write_json(paths.output / "RUN_STATUS.json", run_status)
    if status != "PASS":
        raise RuntimeError("M7 final materialization failed after lockbox access")
    return audit
