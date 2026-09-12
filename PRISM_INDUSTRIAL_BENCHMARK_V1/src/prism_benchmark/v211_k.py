from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    inner_folds,
    input_columns,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_basis import tensor_design
from .v2_k import (
    CHANNEL_SAMPLE_COLUMNS,
    FAMILY_ORDER,
    _als_kwargs,
    _candidate_valid,
    _cap,
    channel_profiles,
    profile_values,
)
from .strict_oof_selection import strict_nested_oof_select, tune_best_nonzero
from .v2_runtime import run_parallel
from .v2_urysohn import (
    PreparedContractFit,
    fit_contract,
    fit_prepared_contract,
    predict_contract,
    predict_contract_from_design,
    prepare_contract_fit,
)
from .v21_views import sru_input_views
from .v211_config import load_v211_configs
from .v211_support import (
    SUPPORT_CONTRACT,
    apply_native_support,
    load_native_samples,
    registered_fold_native_masks,
    support_audit,
    support_id_hash,
)


EXACT_ZERO = "EXACT_ZERO"
K_INNER_WORKERS_ENV = "PRISM_V211_K_INNER_WORKERS"
K_CHECKPOINT_BATCH_ENV = "PRISM_V211_K_CHECKPOINT_BATCH"
K_NUMERICAL_ENGINE = "PRISM_V211_K_SUFFICIENT_STATISTICS_ALS_V1"


def _k_checkpoint_batch() -> int:
    raw = os.environ.get(K_CHECKPOINT_BATCH_ENV, "16")
    try:
        batch = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{K_CHECKPOINT_BATCH_ENV} must be an integer") from error
    if batch < 1:
        raise RuntimeError(f"{K_CHECKPOINT_BATCH_ENV} must be positive")
    return batch


def _k_inner_workers() -> int:
    raw = os.environ.get(K_INNER_WORKERS_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{K_INNER_WORKERS_ENV} must be an integer") from error
    cpu_count = os.cpu_count() or 1
    if workers < 1 or workers > cpu_count:
        raise RuntimeError(f"{K_INNER_WORKERS_ENV} must be within [1, {cpu_count}]")
    return workers


def _ordered_parallel_map(
    function: Callable[..., Any],
    jobs: Sequence[tuple[Any, ...]],
    workers: int,
    *,
    executor: ThreadPoolExecutor | None = None,
) -> list[Any]:
    """Evaluate independent jobs concurrently while preserving registration order."""
    if workers <= 1 or len(jobs) <= 1:
        return [function(*arguments) for arguments in jobs]
    if executor is not None:
        futures = [executor.submit(function, *arguments) for arguments in jobs]
        return [future.result() for future in futures]
    with ThreadPoolExecutor(max_workers=min(int(workers), len(jobs))) as local_executor:
        futures = [
            local_executor.submit(function, *arguments) for arguments in jobs
        ]
        return [future.result() for future in futures]


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonable_structural_spec(spec: tuple[Any, ...]) -> list[Any]:
    profile, m_tau, family, m_x, lambda_tau, lambda_x = spec
    return [
        [int(profile[0]), int(profile[1])],
        int(m_tau),
        str(family),
        int(m_x),
        float(lambda_tau),
        float(lambda_x),
    ]


def _structural_specs(
    profiles: Sequence[tuple[int, int]],
    m_tau_values: Sequence[int],
    lambda_tau_values: Sequence[float],
    m_x_values: Sequence[int],
    lambda_x_values: Sequence[float],
) -> list[tuple[Any, ...]]:
    specs: list[tuple[Any, ...]] = []
    for profile in profiles:
        for m_tau in m_tau_values:
            for lambda_tau in lambda_tau_values:
                specs.append(
                    (
                        tuple(profile),
                        int(m_tau),
                        "LINEAR_DISTRIBUTED_LAG",
                        1,
                        float(lambda_tau),
                        0.0,
                    )
                )
                for m_x in m_x_values:
                    for family in FAMILY_ORDER[2:]:
                        for lambda_x in lambda_x_values:
                            specs.append(
                                (
                                    tuple(profile),
                                    int(m_tau),
                                    str(family),
                                    int(m_x),
                                    float(lambda_tau),
                                    float(lambda_x),
                                )
                            )
    return specs


def _checkpoint_fingerprint(
    specs: Sequence[tuple[Any, ...]], context: Mapping[str, Any]
) -> str:
    payload = {
        "schema": "PRISM_V211_K_STRUCTURAL_CHECKPOINT_V1",
        "context": dict(context),
        "specs": [_jsonable_structural_spec(spec) for spec in specs],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _StructuralCheckpoint:
    def __init__(
        self,
        root: Path,
        specs: Sequence[tuple[Any, ...]],
        context: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.specs = list(specs)
        self.fingerprint = _checkpoint_fingerprint(self.specs, context)
        self.manifest_path = root / "MANIFEST.json"
        self.progress_path = root.parent / "STRUCTURAL_PROGRESS.json"
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "PRISM_V211_K_STRUCTURAL_CHECKPOINT_V1",
            "fingerprint": self.fingerprint,
            "candidate_count": len(self.specs),
            "context": dict(context),
            "test_accessed": False,
            "ood_accessed": False,
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise RuntimeError("structural checkpoint manifest mismatch")
        else:
            _atomic_write_json(self.manifest_path, manifest)
        self.losses = self._load_batches()
        self.reused = len(self.losses)
        self._write_progress("RUNNING")

    def _load_batches(self) -> list[list[float]]:
        losses: list[list[float]] = []
        for path in sorted(self.root.glob("BATCH_*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            start = int(value["start"])
            items = list(value["items"])
            if value.get("fingerprint") != self.fingerprint or start != len(losses):
                raise RuntimeError("structural checkpoint batch is not a valid prefix")
            for item in items:
                index = len(losses)
                if int(item["index"]) != index:
                    raise RuntimeError("structural checkpoint candidate index mismatch")
                if item["spec"] != _jsonable_structural_spec(self.specs[index]):
                    raise RuntimeError("structural checkpoint candidate registry mismatch")
                fold_losses = [float(value) for value in item["losses"]]
                losses.append(fold_losses)
        return losses

    def append(self, start: int, values: Sequence[list[float]]) -> None:
        if start != len(self.losses):
            raise RuntimeError("structural checkpoint append is not contiguous")
        items = [
            {
                "index": start + offset,
                "spec": _jsonable_structural_spec(self.specs[start + offset]),
                "losses": [float(loss) for loss in losses],
            }
            for offset, losses in enumerate(values)
        ]
        stop = start + len(items)
        path = self.root / f"BATCH_{start:06d}_{stop:06d}.json"
        if path.exists():
            raise RuntimeError("structural checkpoint batch already exists")
        _atomic_write_json(
            path,
            {
                "fingerprint": self.fingerprint,
                "start": start,
                "stop": stop,
                "items": items,
                "test_accessed": False,
                "ood_accessed": False,
            },
        )
        self.losses.extend([list(value) for value in values])
        self._write_progress("RUNNING")

    def complete(self) -> dict[str, Any]:
        if len(self.losses) != len(self.specs):
            raise RuntimeError("structural checkpoint is incomplete")
        self._write_progress("PASS")
        return {
            "status": "PASS",
            "schema": "PRISM_V211_K_STRUCTURAL_CHECKPOINT_V1",
            "fingerprint": self.fingerprint,
            "candidate_count": len(self.specs),
            "reused_candidate_count": self.reused,
            "checkpoint_path": str(self.root),
            "test_accessed": False,
            "ood_accessed": False,
        }

    def _write_progress(self, status: str) -> None:
        _atomic_write_json(
            self.progress_path,
            {
                "status": status,
                "fingerprint": self.fingerprint,
                "completed_candidates": len(self.losses),
                "total_candidates": len(self.specs),
                "reused_candidate_count": self.reused,
                "test_accessed": False,
                "ood_accessed": False,
            },
        )


@dataclass
class _PreparedKFold:
    fit_values: np.ndarray
    fit_target: np.ndarray
    evaluation_values: np.ndarray
    evaluation_target: np.ndarray
    fit_rows: int
    _prepared: dict[int, tuple[PreparedContractFit, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def for_width(self, requested_m_x: int) -> tuple[PreparedContractFit, np.ndarray]:
        width = int(requested_m_x)
        with self._lock:
            cached = self._prepared.get(width)
            if cached is not None:
                return cached
            prepared = prepare_contract_fit(self.fit_values, self.fit_target, width)
            evaluation_design = tensor_design(self.evaluation_values, prepared.basis)
            value = (prepared, evaluation_design)
            self._prepared[width] = value
            return value


def _evaluate_prepared_candidate(
    records: Sequence[_PreparedKFold],
    family: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    config: Mapping[str, Any],
) -> list[float]:
    losses: list[float] = []
    requested = 1 if family == "LINEAR_DISTRIBUTED_LAG" else int(m_x)
    for record in records:
        prepared, evaluation_design = record.for_width(requested)
        contract = fit_prepared_contract(
            prepared,
            family,
            int(m_x),
            lambdas,
            **_als_kwargs(dict(config)),
        )
        if not _candidate_valid(contract, record.fit_rows, dict(config)):
            losses.append(float("inf"))
            continue
        prediction = predict_contract_from_design(evaluation_design, contract)
        losses.append(mse(record.evaluation_target, prediction))
    return losses


def oof_replay_audit(
    selection_fold_losses: list[float],
    materialized_fold_losses: list[float],
) -> dict[str, Any]:
    selection = np.asarray(selection_fold_losses, dtype=np.float64)
    materialized = np.asarray(materialized_fold_losses, dtype=np.float64)
    if selection.shape != materialized.shape or selection.ndim != 1:
        raise ValueError("OOF replay loss vectors are incompatible")
    difference = materialized - selection
    return {
        "status": "RECORDED_NONSELECTING_REPLAY_AUDIT",
        "candidate_unchanged": True,
        "selection_fold_losses": selection.tolist(),
        "materialized_fold_losses": materialized.tolist(),
        "maximum_absolute_loss_difference": float(
            np.max(np.abs(difference), initial=0.0)
        ),
        "strict_1e_12_replay_match": bool(
            np.allclose(materialized, selection, rtol=1e-12, atol=1e-15)
        ),
        "selection_use": False,
    }


def select_smallest_stable_full_and_folds(
    candidates: Sequence[float],
    fit_full: Callable[[float], Mapping[str, Any]],
    fit_folds: Callable[[float], list[dict[str, Any]]],
    *,
    valid_full: Callable[[Mapping[str, Any]], bool],
    valid_fold: Callable[[dict[str, Any]], bool],
    parallel_workers: int = 1,
) -> tuple[float, Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the first ridge whose full refit and every inner fold certify."""
    def certificate(contract: Mapping[str, Any]) -> Mapping[str, Any]:
        return contract.get(
            "numerical_certificate", contract.get("certificate", {})
        )

    def evaluate_candidate(candidate: float) -> tuple[Mapping[str, Any], bool, list[dict[str, Any]]]:
        full_contract = fit_full(float(candidate))
        full_pass = bool(valid_full(full_contract))
        fold_payloads = fit_folds(float(candidate)) if full_pass else []
        return full_contract, full_pass, fold_payloads

    registered = [float(candidate) for candidate in candidates]
    batch_size = max(1, int(parallel_workers))
    audits = []
    for start in range(0, len(registered), batch_size):
        batch = registered[start : start + batch_size]
        evaluations = _ordered_parallel_map(
            evaluate_candidate,
            [(candidate,) for candidate in batch],
            batch_size,
        )
        for candidate, (full_contract, full_pass, fold_payloads) in zip(
            batch, evaluations, strict=True
        ):
            fold_passes = [bool(valid_fold(payload)) for payload in fold_payloads]
            passed = bool(full_pass and fold_passes and all(fold_passes))
            audits.append(
                {
                    "candidate": float(candidate),
                    "pass": passed,
                    "full_refit_pass": full_pass,
                    "all_inner_folds_pass": bool(
                        fold_passes and all(fold_passes)
                    ),
                    "full_refit_certificate": certificate(full_contract),
                    "inner_fold_certificates": [
                        {
                            "fold": int(payload["fold"]),
                            "pass": fold_pass,
                            "certificate": certificate(payload["contract"]),
                        }
                        for payload, fold_pass in zip(
                            fold_payloads, fold_passes, strict=True
                        )
                    ],
                }
            )
            if passed:
                return float(candidate), full_contract, fold_payloads, audits
    # Keep a compact rejection trace in the exception.  The normal successful
    # path is unchanged; this makes retained solver failures diagnosable
    # without rerunning the entire structural search.
    rejection_summary = []
    for audit in audits:
        full_certificate = audit.get("full_refit_certificate", {})
        inner = audit.get("inner_fold_certificates", [])
        rejection_summary.append({
            "candidate": audit["candidate"],
            "full_refit_pass": audit["full_refit_pass"],
            "full_status": full_certificate.get("status"),
            "full_iterations": full_certificate.get("iterations"),
            "full_increases": full_certificate.get("consecutive_increases", full_certificate.get("increase_count")),
            "full_fixed_support_status": (full_certificate.get("fixed_support_refit_certificate") or {}).get("status"),
            "inner": [
                {
                    "fold": item.get("fold"),
                    "pass": item.get("pass"),
                    "status": (item.get("certificate") or {}).get("status"),
                    "iterations": (item.get("certificate") or {}).get("iterations"),
                    "fixed_support_status": ((item.get("certificate") or {}).get("fixed_support_refit_certificate") or {}).get("status"),
                }
                for item in inner
            ],
        })
    raise RuntimeError(
        "no registered ridge candidate passed full-refit and all-inner-fold certificates; "
        + json.dumps({"ridge_rejections": rejection_summary}, sort_keys=True)
    )


def _profile_complexity(profile: tuple[int, int]) -> tuple[int, int]:
    return int(profile[1]), -int(profile[0])


def _structural_complexity(candidate: Any) -> tuple[Any, ...]:
    if candidate == EXACT_ZERO:
        return (0,)
    profile, m_tau, family, m_x = candidate
    width = int(m_tau) * (1 if family == "LINEAR_DISTRIBUTED_LAG" else int(m_x))
    return (
        1,
        FAMILY_ORDER.index(str(family)),
        width,
        *_profile_complexity(tuple(profile)),
        int(m_tau),
        int(m_x),
    )


def evaluate_candidate(
    accessor: BaseAccessor,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    family: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    config: dict[str, Any],
    scoring_history_steps: int | None = None,
) -> list[float]:
    """Fit on candidate-native rows and score on one local common support."""
    history = int(profile[1])
    scoring_history = history if scoring_history_steps is None else int(
        scoring_history_steps
    )
    records = registered_fold_native_masks(
        train,
        folds,
        fit_history_steps=history,
        scoring_history_steps=scoring_history,
        fit_cap=int(config["row_caps"]["single_channel_k_fit"]),
        evaluation_cap=int(config["row_caps"]["validation_selection_per_fold"]),
    )
    losses = []
    for record in records:
        fit = record["fit"]
        evaluation = record["evaluation"]
        if family == EXACT_ZERO:
            prediction = np.zeros(len(evaluation), dtype=np.float64)
        else:
            fit_values, _ = profile_values(accessor, fit, channel, profile, m_tau)
            evaluation_values, _ = profile_values(
                accessor, evaluation, channel, profile, m_tau
            )
            contract = fit_contract(
                fit_values,
                fit["y_true"].to_numpy(dtype=np.float64),
                family,
                m_x,
                lambdas,
                **_als_kwargs(config),
            )
            if not _candidate_valid(contract, len(fit), config):
                losses.append(float("inf"))
                continue
            prediction = predict_contract(evaluation_values, contract)
        losses.append(
            mse(evaluation["y_true"].to_numpy(dtype=np.float64), prediction)
        )
    return losses


def _prepare_structural_records(
    accessor: BaseAccessor,
    records: Sequence[Mapping[str, Any]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
) -> list[_PreparedKFold]:
    prepared: list[_PreparedKFold] = []
    for record in records:
        fit = record["fit"]
        evaluation = record["evaluation"]
        fit_values, _ = profile_values(accessor, fit, channel, profile, m_tau)
        evaluation_values, _ = profile_values(
            accessor, evaluation, channel, profile, m_tau
        )
        prepared.append(
            _PreparedKFold(
                fit_values=fit_values,
                fit_target=fit["y_true"].to_numpy(dtype=np.float64),
                evaluation_values=evaluation_values,
                evaluation_target=evaluation["y_true"].to_numpy(dtype=np.float64),
                fit_rows=len(fit),
            )
        )
    return prepared


def _run_structural_grid(
    *,
    accessor: BaseAccessor,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profiles: Sequence[tuple[int, int]],
    local_comparison_history: int,
    m_tau_values: Sequence[int],
    m_x_values: Sequence[int],
    lambda_tau_values: Sequence[float],
    lambda_x_values: Sequence[float],
    pilot_lambda_0: float,
    v2: dict[str, Any],
    inner_workers: int,
    destination: Path,
    checkpoint_context: Mapping[str, Any],
) -> tuple[dict[Any, list[float]], dict[str, Any]]:
    specs = _structural_specs(
        profiles,
        m_tau_values,
        lambda_tau_values,
        m_x_values,
        lambda_x_values,
    )
    checkpoint = _StructuralCheckpoint(
        destination / "STRUCTURAL_CHECKPOINTS", specs, checkpoint_context
    )
    batch_size = max(int(inner_workers), _k_checkpoint_batch())
    index = len(checkpoint.losses)
    executor = (
        ThreadPoolExecutor(max_workers=int(inner_workers))
        if int(inner_workers) > 1
        else None
    )
    try:
        while index < len(specs):
            profile = tuple(specs[index][0])
            m_tau = int(specs[index][1])
            group_stop = index
            while group_stop < len(specs):
                candidate = specs[group_stop]
                if tuple(candidate[0]) != profile or int(candidate[1]) != m_tau:
                    break
                group_stop += 1
            native_records = registered_fold_native_masks(
                train,
                folds,
                fit_history_steps=int(profile[1]),
                scoring_history_steps=int(local_comparison_history),
                fit_cap=int(v2["row_caps"]["single_channel_k_fit"]),
                evaluation_cap=int(v2["row_caps"]["validation_selection_per_fold"]),
            )
            prepared_records = _prepare_structural_records(
                accessor, native_records, channel, profile, m_tau
            )
            while index < group_stop:
                stop = min(group_stop, index + batch_size)
                arguments = []
                for spec in specs[index:stop]:
                    _, _, family, m_x, lambda_tau, lambda_x = spec
                    arguments.append(
                        (
                            prepared_records,
                            str(family),
                            int(m_x),
                            (
                                float(pilot_lambda_0),
                                float(lambda_tau),
                                float(lambda_x),
                            ),
                            v2,
                        )
                    )
                values = _ordered_parallel_map(
                    _evaluate_prepared_candidate,
                    arguments,
                    inner_workers,
                    executor=executor,
                )
                checkpoint.append(index, values)
                index = stop
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    audit = checkpoint.complete()
    audit["checkpoint_path"] = str(
        (destination / "STRUCTURAL_CHECKPOINTS").relative_to(destination)
    )
    return dict(zip(specs, checkpoint.losses, strict=True)), audit


def _smoothness_selection(
    *,
    accessor: BaseAccessor,
    train: Any,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    family: str,
    m_x: int,
    v2: dict[str, Any],
    minimum_folds: int,
    parallel_workers: int = 1,
) -> tuple[float, float, dict[str, Any]]:
    lambda_0 = 0.0
    lambda_tau = float(v2["K_module"]["penalties"]["pilot"]["lambda_tau"])
    lambda_x = float(v2["K_module"]["penalties"]["pilot"]["lambda_x"])
    audit: dict[str, Any] = {}
    for name in ("lambda_tau", "lambda_x"):
        if name == "lambda_x" and family == "LINEAR_DISTRIBUTED_LAG":
            lambda_x = 0.0
            audit[name] = {"status": "NOT_APPLICABLE_LINEAR_FAMILY", "selected": 0.0}
            continue
        values = [float(value) for value in v2["K_module"]["penalties"][name]]
        jobs = []
        for value in values:
            candidate_tau = value if name == "lambda_tau" else lambda_tau
            candidate_x = value if name == "lambda_x" else lambda_x
            jobs.append(
                (
                    accessor,
                    train,
                    folds,
                    channel,
                    profile,
                    m_tau,
                    family,
                    m_x,
                    (lambda_0, candidate_tau, candidate_x),
                    v2,
                    int(profile[1]),
                )
            )
        scan = dict(
            zip(
                values,
                _ordered_parallel_map(evaluate_candidate, jobs, parallel_workers),
                strict=True,
            )
        )
        selected = tune_best_nonzero(
            scan, minimum_usable_folds=minimum_folds
        )
        if name == "lambda_tau":
            lambda_tau = float(selected)
        else:
            lambda_x = float(selected)
        audit[name] = {
            "selection": {
                "selected": float(selected),
                "selection_rule": "MINIMUM_EMPIRICAL_RISK_IN_NONZERO_FAMILY",
                "one_se_used": False,
                "complexity_preference_used": False,
            },
            "fold_losses": {str(key): value for key, value in scan.items()},
        }
    return lambda_tau, lambda_x, audit


def run_k_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
    protocol: str = "sru",
) -> dict[str, Any]:
    started = time.time()
    destination = (
        output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy / channel
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v211, v21, v2 = load_v211_configs(project, protocol=protocol)
        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        train = load_native_samples(
            shared, view, "train", columns=CHANNEL_SAMPLE_COLUMNS
        )
        validation = load_native_samples(
            shared, view, "validation", columns=CHANNEL_SAMPLE_COLUMNS
        )
        accessor = BaseAccessor(shared, view.head.dataset, "validation", [channel])
        accessor.warm_prefixes([channel])
        inner_workers = _k_inner_workers()
        folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
        profiles = channel_profiles(view, channel, v2)
        profile_comparison_history = max(int(profile[1]) for profile in profiles)
        pilot = v2["K_module"]["penalties"]["pilot"]
        pilot_lambdas = (
            float(pilot["lambda_0"]),
            float(pilot["lambda_tau"]),
            float(pilot["lambda_x"]),
        )
        pilot_m_tau = 8
        profile_jobs = [
            (
                accessor,
                train,
                folds,
                channel,
                profile,
                pilot_m_tau,
                "LINEAR_DISTRIBUTED_LAG",
                1,
                pilot_lambdas,
                v2,
                profile_comparison_history,
            )
            for profile in profiles
        ]
        profile_losses = dict(
            zip(
                profiles,
                _ordered_parallel_map(evaluate_candidate, profile_jobs, inner_workers),
                strict=True,
            )
        )
        best_profile = tuple(
            tune_best_nonzero(
                profile_losses, minimum_usable_folds=minimum_folds
            )
        )
        profile_selection_json = {
            "selected": str(best_profile),
            "selection_rule": "MINIMUM_EMPIRICAL_RISK_IN_NONZERO_FAMILY",
            "one_se_used": False,
            "regret_guard_used": False,
            "complexity_preference_used": False,
        }
        # Keep all profiles in H_K^+ so the outer-fold winner is tuned without
        # seeing its own held-out loss.
        retained_profiles = [tuple(value) for value in profiles]
        local_comparison_history = max(
            int(profile[1]) for profile in retained_profiles
        )
        local_scoring_records = registered_fold_native_masks(
            train,
            folds,
            fit_history_steps=local_comparison_history,
            scoring_history_steps=local_comparison_history,
            fit_cap=int(v2["row_caps"]["single_channel_k_fit"]),
            evaluation_cap=int(v2["row_caps"]["validation_selection_per_fold"]),
        )
        activation_profile_losses = dict(
            zip(
                retained_profiles,
                _ordered_parallel_map(
                    evaluate_candidate,
                    [
                        (
                            accessor,
                            train,
                            folds,
                            channel,
                            profile,
                            pilot_m_tau,
                            "LINEAR_DISTRIBUTED_LAG",
                            1,
                            pilot_lambdas,
                            v2,
                            local_comparison_history,
                        )
                        for profile in retained_profiles
                    ],
                    inner_workers,
                ),
                strict=True,
            )
        )

        zero_losses = []
        for record in local_scoring_records:
            subset = record["evaluation"]
            target = subset["y_true"].to_numpy(dtype=np.float64)
            zero_losses.append(float(np.mean(target * target, dtype=np.float64)))

        linear_activation_losses: dict[Any, list[float]] = {}
        for profile in retained_profiles:
            linear_activation_losses[(profile, "LINEAR_DISTRIBUTED_LAG")] = (
                activation_profile_losses[profile]
            )
        activation = strict_nested_oof_select(
            linear_activation_losses,
            zero_losses,
            identity=EXACT_ZERO,
            fold_weights=[len(record["evaluation"]) for record in local_scoring_records],
            minimum_inner_folds=max(2, minimum_folds - 1),
            minimum_outer_folds=minimum_folds,
        )

        lambda_tau_values = [
            float(value) for value in v2["K_module"]["penalties"]["lambda_tau"]
        ]
        lambda_x_values = [
            float(value) for value in v2["K_module"]["penalties"]["lambda_x"]
        ]
        structural_losses, structural_execution_audit = _run_structural_grid(
            accessor=accessor,
            train=train,
            folds=folds,
            channel=channel,
            profiles=retained_profiles,
            local_comparison_history=local_comparison_history,
            m_tau_values=[int(value) for value in v21["K_C"]["m_tau"]],
            m_x_values=[int(value) for value in v21["K_C"]["m_x"]],
            lambda_tau_values=lambda_tau_values,
            lambda_x_values=lambda_x_values,
            pilot_lambda_0=float(pilot["lambda_0"]),
            v2=v2,
            inner_workers=inner_workers,
            destination=destination,
            checkpoint_context={
                "protocol": protocol,
                "target_head": view.head.head_id,
                "proxy_policy": view.proxy_policy,
                "channel": channel,
                "train_support_hash": support_id_hash(train),
                "inner_folds": int(v21["selection"]["inner_folds"]),
                "minimum_usable_folds": minimum_folds,
                "local_comparison_history": local_comparison_history,
                "row_caps": dict(v2["row_caps"]),
                "als": dict(v2["numerical_certification"]["als"]),
                "als_initialization_seeds": list(
                    v2["randomness"]["als_initialization_seeds"]
                ),
                "numerical_engine": K_NUMERICAL_ENGINE,
            },
        )
        structural_selection = strict_nested_oof_select(
            structural_losses,
            zero_losses,
            identity=EXACT_ZERO,
            fold_weights=[len(record["evaluation"]) for record in local_scoring_records],
            minimum_inner_folds=max(2, minimum_folds - 1),
            minimum_outer_folds=minimum_folds,
        )
        tuned_nonzero = structural_selection.tuned_nonzero_candidate
        (
            selected_profile,
            selected_m_tau,
            tuned_family,
            selected_m_x,
            lambda_tau,
            lambda_x,
        ) = tuned_nonzero
        selected_profile = tuple(selected_profile)
        selected_m_tau = int(selected_m_tau)
        selected_m_x = int(selected_m_x)
        selected_family = (
            str(tuned_family) if structural_selection.active else EXACT_ZERO
        )
        lambda_tau = float(lambda_tau)
        lambda_x = float(lambda_x)
        smoothness_audit = {
            "status": "INCLUDED_IN_NONZERO_NESTED_OOF_TUNING",
            "one_se_used": False,
        }

        selected_history = int(selected_profile[1])
        selected_support_history = (
            local_comparison_history
            if selected_family == EXACT_ZERO
            else selected_history
        )
        final_train_native = apply_native_support(train, selected_support_history)
        final_train = _cap(
            final_train_native, int(v2["row_caps"]["single_channel_k_fit"])
        )
        selected_validation = apply_native_support(
            validation, selected_support_history
        )
        selected_fold_records = registered_fold_native_masks(
            train,
            folds,
            fit_history_steps=selected_support_history,
            scoring_history_steps=selected_support_history,
            fit_cap=int(v2["row_caps"]["single_channel_k_fit"]),
            evaluation_cap=int(v2["row_caps"]["validation_selection_per_fold"]),
        )
        train_values, intervals = profile_values(
            accessor, final_train, channel, selected_profile, selected_m_tau
        )
        validation_values, _ = profile_values(
            accessor, selected_validation, channel, selected_profile, selected_m_tau
        )
        ridge_audit: list[dict[str, Any]] = []
        certified_fold_payloads: list[dict[str, Any]] = []
        if selected_family == EXACT_ZERO:
            selected_lambdas = pilot_lambdas
            refit_contract = fit_contract(
                train_values,
                final_train["y_true"].to_numpy(dtype=np.float64),
                EXACT_ZERO,
                selected_m_x,
                selected_lambdas,
                **_als_kwargs(v2),
            )
        else:
            ridge_values = sorted(
                {float(value) for value in v2["K_module"]["penalties"]["lambda_0"]}
            )

            def fit_ridge(lambda_0: float) -> dict[str, Any]:
                return fit_contract(
                    train_values,
                    final_train["y_true"].to_numpy(dtype=np.float64),
                    selected_family,
                    selected_m_x,
                    (float(lambda_0), lambda_tau, lambda_x),
                    **_als_kwargs(v2),
                )

            def fit_ridge_folds(lambda_0: float) -> list[dict[str, Any]]:
                payloads = []
                for fold, record in enumerate(selected_fold_records):
                    fit = record["fit"]
                    evaluation = record["evaluation"]
                    fit_values, _ = profile_values(
                        accessor, fit, channel, selected_profile, selected_m_tau
                    )
                    evaluation_values, _ = profile_values(
                        accessor,
                        evaluation,
                        channel,
                        selected_profile,
                        selected_m_tau,
                    )
                    fold_contract = fit_contract(
                        fit_values,
                        fit["y_true"].to_numpy(dtype=np.float64),
                        selected_family,
                        selected_m_x,
                        (float(lambda_0), lambda_tau, lambda_x),
                        **_als_kwargs(v2),
                    )
                    fold_certificate_pass = _candidate_valid(
                        fold_contract, len(fit), v2
                    )
                    if fold_certificate_pass:
                        fold_prediction = predict_contract(
                            evaluation_values, fold_contract
                        )
                        fold_loss = mse(
                            evaluation["y_true"].to_numpy(dtype=np.float64),
                            fold_prediction,
                        )
                    else:
                        fold_prediction = np.full(
                            len(evaluation), np.nan, dtype=np.float64
                        )
                        fold_loss = float("inf")
                    payloads.append(
                        {
                            "fold": fold,
                            "fit_rows": len(fit),
                            "evaluation": evaluation,
                            "contract": fold_contract,
                            "prediction": fold_prediction,
                            "loss": fold_loss,
                        }
                    )
                return payloads

            (
                lambda_0,
                refit_contract,
                certified_fold_payloads,
                ridge_audit,
            ) = select_smallest_stable_full_and_folds(
                ridge_values,
                fit_ridge,
                fit_ridge_folds,
                valid_full=lambda contract: _candidate_valid(
                    dict(contract), len(final_train), v2
                ),
                valid_fold=lambda payload: (
                    _candidate_valid(
                        dict(payload["contract"]), int(payload["fit_rows"]), v2
                    )
                    and np.isfinite(
                        np.asarray(payload["prediction"], dtype=np.float64)
                    ).all()
                    and np.isfinite(float(payload["loss"]))
                ),
                parallel_workers=inner_workers,
            )
            selected_lambdas = (float(lambda_0), lambda_tau, lambda_x)

        prediction = predict_contract(validation_values, dict(refit_contract))
        frame = selected_validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = f"PRISM_V2_1_1_K_{channel}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        active = selected_family != EXACT_ZERO and _candidate_valid(
            dict(refit_contract), len(final_train), v2
        )
        final_fold_losses = list(structural_selection.final_selected_fold_losses)
        oof_records = local_scoring_records
        if selected_family == EXACT_ZERO:
            certified_fold_predictions = [
                np.zeros(len(record["evaluation"]), dtype=np.float64)
                for record in oof_records
            ]
        else:
            outer_winners = dict(
                zip(
                    structural_selection.outer_fold_indices,
                    structural_selection.outer_selected_nonzero_candidates,
                    strict=True,
                )
            )
            certified_fold_predictions = []
            for fold, record in enumerate(oof_records):
                (
                    fold_profile,
                    fold_m_tau,
                    fold_family,
                    fold_m_x,
                    fold_lambda_tau,
                    fold_lambda_x,
                ) = outer_winners[fold]
                fold_fit_values, _ = profile_values(
                    accessor,
                    record["fit"],
                    channel,
                    tuple(fold_profile),
                    int(fold_m_tau),
                )
                fold_evaluation_values, _ = profile_values(
                    accessor,
                    record["evaluation"],
                    channel,
                    tuple(fold_profile),
                    int(fold_m_tau),
                )
                fold_contract = fit_contract(
                    fold_fit_values,
                    record["fit"]["y_true"].to_numpy(dtype=np.float64),
                    str(fold_family),
                    int(fold_m_x),
                    (
                        float(pilot["lambda_0"]),
                        float(fold_lambda_tau),
                        float(fold_lambda_x),
                    ),
                    **_als_kwargs(v2),
                )
                certified_fold_predictions.append(
                    predict_contract(fold_evaluation_values, fold_contract)
                )
        oof_frames = []
        oof_losses = []
        for fold, (record, fold_prediction) in enumerate(
            zip(oof_records, certified_fold_predictions, strict=True)
        ):
            evaluation = record["evaluation"]
            oof_loss = mse(
                evaluation["y_true"].to_numpy(dtype=np.float64), fold_prediction
            )
            oof_losses.append(oof_loss)
            oof_frame = evaluation[
                ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
            ].copy()
            oof_frame["y_pred"] = fold_prediction
            oof_frame["oof_fold"] = fold
            oof_frames.append(oof_frame)
        replay_audit = oof_replay_audit(final_fold_losses, oof_losses)
        oof_path = destination / "SELECTED_OOF.parquet"
        pd.concat(oof_frames, ignore_index=True).to_parquet(
            oof_path, index=False, compression="zstd"
        )
        result = {
            "status": "PASS",
            "stage": "E2R_K",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "selected_profile": list(selected_profile),
            "support_contract": SUPPORT_CONTRACT,
            "selected_profile_history_steps": selected_history,
            "selected_scoring_history_steps": selected_support_history,
            "retained_profiles": [list(value) for value in retained_profiles],
            "selected_intervals": [list(value) for value in intervals],
            "selected_family": selected_family,
            "selected_m_tau": selected_m_tau,
            "selected_m_x": selected_m_x,
            "selected_lambdas": list(selected_lambdas),
            "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
            "active": active,
            "profile_selection": profile_selection_json,
            "profile_fold_losses": {
                str(key): value for key, value in profile_losses.items()
            },
            "linear_activation_selection": activation.to_json(),
            "linear_activation_profile_fold_losses": {
                str(key): value for key, value in activation_profile_losses.items()
            },
            "structural_selection": structural_selection.to_json(),
            "routing_status": structural_selection.routing_status,
            "structural_fold_losses": {
                str(key): value for key, value in structural_losses.items()
            },
            "structural_execution_audit": structural_execution_audit,
            "smoothness_audit": smoothness_audit,
            "minimal_stabilizing_ridge_audit": ridge_audit,
            "contract": refit_contract,
            "final_selected_candidate": str(
                structural_selection.final_selected_candidate
            ),
            "final_selected_fold_losses": list(oof_losses),
            "selection_fold_losses_before_oof_replay": list(final_fold_losses),
            "oof_replay_audit": replay_audit,
            "final_selected_prediction_path": str(
                prediction_path.relative_to(output)
            ),
            "final_selected_contract": refit_contract,
            "final_prediction_loss": mse(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "oof_prediction_path": str(oof_path.relative_to(output)),
            "oof_prediction_sha256": sha256_file(oof_path),
            "oof_prediction_fold_losses": oof_losses,
            "native_fit_rows_by_fold": [
                len(record["fit_native"]) for record in oof_records
            ],
            "native_fit_support_hash_by_fold": [
                support_id_hash(record["fit_native"])
                for record in oof_records
            ],
            "local_scoring_rows_by_fold": [
                len(record["evaluation"]) for record in oof_records
            ],
            "local_scoring_support_hash_by_fold": [
                support_id_hash(record["evaluation"])
                for record in oof_records
            ],
            "selected_native_train_rows": len(final_train_native),
            "selected_native_validation_rows": len(selected_validation),
            "selected_native_support_hash": {
                "train": support_id_hash(final_train_native),
                "validation": support_id_hash(selected_validation),
            },
            "selected_native_support_audit": {
                "train": support_audit(final_train_native),
                "validation": support_audit(selected_validation),
            },
            "row_cap_applied_after_native_mask": True,
            "cross_channel_loss_comparable": False,
            "historical_global_lmax_used": False,
            "exact_zero_scoring_support_hash": [
                support_id_hash(record["evaluation"])
                for record in local_scoring_records
            ],
            "nonzero_scoring_support_hash": [
                support_id_hash(record["evaluation"])
                for record in local_scoring_records
            ],
            "row_cap_audit": {
                "cap_name": "single_channel_k_fit",
                "cap": int(v2["row_caps"]["single_channel_k_fit"]),
                "fit_rows": len(final_train),
                "validation_rows": len(selected_validation),
                "fit_source": "train_only",
                "row_cap_applied_after_native_mask": True,
            },
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E2R_K",
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def load_active_channels(output: Path, view: ViewSpec) -> list[dict[str, Any]]:
    root = output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy
    results = []
    for path in sorted(root.glob("*/RESULT.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "PASS" and value.get("active"):
            results.append(value)
    return results


def run_e2r_k(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    jobs = []
    for view in sru_input_views(shared):
        for channel in input_columns(shared, view.head.task_id, view.proxy_policy):
            jobs.append((shared, project, output, view, channel))
    results = run_parallel(
        run_k_channel,
        jobs,
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E2R_K",
    )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E2R_K",
        "jobs": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "active": sum(bool(item.get("active")) for item in results),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "K" / "SUMMARY.json", summary)
    return summary
