#!/usr/bin/env python3
"""Pure-data audits and partial protocol freeze for PB1 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from ar_raphu.datasets.loaders import load_cascaded_tanks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_record_sha256(
    *, reference: np.ndarray, x: np.ndarray, y: np.ndarray, fs: float
) -> str:
    digest = hashlib.sha256(b"PB1_WHPN_RECORD_V1\0")
    for name, values in (
        ("reference", reference),
        ("input", x),
        ("output", y),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    digest.update(np.asarray([fs], dtype="<f8").tobytes())
    return digest.hexdigest()


def _longest_exact_run(values: np.ndarray, target: float) -> int:
    mask = np.asarray(values) == target
    best = current = 0
    for matched in mask:
        current = current + 1 if matched else 0
        best = max(best, current)
    return int(best)


def _exact_run_count(values: np.ndarray, target: float) -> int:
    mask = np.asarray(values) == target
    if not len(mask):
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.count_nonzero(starts))


def _psd_profile(values: np.ndarray, fs: float, *, nperseg: int) -> dict[str, Any]:
    frequencies, power = welch(
        np.asarray(values, dtype=np.float64),
        fs=fs,
        nperseg=min(nperseg, len(values)),
        detrend="linear",
        scaling="density",
    )
    total = float(np.trapezoid(power, frequencies))
    normalized = power / max(total, np.finfo(float).tiny)
    weighted = normalized * np.gradient(frequencies)
    cumulative = np.cumsum(weighted)
    cumulative /= cumulative[-1]
    lower = float(frequencies[np.searchsorted(cumulative, 0.01)])
    upper = float(frequencies[np.searchsorted(cumulative, 0.99)])
    centroid = float(np.sum(frequencies * weighted) / np.sum(weighted))
    signature = hashlib.sha256(
        np.ascontiguousarray(
            np.column_stack((frequencies, normalized)), dtype="<f8"
        ).tobytes()
    ).hexdigest()
    return {
        "nperseg": int(min(nperseg, len(values))),
        "power_01_99_hz": [lower, upper],
        "spectral_centroid_hz": centroid,
        "normalized_psd_sha256": signature,
    }


def _series_profile(values: np.ndarray, fs: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    normalized_time = np.linspace(0.0, 1.0, len(array), dtype=np.float64)
    slope = float(np.polyfit(normalized_time, array, deg=1)[0])
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "min": minimum,
        "max": maximum,
        "trend_per_record": slope,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(array))),
        "exact_min_count": int(np.count_nonzero(array == minimum)),
        "exact_max_count": int(np.count_nonzero(array == maximum)),
        "longest_exact_min_run": _longest_exact_run(array, minimum),
        "longest_exact_max_run": _longest_exact_run(array, maximum),
        "saturation_status": "NOT_DETERMINABLE_WITHOUT_SENSOR_LIMITS",
        "psd": _psd_profile(array, fs, nperseg=1024),
    }


def audit_whpn(raw_root: Path) -> dict[str, Any]:
    path = (
        raw_root
        / "WHPN"
        / "WienerHammersteinFiles"
        / "WH_EstimationExample.mat"
    )
    source = loadmat(path, squeeze_me=True, struct_as_record=False)["dataMeas"]
    reference = np.asarray(source.r, dtype=np.float64)
    x = np.asarray(source.u, dtype=np.float64)
    y = np.asarray(source.y, dtype=np.float64)
    fs = float(np.asarray(source.fs).squeeze())
    if reference.shape != x.shape or x.shape != y.shape:
        raise ValueError("WHPN r/u/y shapes differ.")
    records = []
    for index in range(x.shape[1]):
        record_id = (
            "WH_EstimationExample.mat::dataMeas"
            f"[realization_zero_based={index}]"
        )
        records.append(
            {
                "record_id": record_id,
                "realization_zero_based": index,
                "canonical_sha256": _canonical_record_sha256(
                    reference=reference[:, index],
                    x=x[:, index],
                    y=y[:, index],
                    fs=fs,
                ),
                "sample_count": int(x.shape[0]),
                "sampling_rate_hz": fs,
                "input": _series_profile(x[:, index], fs),
                "output": _series_profile(y[:, index], fs),
                "measurement_campaign_id": "NOT_YET_AVAILABLE",
            }
        )
    validation_indices = (8, 9)
    consistent_shape = len({row["sample_count"] for row in records}) == 1
    consistent_fs = len({row["sampling_rate_hz"] for row in records}) == 1
    return {
        "schema_version": 6,
        "dataset": "whpn",
        "audit_scope": "ESTIMATION_RECORDS_ONLY_NO_MODEL_RESULTS",
        "source_file": str(path),
        "source_file_sha256": _sha256_file(path),
        "record_count": len(records),
        "records": records,
        "validation_record_ids": [
            records[index]["record_id"] for index in validation_indices
        ],
        "validation_record_sha256": [
            records[index]["canonical_sha256"] for index in validation_indices
        ],
        "protocol_consistency": {
            "same_mat_file": True,
            "sample_count_consistent": consistent_shape,
            "sampling_rate_consistent": consistent_fs,
            "measurement_campaign_id": "NOT_YET_AVAILABLE",
        },
        "validation_interpretation": (
            "WITHIN_FILE_REALIZATION_HOLDOUT_CAMPAIGN_ID_UNAVAILABLE"
        ),
        "alignment": {
            "primary": "RAW_UNSHIFTED",
            "input_shift_samples": 0,
            "appendix_sensitivity": [-1, 0, 1],
            "sensitivity_scope": "TRAIN_VALIDATION_ONLY",
            "may_change_primary": False,
            "abnormal_difference_flag": "ALIGNMENT_AUDIT_REQUIRED",
        },
        "status": (
            "COMPLETED"
            if consistent_shape and consistent_fs
            else "FAILED"
        ),
    }


def _split_profile(values: np.ndarray, fs: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    delta = np.diff(array)
    return {
        "n": int(len(array)),
        "range": [float(array.min()), float(array.max())],
        "mean": float(array.mean()),
        "std": float(array.std()),
        "rms": float(np.sqrt(np.mean(array * array))),
        "slew_abs_quantiles": {
            "q50": float(np.quantile(np.abs(delta), 0.50)),
            "q90": float(np.quantile(np.abs(delta), 0.90)),
            "q99": float(np.quantile(np.abs(delta), 0.99)),
            "max": float(np.max(np.abs(delta))),
        },
        "psd": _psd_profile(array, fs, nperseg=128),
    }


def _psd_overlap(train: np.ndarray, validation: np.ndarray, fs: float) -> float:
    frequency_train, power_train = welch(
        train, fs=fs, nperseg=128, detrend="linear", scaling="density"
    )
    frequency_validation, power_validation = welch(
        validation, fs=fs, nperseg=128, detrend="linear", scaling="density"
    )
    np.testing.assert_allclose(frequency_train, frequency_validation)
    train_mass = power_train / max(
        np.trapezoid(power_train, frequency_train), np.finfo(float).tiny
    )
    validation_mass = power_validation / max(
        np.trapezoid(power_validation, frequency_validation),
        np.finfo(float).tiny,
    )
    return float(
        np.trapezoid(
            np.minimum(train_mass, validation_mass), frequency_train
        )
    )


def audit_tanks(raw_root: Path) -> dict[str, Any]:
    dataset = load_cascaded_tanks(raw_root, include_test=False)
    n = dataset.n_time
    boundary = int(np.floor(0.80 * n))
    x = np.asarray(dataset.x[:, 0], dtype=np.float64)
    y = np.asarray(dataset.y[:, 0], dtype=np.float64)
    fs = 1.0 / float(dataset.metadata["sampling_time_by_sequence"]["estimation"])
    train_x, validation_x = x[:boundary], x[boundary:]
    train_y, validation_y = y[:boundary], y[boundary:]
    input_ood = (validation_x < train_x.min()) | (validation_x > train_x.max())
    output_ood = (validation_y < train_y.min()) | (
        validation_y > train_y.max()
    )
    empirical_ceiling = float(np.max(y))
    train_ceiling_count = int(np.count_nonzero(train_y == empirical_ceiling))
    validation_ceiling_count = int(
        np.count_nonzero(validation_y == empirical_ceiling)
    )
    return {
        "schema_version": 6,
        "dataset": "cascaded_tanks",
        "audit_scope": "ESTIMATION_RECORD_ONLY_NO_MODEL_RESULTS",
        "split": {
            "train_rows_zero_based": [0, boundary],
            "validation_rows_zero_based": [boundary, n],
            "boundary": boundary,
            "validation_fraction_realized": (n - boundary) / n,
        },
        "input": {
            "train": _split_profile(train_x, fs),
            "validation": _split_profile(validation_x, fs),
            "validation_ood_fraction_against_train_range": float(
                np.mean(input_ood)
            ),
            "normalized_psd_overlap": _psd_overlap(
                train_x, validation_x, fs
            ),
        },
        "output": {
            "train": _split_profile(train_y, fs),
            "validation": _split_profile(validation_y, fs),
            "validation_ood_fraction_against_train_range": float(
                np.mean(output_ood)
            ),
            "normalized_psd_overlap": _psd_overlap(
                train_y, validation_y, fs
            ),
        },
        "overflow_event_count": {
            "status": "BLOCKED_BY_MISSING_METADATA",
            "reason": "Official benchmark supplies no sample-level overflow flag or numeric threshold.",
        },
        "near_overflow_sample_count": {
            "status": "BLOCKED_BY_MISSING_METADATA",
            "reason": "Near-overflow margin cannot be defined without an overflow threshold.",
        },
        "empirical_output_ceiling_contact": {
            "value": empirical_ceiling,
            "interpretation": (
                "OBSERVED_EXACT_CEILING_CONTACT_NOT_AN_OFFICIAL_OVERFLOW_LABEL"
            ),
            "train": {
                "sample_count": train_ceiling_count,
                "fraction": train_ceiling_count / len(train_y),
                "contiguous_run_count": _exact_run_count(
                    train_y, empirical_ceiling
                ),
                "longest_run": _longest_exact_run(
                    train_y, empirical_ceiling
                ),
            },
            "validation": {
                "sample_count": validation_ceiling_count,
                "fraction": validation_ceiling_count / len(validation_y),
                "contiguous_run_count": _exact_run_count(
                    validation_y, empirical_ceiling
                ),
                "longest_run": _longest_exact_run(
                    validation_y, empirical_ceiling
                ),
            },
        },
        "official_test_accessed": False,
        "purge": {
            "required_formula": "G=L_star+h+b",
            "extra_mixing_gap_b": "NOT_YET_FROZEN",
            "windows_cross_boundary": False,
        },
        "status": "PENDING_SPLIT_ADEQUACY_AUDIT",
        "decision": (
            "AVAILABLE_DISTRIBUTION_CHECKS_COMPLETED_BUT_OVERFLOW_COVERAGE_UNRESOLVED"
        ),
    }


def build_freeze(whpn: dict[str, Any], tanks: dict[str, Any]) -> dict[str, Any]:
    pwh_validation = [
        f"ParWHData.mat::uEst[phase_zero_based={phase},amplitude_zero_based={amp}]"
        for phase in range(16, 20)
        for amp in range(5)
    ]
    return {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "freeze_status": "PARTIALLY_FROZEN",
        "private_cz_access": "FORBIDDEN",
        "selection_uses_official_test": False,
        "datasets": {
            "pwh": {
                "status": "FROZEN",
                "train": {
                    "phase_zero_based": [0, 15],
                    "amplitude_zero_based": [0, 4],
                    "record_count": 80,
                },
                "validation_record_ids": pwh_validation,
                "validation_record_count": 20,
                "periods_per_record": 2,
                "periods_are_atomic": True,
                "primary_confirmation_test": "uVal/yVal all five amplitude levels",
                "secondary_ood_test": "uValArr/yValArr increasing-amplitude record",
                "confirmation_test_access": "ONCE_AFTER_PROTOCOL_FREEZE",
            },
            "whpn": {
                "status": "FROZEN",
                "train_realization_zero_based": list(range(8)),
                "validation_record_ids": whpn["validation_record_ids"],
                "validation_record_sha256": whpn[
                    "validation_record_sha256"
                ],
                "validation_interpretation": whpn[
                    "validation_interpretation"
                ],
                "alignment": whpn["alignment"],
                "official_test_access": "ONCE_AFTER_PROTOCOL_FREEZE",
            },
            "cascaded_tanks": {
                "status": "PENDING_SPLIT_ADEQUACY_AUDIT",
                "candidate_split": tanks["split"],
                "blocked_metrics": [
                    "overflow_event_count",
                    "near_overflow_sample_count",
                ],
                "purge": tanks["purge"],
                "official_test_access": "LOCKED",
            },
            "silverbox": {
                "status": "BLOCKED_BY_MISSING_METADATA",
                "reason": "LICENSE_METADATA_UNRESOLVED",
                "loader": "COMPLETED",
                "checksum": "VERIFIED",
                "unit_tests": "ALLOWED",
                "internal_smoke": "OPTIONAL",
                "formal_experiment": "BLOCKED",
                "raw_redistribution": "BLOCKED",
            },
        },
        "formal_development_authorized": ["pwh", "whpn"],
        "formal_development_not_authorized": [
            "cascaded_tanks",
            "silverbox",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/public_benchmarks/pb1/protocol_audit"),
    )
    parser.add_argument(
        "--freeze-output",
        type=Path,
        default=Path("configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    whpn = audit_whpn(args.raw_root)
    tanks = audit_tanks(args.raw_root)
    freeze = build_freeze(whpn, tanks)
    for path, payload in (
        (args.output_root / "whpn_realization_audit.json", whpn),
        (args.output_root / "tanks_split_adequacy_audit.json", tanks),
        (args.freeze_output, freeze),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
