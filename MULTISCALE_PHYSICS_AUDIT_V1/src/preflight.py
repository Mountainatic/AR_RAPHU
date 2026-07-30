"""Non-model time-scale and breakpoint audit."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import periodogram

from .data_loader import WorkbookData
from .segmentation import detect_breakpoints


def _acf_decay(values: np.ndarray, threshold: float = np.e**-1) -> int:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    variance = float(centered @ centered)
    if variance <= np.finfo(np.float64).eps:
        return 0
    maximum = min(len(centered) // 3, 5000)
    fft_length = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    autocovariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_length
    )[: maximum + 1]
    autocorrelation = autocovariance / max(
        autocovariance[0], np.finfo(np.float64).eps
    )
    crossings = np.flatnonzero(autocorrelation[1:] <= threshold)
    if len(crossings):
        return int(crossings[0] + 1)
    return maximum


def _update_statistics(values: np.ndarray, sample_period_sec: float) -> dict[str, float]:
    differences = np.diff(values)
    nonzero = np.flatnonzero(np.abs(differences) > 1.0e-12) + 1
    intervals = np.diff(np.concatenate(([0], nonzero, [len(values) - 1])))
    positive = intervals[intervals > 0]
    return {
        "nonzero_update_fraction": float(np.mean(np.abs(differences) > 1.0e-12)),
        "median_update_interval_sec": (
            float(np.median(positive) * sample_period_sec) if len(positive) else 0.0
        ),
        "median_hold_sec": (
            float(np.median(positive) * sample_period_sec) if len(positive) else 0.0
        ),
        "diff_median_abs": float(np.median(np.abs(differences))),
        "diff_p95_abs": float(np.quantile(np.abs(differences), 0.95)),
        "acf_decay_sec": float(_acf_decay(values) * sample_period_sec),
    }


def _spectral_bands(values: np.ndarray, sample_period_sec: float) -> dict[str, float]:
    frequency, power = periodogram(
        np.asarray(values, dtype=np.float64),
        fs=1.0 / sample_period_sec,
        detrend="linear",
    )
    total = float(np.sum(power))
    if total <= np.finfo(np.float64).eps:
        return {"lt_10min": 0.0, "10_to_60min": 0.0, "gt_60min": 0.0}
    periods = np.full_like(frequency, np.inf)
    periods[frequency > 0] = 1.0 / frequency[frequency > 0]
    return {
        "lt_10min": float(np.sum(power[periods < 600.0]) / total),
        "10_to_60min": float(
            np.sum(power[(periods >= 600.0) & (periods < 3600.0)]) / total
        ),
        "gt_60min": float(np.sum(power[periods >= 3600.0]) / total),
    }


def _lag_correlation_summary(
    x: np.ndarray, y: np.ndarray, sample_period_sec: float
) -> dict[str, float]:
    window = max(1, int(round(600.0 / sample_period_sec)))
    future_change = np.convolve(y, np.ones(window) / window, mode="valid")
    future_change = future_change[window:] - future_change[:-window]
    base_x = x[window - 1 : window - 1 + len(future_change)]
    best_correlation = 0.0
    best_lag = 0
    for lag_min in (0, 2, 5, 10, 20, 40, 60, 90, 120, 180):
        lag = int(round(lag_min * 60.0 / sample_period_sec))
        if lag >= len(future_change) - 20:
            continue
        left = base_x[: len(base_x) - lag] if lag else base_x
        right = future_change[lag:] if lag else future_change
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        if abs(correlation) > abs(best_correlation):
            best_correlation, best_lag = correlation, lag_min
    return {
        "max_abs_lag_correlation": best_correlation,
        "lag_at_max_abs_correlation_min": float(best_lag),
    }


def run_preflight(
    workbook: WorkbookData,
    config: dict[str, Any],
    *,
    sample_period_sec: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": config["schema"],
        "status": "COMPLETED",
        "data_sha256": workbook.sha256,
        "sample_period_sec": float(sample_period_sec),
        "timestamp_status": "NOT_AVAILABLE",
        "sheets": {},
    }
    common_ranges: dict[str, list[float]] = {}
    diameter_name = "晶体直径"
    for sheet, rod in workbook.rods.items():
        detected = detect_breakpoints(
            rod.columns[diameter_name],
            threshold=float(config["data"]["diameter_break_threshold_mm"]),
        )
        frozen = [int(value) for value in config["data"]["frozen_breakpoints"][sheet]]
        stable_start = max(frozen) + 1 if frozen else 0
        channel_stats: dict[str, Any] = {}
        for name, values in rod.columns.items():
            stable = values[stable_start:]
            channel_stats[name] = {
                **_update_statistics(stable, sample_period_sec),
                "spectral_energy_fraction": _spectral_bands(
                    stable, sample_period_sec
                ),
                "range": [float(np.min(stable)), float(np.max(stable))],
            }
            if name != diameter_name:
                channel_stats[name].update(
                    _lag_correlation_summary(
                        stable,
                        rod.columns[diameter_name][stable_start:],
                        sample_period_sec,
                    )
                )
        report["sheets"][sheet] = {
            "samples": rod.samples,
            "detected_breakpoints": detected,
            "frozen_breakpoints": frozen,
            "frozen_breakpoints_verified": detected == frozen,
            "main_stable_segment": [stable_start, rod.samples],
            "main_stable_duration_hours": (
                (rod.samples - stable_start) * sample_period_sec / 3600.0
            ),
            "channels": channel_stats,
        }
    for name in workbook.rods["Sheet1"].columns:
        if name == diameter_name:
            continue
        ranges = [
            report["sheets"][sheet]["channels"][name]["range"]
            for sheet in ("Sheet1", "Sheet2")
        ]
        common_ranges[name] = [
            max(ranges[0][0], ranges[1][0]),
            min(ranges[0][1], ranges[1][1]),
        ]
    report["common_amplitude_domains"] = common_ranges
    report["preflight_pass"] = all(
        sheet["frozen_breakpoints_verified"]
        for sheet in report["sheets"].values()
    )
    if not report["preflight_pass"]:
        report["status"] = "FAILED"
    return report
