from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism_benchmark.neurobem_literature import (
    MOTOR_COLUMNS,
    TRACK_B_STATE_COLUMNS,
    resample_track_b_100hz,
)


SUPPORTED_RATES = (100, 200, 400)
REQUIRED_COLUMNS = TRACK_B_STATE_COLUMNS + MOTOR_COLUMNS


@dataclass(frozen=True)
class ResamplingAudit:
    sampling_rate_hz: int
    operator: str
    source_rate_hz: int
    rows_before: int
    rows_after: int
    median_dt_seconds: float
    maximum_dt_error_seconds: float
    missing_registered_fields: bool


def _mean_resample(frame: pd.DataFrame, period_seconds: float) -> pd.DataFrame:
    data = frame.copy()
    data["t"] = data["t"] - data["t"].iloc[0]
    data["t"] = pd.to_datetime(data["t"], unit="s")
    rule_ns = int(round(period_seconds * 1_000_000_000))
    # This is the same left-labelled, left-closed offline bin mean used by the
    # frozen official 100-Hz Track-B preprocessing. It is applied identically
    # and independently inside every trajectory and partition.
    return data.set_index("t").resample(f"{rule_ns}ns").mean().reset_index()


def resample_track_b(frame_native_400hz: pd.DataFrame, sampling_rate_hz: int) -> pd.DataFrame:
    if sampling_rate_hz not in SUPPORTED_RATES:
        raise ValueError("UNREGISTERED_NEUROBEM_SAMPLING_RATE")
    if sampling_rate_hz == 100:
        # Keep this exact call as the hard R3 reproduction boundary.
        value = resample_track_b_100hz(frame_native_400hz)
    elif sampling_rate_hz == 200:
        value = _mean_resample(frame_native_400hz, 1.0 / sampling_rate_hz)
    else:
        value = frame_native_400hz.copy()
        value["t"] = value["t"] - value["t"].iloc[0]
    if value.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("RATE_RESAMPLE_PRODUCED_MISSING_REGISTERED_FIELD")
    return value


def timestamp_seconds(frame: pd.DataFrame) -> np.ndarray:
    values = frame["t"]
    if pd.api.types.is_datetime64_any_dtype(values):
        raw = values.astype("int64").to_numpy(dtype=np.float64) / 1e9
    else:
        raw = values.to_numpy(dtype=np.float64)
    return raw - raw[0]


def audit_resampling(native: pd.DataFrame, sampled: pd.DataFrame, sampling_rate_hz: int) -> ResamplingAudit:
    time = timestamp_seconds(sampled)
    delta = np.diff(time)
    expected = 1.0 / sampling_rate_hz
    maximum_error = 0.0 if len(delta) == 0 else float(np.max(np.abs(delta - expected)))
    operator = {
        100: "OFFLINE_LEFT_CLOSED_10MS_BIN_MEAN_EXACT_R3",
        200: "OFFLINE_LEFT_CLOSED_5MS_BIN_MEAN",
        400: "NATIVE_2P5MS_NO_DOWNSAMPLING",
    }[sampling_rate_hz]
    return ResamplingAudit(
        sampling_rate_hz=sampling_rate_hz,
        operator=operator,
        source_rate_hz=400,
        rows_before=len(native),
        rows_after=len(sampled),
        median_dt_seconds=float(np.median(delta)),
        maximum_dt_error_seconds=maximum_error,
        missing_registered_fields=bool(sampled.loc[:, REQUIRED_COLUMNS].isna().any().any()),
    )

