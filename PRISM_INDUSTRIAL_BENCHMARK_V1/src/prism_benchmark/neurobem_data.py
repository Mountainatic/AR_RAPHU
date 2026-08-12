"""NeuroBEM data contracts for the prospective PRISM v2.1.1 audit.

The processed CSV segment, not merely the parent flight, is the smallest legal
history entity.  This module deliberately has no interpolation or concatenation
path that can bridge segment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable, Mapping
import zipfile

import numpy as np
import pandas as pd


EXPECTED_COLUMN_COUNT = 29
MOTOR_COLUMNS = np.array([20, 21, 22, 23])
ANGULAR_ACCELERATION_COLUMNS = np.array([1, 2, 3])
ANGULAR_VELOCITY_COLUMNS = np.array([4, 5, 6])
BODY_ACCELERATION_COLUMNS = np.array([11, 12, 13])
BODY_VELOCITY_COLUMNS = np.array([14, 15, 16])
FLIGHT_RE = re.compile(r'dataset\s*=\s*"([0-9-]+)"')
SEGMENT_RE = re.compile(r"(?P<flight>[0-9-]+)_seg_(?P<segment>[0-9]+)\.csv$")


class NeuroBEMProtocolError(RuntimeError):
    """Raised when a frozen data or lockbox contract is violated."""


@dataclass(frozen=True)
class SegmentRecord:
    flight_id: str
    segment_id: str
    filename: str
    partition: str
    inner_fold: int | None
    zip_uncompressed_bytes: int
    zip_crc32: str


@dataclass(frozen=True)
class SegmentData:
    record: SegmentRecord
    values: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_flights(path: Path) -> list[str]:
    flights = FLIGHT_RE.findall(path.read_text(encoding="utf-8"))
    if len(flights) != len(set(flights)):
        raise NeuroBEMProtocolError("DUPLICATE_FLIGHT_ID")
    return flights


def parse_test_segments(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise NeuroBEMProtocolError("DUPLICATE_OFFICIAL_TEST_SEGMENT")
    return values


def parent_flight(segment_stem: str) -> str:
    match = re.fullmatch(r"(?P<flight>[0-9-]+)_seg_[0-9]+", Path(segment_stem).stem)
    if not match:
        raise NeuroBEMProtocolError(f"INVALID_SEGMENT_NAME:{segment_stem}")
    return match.group("flight")


def _hash_order(value: str, salt: str) -> str:
    return sha256(f"{salt}|{value}".encode()).hexdigest()


def frozen_parent_partitions(
    flights: Iterable[str],
    official_test_segments: Iterable[str],
    validation_parent_count: int,
    validation_salt: str,
) -> dict[str, str]:
    flights = list(flights)
    test_parents = {parent_flight(value) for value in official_test_segments}
    missing = test_parents.difference(flights)
    if missing:
        raise NeuroBEMProtocolError(f"TEST_PARENT_NOT_IN_FLIGHTS:{sorted(missing)}")
    eligible = sorted(set(flights).difference(test_parents), key=lambda x: _hash_order(x, validation_salt))
    if len(eligible) < validation_parent_count:
        raise NeuroBEMProtocolError("INSUFFICIENT_NONTEST_FLIGHTS")
    validation = set(eligible[:validation_parent_count])
    return {
        flight: "test" if flight in test_parents else "validation" if flight in validation else "train"
        for flight in flights
    }


def frozen_inner_fold(flight_id: str, count: int, salt: str) -> int:
    if count < 2:
        raise ValueError("fold count must be at least two")
    return int(_hash_order(flight_id, salt)[:16], 16) % count


def registry_from_zip(
    zip_path: Path,
    flights_path: Path,
    testset_path: Path,
    config: Mapping[str, object],
) -> list[SegmentRecord]:
    flights = parse_flights(flights_path)
    test_segments = parse_test_segments(testset_path)
    entity = config["entity_contract"]
    partitions = frozen_parent_partitions(
        flights,
        test_segments,
        int(entity["validation_parent_count"]),
        str(entity["validation_salt"]),
    )
    records: list[SegmentRecord] = []
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".csv"):
                continue
            match = SEGMENT_RE.search(info.filename)
            if not match:
                raise NeuroBEMProtocolError(f"UNREGISTERED_CSV_NAME:{info.filename}")
            flight = match.group("flight")
            if flight not in partitions:
                raise NeuroBEMProtocolError(f"CSV_PARENT_NOT_IN_FLIGHTS:{flight}")
            segment_id = Path(info.filename).stem
            if segment_id in seen:
                raise NeuroBEMProtocolError(f"DUPLICATE_SEGMENT:{segment_id}")
            seen.add(segment_id)
            partition = partitions[flight]
            fold = None
            if partition == "train":
                fold = frozen_inner_fold(flight, int(entity["inner_folds"]), str(entity["inner_fold_salt"]))
            records.append(
                SegmentRecord(
                    flight_id=flight,
                    segment_id=segment_id,
                    filename=info.filename,
                    partition=partition,
                    inner_fold=fold,
                    zip_uncompressed_bytes=int(info.file_size),
                    zip_crc32=f"{info.CRC:08x}",
                )
            )
    present_flights = {record.flight_id for record in records}
    if present_flights != set(flights):
        raise NeuroBEMProtocolError(
            f"FLIGHT_FILE_COVERAGE_MISMATCH:missing={sorted(set(flights)-present_flights)}"
        )
    official = set(test_segments)
    missing_official = official.difference({record.segment_id for record in records})
    if missing_official:
        raise NeuroBEMProtocolError(f"OFFICIAL_TEST_SEGMENT_MISSING:{sorted(missing_official)}")
    return sorted(records, key=lambda value: value.segment_id)


def registry_json(records: Iterable[SegmentRecord]) -> list[dict[str, object]]:
    return [record.__dict__.copy() for record in records]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_processed_archive(zip_path: Path, destination: Path) -> None:
    """Extract bytes only; model code still guards locked numeric reads."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise NeuroBEMProtocolError(f"ZIP_CRC_FAILURE:{bad}")
        archive.extractall(destination)


def locate_extracted_csv(extracted_root: Path, record: SegmentRecord) -> Path:
    exact = extracted_root / record.filename
    if exact.is_file():
        return exact
    matches = list(extracted_root.rglob(f"{record.segment_id}.csv"))
    if len(matches) != 1:
        raise NeuroBEMProtocolError(f"EXTRACTED_SEGMENT_LOOKUP_FAILED:{record.segment_id}")
    return matches[0]


def load_segment(
    extracted_root: Path,
    record: SegmentRecord,
    *,
    allow_locked_test: bool = False,
) -> SegmentData:
    if record.partition == "test" and not allow_locked_test:
        raise NeuroBEMProtocolError(f"TEST_LOCKBOX_ACCESS_FORBIDDEN:{record.segment_id}")
    path = locate_extracted_csv(extracted_root, record)
    frame = pd.read_csv(path)
    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        raise NeuroBEMProtocolError(f"COLUMN_COUNT_MISMATCH:{record.segment_id}:{frame.shape[1]}")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[0] < 2:
        raise NeuroBEMProtocolError(f"EMPTY_OR_SHORT_SEGMENT:{record.segment_id}")
    if not np.isfinite(values).all():
        raise NeuroBEMProtocolError(f"NONFINITE_SEGMENT:{record.segment_id}")
    dt = np.diff(values[:, 0])
    if not np.all(dt > 0.0):
        raise NeuroBEMProtocolError(f"NONMONOTONE_TIME:{record.segment_id}")
    return SegmentData(record=record, values=values)


def generalized_targets(segment: SegmentData, mass: float, inertia: Iterable[float]) -> np.ndarray:
    values = segment.values
    angular_acceleration = values[:, ANGULAR_ACCELERATION_COLUMNS]
    angular_velocity = values[:, ANGULAR_VELOCITY_COLUMNS]
    inertia = np.asarray(list(inertia), dtype=np.float64)
    angular_momentum = angular_velocity * inertia[None, :]
    torque = angular_acceleration * inertia[None, :] + np.cross(angular_velocity, angular_momentum)
    force_z = mass * values[:, BODY_ACCELERATION_COLUMNS[2]]
    return np.column_stack((torque, force_z))


def motor_thrust_proxy(segment: SegmentData) -> np.ndarray:
    return np.square(segment.values[:, MOTOR_COLUMNS], dtype=np.float64)


def body_context(segment: SegmentData) -> np.ndarray:
    return np.column_stack(
        (segment.values[:, BODY_VELOCITY_COLUMNS], segment.values[:, ANGULAR_VELOCITY_COLUMNS])
    )


def legal_target_rows(row_count: int, history_steps: int) -> np.ndarray:
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    return np.arange(history_steps, row_count, dtype=np.int64)


def sample_id(record: SegmentRecord, target_row_index: int) -> str:
    payload = f"NeuroBEM|{record.flight_id}|{record.segment_id}|{target_row_index}"
    return sha256(payload.encode()).hexdigest()


def support_hash(record: SegmentRecord, target_rows: Iterable[int]) -> str:
    digest = sha256()
    for row in target_rows:
        digest.update(bytes.fromhex(sample_id(record, int(row))))
    return digest.hexdigest()


def assert_partition_disjoint(records: Iterable[SegmentRecord]) -> None:
    by_flight: dict[str, set[str]] = {}
    for record in records:
        by_flight.setdefault(record.flight_id, set()).add(record.partition)
    bad = {flight: sorted(parts) for flight, parts in by_flight.items() if len(parts) != 1}
    if bad:
        raise NeuroBEMProtocolError(f"PARENT_FLIGHT_SPLIT_LEAKAGE:{bad}")


def development_data_audit(segments: Iterable[SegmentData], expected_dt: float = 0.0025) -> dict[str, object]:
    segments = list(segments)
    rows = int(sum(segment.row_count for segment in segments))
    dts = np.concatenate([np.diff(segment.values[:, 0]) for segment in segments])
    return {
        "status": "PASS",
        "development_segments_parsed": len(segments),
        "development_rows": rows,
        "median_dt_seconds": float(np.median(dts)),
        "maximum_abs_dt_deviation_seconds": float(np.max(np.abs(dts - expected_dt))),
        "test_numeric_values_accessed": False,
        "history_entity": "CONTIGUOUS_PROCESSED_SEGMENT_ID",
        "cross_segment_history_allowed": False,
    }
