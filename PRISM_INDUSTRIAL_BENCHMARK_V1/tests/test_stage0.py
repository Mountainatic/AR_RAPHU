from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.stage0 import audit_metropt, read_numeric_text, sha256_file, write_json


def test_numeric_text_reader_keeps_only_exact_width_rows(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("title\n\n1 2 3\n4 5 6\n", encoding="utf-8")
    values, header = read_numeric_text(source, 3)
    np.testing.assert_array_equal(values, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    assert header == ["title"]


def test_json_is_stable_and_hashable(tmp_path: Path) -> None:
    target = tmp_path / "audit.json"
    write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    assert sha256_file(target) == sha256_file(target)


def test_metropt_audit_freezes_fault_windows(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    metro_root = raw_root / "metropt"
    metro_root.mkdir(parents=True)
    source = tmp_path / "MetroPT3(AirCompressor).csv"
    frame = pd.DataFrame(
        {
            "timestamp": ["2020-02-01 00:00:00", "2020-02-01 00:00:10"],
            "Reservoirs": [1.0, 1.1],
            "TP3": [2.0, 2.1],
            "Oil_temperature": [3.0, 3.1],
            "DV_pressure": [4.0, 4.1],
        }
    )
    frame.to_csv(source, index=False)
    archive_path = metro_root / "metropt_3.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(source, arcname="MetroPT3(AirCompressor).csv")

    audit = audit_metropt(raw_root)

    assert audit.decision == "PASS"
    assert audit.cadence["physical_seconds"] == 10
    assert audit.split_registry["fault_windows"] == [
        ["2020-04-18 00:00", "2020-04-18 23:59"],
        ["2020-05-29 23:30", "2020-05-30 06:00"],
        ["2020-06-05 10:00", "2020-06-07 14:30"],
        ["2020-07-15 14:30", "2020-07-15 19:00"],
    ]
