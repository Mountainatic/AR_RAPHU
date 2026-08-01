from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.stage0 import read_numeric_text, sha256_file, write_json


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

