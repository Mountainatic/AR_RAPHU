from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from prism_benchmark.unified_hw_protocol import protocols


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_unified_hw_r2.py"
    spec = importlib.util.spec_from_file_location("report_unified_hw_r2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reporter_ignores_non_numeric_metadata_columns(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "predictions.csv"
    path.write_text(
        "sample_id,entity_id,delta_true,delta_pred,current_level\n"
        "a,unit-1,1.0,0.5,10.0\n",
        encoding="utf-8",
    )
    columns = module._columns(
        path, {"delta_true", "delta_pred", "current_level"}
    )
    assert columns == {
        "delta_true": [1.0],
        "delta_pred": [0.5],
        "current_level": [10.0],
    }


def test_reporter_reconstructs_and_rejects_wrong_registered_target() -> None:
    module = _module()
    protocol = protocols()["SRU_H2S_REP_H1__H1__W1"]
    values = np.arange(20, dtype=np.float64)
    valid = {
        "origin": [7.0],
        "current_level": [6.0],
        "delta_true": [2.0],
    }
    module._verify_targets(protocol, valid, values)
    invalid = {**valid, "delta_true": [1.0]}
    with np.testing.assert_raises(RuntimeError):
        module._verify_targets(protocol, invalid, values)
