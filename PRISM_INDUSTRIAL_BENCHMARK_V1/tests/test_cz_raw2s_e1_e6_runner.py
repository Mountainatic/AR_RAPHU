from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cz_raw2s_e1_e6.py"
    spec = importlib.util.spec_from_file_location("run_cz_raw2s_e1_e6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dependency_frame(rows: int = 1400) -> pd.DataFrame:
    origin = np.arange(256, 256 + rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "absolute_origin": origin,
            "entity_id": "synthetic_segment",
            "origin": origin,
            "absolute_dependency_start": origin - 256,
            "absolute_dependency_stop_exclusive": origin + 4,
        }
    )


def test_cz_runner_uses_full_history_plus_horizon_purge() -> None:
    module = _module()
    assert module.PURGE == module.HISTORY_LENGTH + module.H == 260
    frame = _dependency_frame()
    train, validation = module.chronological_split(frame)
    assert (
        frame.loc[train, "absolute_dependency_stop_exclusive"].max()
        <= frame.loc[validation, "absolute_dependency_start"].min()
    )


def test_cz_runner_names_mse_and_rmse_skill_separately() -> None:
    module = _module()
    result = module.metrics(
        np.asarray([1.0, -1.0, 2.0, -2.0]),
        np.asarray([0.5, -0.5, 1.0, -1.0]),
        np.full(4, 10.0),
    )
    assert np.isclose(result["persistence_skill"], 0.75)
    assert np.isclose(result["persistence_skill_mse"], 0.75)
    assert np.isclose(result["persistence_skill_rmse"], 0.5)
