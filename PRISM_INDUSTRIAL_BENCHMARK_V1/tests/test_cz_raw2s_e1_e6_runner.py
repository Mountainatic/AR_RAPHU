from __future__ import annotations

import importlib.util
import hashlib
import json
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


def test_cz_runner_delegates_to_authority_without_proxy_model_code() -> None:
    module = _module()
    assert module.AUTHORITY_COMMIT == "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
    assert module.AUTHORITY_RUNNER_RELATIVE_PATH.as_posix() == (
        "scripts/run_independent_extension_20260825.py"
    )
    forbidden = {
        "fit_ridge",
        "predict_ridge",
        "block_design",
        "candidate_specs",
        "select_chain",
        "RouteSpec",
        "RidgeContract",
    }
    assert forbidden.isdisjoint(vars(module))


def test_cz_runner_pins_every_authority_module_and_zero_c_patch() -> None:
    module = _module()
    project = Path(__file__).resolve().parents[1]
    audit = module.authority_audit(project)
    assert audit["status"] == "PASS"
    assert audit["authority_is_ancestor"] is True
    assert audit["custom_prism_feature_or_estimator_code_present"] is False
    statuses = {item["status"] for item in audit["modules"]}
    assert statuses == {
        "BYTE_IDENTICAL_TO_AUTHORITY",
        "AUTHORITY_PLUS_REVIEWED_BEST_K_ENUM_PATCH",
        "AUTHORITY_PLUS_REVIEWED_ZERO_C_PATCH",
        "AUTHORITY_PLUS_REVIEWED_METRIC_LABEL_PATCH",
    }


def test_cz_authority_config_is_h4_only_and_pins_exact_commit() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            project
            / "configs"
            / "cz_raw2s_h4_authoritative_e1_e6_20260922.json"
        ).read_text(encoding="utf-8")
    )
    assert config["baseline_commit"] == (
        "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
    )
    assert config["cz"]["h_steps"] == [4]
    assert config["cz"]["sampling_period_seconds"] == 2
    assert config["cz"]["history_steps"] == 256
    assert config["cz"]["target_formula"] == "D[t+h-1]-D[t-1]"


def test_representative_extension_code_bindings_match_current_files() -> None:
    project = Path(__file__).resolve().parents[1]
    descriptor = json.loads(
        (
            project
            / "configs"
            / "representative_horizon_stage1_tep_sru_cpu_extension_20260825.json"
        ).read_text(encoding="utf-8")
    )
    for record in descriptor["formal_code_bindings"]:
        raw = (project / record["path"]).read_bytes()
        observed = {
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
        assert len(record["sha256"]) == 64
        assert record["sha256"] in observed
