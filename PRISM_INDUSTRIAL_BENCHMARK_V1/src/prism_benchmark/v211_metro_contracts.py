from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_selection import mse
from .stage0 import write_json
from .v21_selection import assert_final_prediction_contract


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def stable_candidate_id(stage: str, descriptor: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(descriptor),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"METRO_P60_{stage}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _selected_descriptor(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stage",
        "dataset",
        "target_head",
        "availability_scenario",
        "proxy_policy",
        "channel",
        "final_selected_candidate",
        "selected_profile",
        "selected_intervals",
        "selected_family",
        "selected_m_tau",
        "selected_m_x",
        "selected_lambdas",
        "selected_alpha",
        "active_channels",
        "ar_profile",
    )
    return {key: result[key] for key in keys if key in result}


def bind_result_candidate_ids(output: Path, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") not in {
        "PASS",
        "JOINT_INPUT_PATH_COLLAPSED",
        "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED",
    }:
        return result
    prediction_path = output / str(result["final_selected_prediction_path"])
    prediction = pd.read_parquet(prediction_path, columns=["y_true", "y_pred"])
    recomputed = mse(
        prediction["y_true"].to_numpy(dtype=np.float64),
        prediction["y_pred"].to_numpy(dtype=np.float64),
    )
    assert_final_prediction_contract(result, recomputed_loss=recomputed)
    descriptor = _selected_descriptor(result)
    selected_id = stable_candidate_id(str(result["stage"]), descriptor)
    registry = []
    for candidate, losses in sorted(
        result.get("candidate_fold_losses", {}).items(), key=lambda pair: str(pair[0])
    ):
        candidate_descriptor = {
            **{key: descriptor[key] for key in descriptor if key not in {"final_selected_candidate"}},
            "candidate": str(candidate),
        }
        registry.append(
            {
                "candidate_id": stable_candidate_id(str(result["stage"]), candidate_descriptor),
                "candidate": str(candidate),
                "fold_losses": losses,
            }
        )
    result.update(
        {
            "final_selected_candidate_id": selected_id,
            "selected_loss_candidate_id": selected_id,
            "selected_prediction_candidate_id": selected_id,
            "selected_contract_candidate_id": selected_id,
            "candidate_registry": registry,
            "candidate_id_binding": {
                "status": "PASS",
                "prediction_loss_recomputed": recomputed,
                "prediction_path": str(result["final_selected_prediction_path"]),
                "prediction_sha256": result.get("prediction_sha256"),
            },
        }
    )
    write_json(result_path, result)
    return result


def assert_candidate_id_binding(result: Mapping[str, Any]) -> None:
    identifiers = {
        result.get("final_selected_candidate_id"),
        result.get("selected_loss_candidate_id"),
        result.get("selected_prediction_candidate_id"),
        result.get("selected_contract_candidate_id"),
    }
    if len(identifiers) != 1 or None in identifiers:
        raise RuntimeError("STOP_CANDIDATE_ID_MISMATCH")
    if result.get("candidate_id_binding", {}).get("status") != "PASS":
        raise RuntimeError("STOP_CANDIDATE_ID_MISMATCH")
