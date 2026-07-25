"""Single-source protocol configuration and phase gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_protocol import (
    CONDITIONAL_L_X_EXTENSION,
    FORMAL_L_X_CANDIDATES,
    FORMAL_L_Y_CANDIDATES,
    PREDICTION_HORIZONS,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "protocol_v2.yaml"
)


class ProtocolNotFrozenError(RuntimeError):
    """Raised when an experiment is requested before preregistration."""


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"Configuration path is missing: {dotted_path}")
        value = value[component]
    return value


def load_protocol_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    require_phase1_frozen: bool = False,
) -> dict[str, Any]:
    """Read JSON-compatible YAML without adding an unfrozen parser dependency."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = payload["frozen_protocol"]
    if tuple(protocol["horizons"]) != PREDICTION_HORIZONS:
        raise ValueError("Configured horizons differ from the frozen protocol.")
    if tuple(protocol["L_x_candidates"]) != FORMAL_L_X_CANDIDATES:
        raise ValueError("Configured L_x candidates differ from the frozen protocol.")
    if tuple(protocol["L_y_candidates"]) != FORMAL_L_Y_CANDIDATES:
        raise ValueError("Configured L_y candidates differ from the frozen protocol.")
    if protocol["conditional_L_x_extension"] != CONDITIONAL_L_X_EXTENSION:
        raise ValueError("Configured conditional L_x extension is incorrect.")
    if protocol["future_X_allowed"] is not False:
        raise ValueError("Primary protocol must forbid future X.")

    unresolved = [
        dotted_path
        for dotted_path in payload["required_before_phase1"]
        if _get_path(payload, dotted_path) is None
    ]
    payload["phase1_gate"] = {
        "status": "FROZEN" if not unresolved else "BLOCKED_PENDING_PREREGISTRATION",
        "unresolved": unresolved,
    }
    if require_phase1_frozen and unresolved:
        raise ProtocolNotFrozenError(
            "Phase 1 is locked; unresolved preregistration fields: "
            + ", ".join(unresolved)
        )
    return payload

