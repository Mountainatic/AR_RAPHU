"""Validated structure signature shared by formal E3--E6 runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StructureSignature:
    task: str
    head: str
    H: int
    W: int
    outer_fold: str | int
    run: str | int
    rod: str | None
    seed: int
    candidate_universe: str
    model_variant: str
    admitted_channels: list[str] = field(default_factory=list)
    rejected_channels: list[str] = field(default_factory=list)
    selected_profile_by_channel: dict[str, Any] = field(default_factory=dict)
    selected_history_by_channel: dict[str, int] = field(default_factory=dict)
    selected_scale_class_by_channel: dict[str, str] = field(default_factory=dict)
    K_admitted: bool = False
    C_admitted: bool = False
    W_admitted: bool = False
    A_admitted: bool = False
    selected_K_candidate: Any = None
    selected_C_candidate: Any = None
    selected_W_candidate: Any = None
    selected_A_candidate: Any = None
    validation_gain_C: float | None = None
    validation_gain_W: float | None = None
    validation_gain_A: float | None = None
    parameter_count: int | None = None
    active_channel_count: int = 0
    active_stage_count: int = 0
    RMSE: float | None = None
    MAE: float | None = None
    R2: float | None = None
    support_id: str = ""
    config_hash: str = ""
    information_set: str = ""
    availability_scenario: str = ""
    proxy_policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["signature_hash"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return value

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
