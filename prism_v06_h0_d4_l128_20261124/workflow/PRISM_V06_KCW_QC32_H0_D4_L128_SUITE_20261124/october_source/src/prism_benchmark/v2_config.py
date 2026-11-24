from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1"
FREEZE_STATUS = "FROZEN_BEFORE_IMPLEMENTATION_AND_V2_DEVELOPMENT_ACCESS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class V2Paths:
    project: Path
    shared: Path
    output: Path

    @property
    def plan(self) -> Path:
        return self.project / "PRISM_V2_MODULAR_NUMERICALLY_FROZEN"

    @property
    def config_path(self) -> Path:
        return self.plan / "PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json"

    @property
    def inheritance_path(self) -> Path:
        return self.plan / "PRISM_V2_FROZEN_DATA_INHERITANCE.json"


def load_frozen_config(project: Path) -> dict[str, Any]:
    path = project / "PRISM_V2_MODULAR_NUMERICALLY_FROZEN" / "PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("PRISM v2 protocol_id mismatch")
    if config.get("status") != FREEZE_STATUS:
        raise RuntimeError("PRISM v2 numerical configuration is not frozen")
    if config.get("unresolved_numeric_semantics") != []:
        raise RuntimeError("PRISM v2 has unresolved numeric semantics")
    if config.get("dtype") != "float64":
        raise RuntimeError("PRISM v2 requires FP64")
    return config


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

