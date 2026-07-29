"""Hash-verified atomic checkpoints for resumable ORSS tasks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    source_commit: str
    config_hash: str
    data_hash: str
    solver_version: str
    task_key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_commit": self.source_commit,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "solver_version": self.solver_version,
            "task_key": self.task_key,
        }


def checkpoint_matches(path: Path, identity: CheckpointIdentity) -> bool:
    done = path / "DONE"
    result = path / "task_result.json"
    if not done.exists() or not result.exists():
        return False
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("identity") != identity.to_dict():
        return False
    expected = done.read_text(encoding="utf-8").strip()
    return expected == sha256(result.read_bytes()).hexdigest()


def write_checkpoint(
    path: Path,
    identity: CheckpointIdentity,
    payload: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    result = path / "task_result.json"
    temporary = result.with_suffix(".json.tmp")
    complete = {"identity": identity.to_dict(), **payload}
    temporary.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(result)
    (path / "DONE").write_text(
        sha256(result.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )

