from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "certify_cz_authority_tiny_pilot_parallelism.py"
)
SPEC = importlib.util.spec_from_file_location("cz_tiny_pilot_parallelism", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_four_way_tiny_pilot_gate_is_scoped_and_resource_checked(
    monkeypatch, tmp_path: Path
) -> None:
    parent = tmp_path / "run"
    _write(parent / "STREAMING_STORAGE_GATE.json", {"status": "PASS"})
    _write(parent / "WORKER_EQUIVALENCE.json", {"status": "PASS"})
    units = []
    for index in range(4):
        unit = parent / "units" / str(index)
        _write(
            unit / "AUTHORITY_REFIT_STATUS.json",
            {
                "status": "PASS",
                "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
                "inner_workers": 1 if index < 2 else 2,
                "formal_target_or_ood_accessed": False,
            },
        )
        units.append(unit)
    gib = 1024**3
    values = iter(
        [
            shutil._ntuple_diskusage(50 * gib, 34 * gib, 16 * gib),
            shutil._ntuple_diskusage(30 * gib, 0, 30 * gib),
        ]
    )
    monkeypatch.setattr(MODULE.shutil, "disk_usage", lambda _: next(values))
    result = MODULE.certify(parent, units, tmp_path)
    assert result["status"] == "PASS"
    assert result["maximum_concurrent_outer_units"] == 4
    assert result["inner_workers_per_unit"] == 2
    gate = json.loads(
        (parent / "STREAMING_STORAGE_GATE.json").read_text(encoding="utf-8")
    )
    assert gate["tiny_pilot_parallelism"] == "PASS_RESOURCE_CERTIFICATE"
    assert gate["tiny_pilot_maximum_concurrent_outer_units"] == 4
