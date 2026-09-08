from __future__ import annotations

import json
from pathlib import Path

from experiments.tim_validation.common.hw_registry import EXPECTED, build_registry
from experiments.tim_validation.common.structure_signature import StructureSignature


PROJECT = Path(__file__).resolve().parents[1] / "PRISM_INDUSTRIAL_BENCHMARK_V1"


def test_frozen_hw_registry_resolves_cz_without_guessing(tmp_path: Path) -> None:
    audit = build_registry(PROJECT, tmp_path)
    assert audit["status"] == "PASS"
    assert audit["cz_resolution"] == {
        "H_steps": 4,
        "W_steps": 1,
        "source": "stagewise frozen task_matrix",
        "conflict": False,
    }
    assert len(audit["registry_rows"]) == len(EXPECTED)
    assert audit["outer_test_accessed"] is False


def test_structure_signature_contains_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "STRUCTURE_SIGNATURE.json"
    StructureSignature(
        task="DEB_C4",
        head="DEB_C4__H5__W1",
        H=5,
        W=1,
        outer_fold=0,
        run=0,
        rod=None,
        seed=0,
        candidate_universe="standard",
        model_variant="PRISM_V2_1_1_PHYSICS_FIRST",
        support_id="abc",
        config_hash="def",
    ).write(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "task", "head", "H", "W", "outer_fold", "run", "rod", "seed",
        "candidate_universe", "model_variant", "admitted_channels",
        "selected_history_by_channel", "K_admitted", "C_admitted",
        "W_admitted", "A_admitted", "RMSE", "MAE", "R2", "support_id",
        "config_hash", "signature_hash",
    ):
        assert field in value
