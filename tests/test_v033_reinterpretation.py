import csv
import json

from ar_raphu.spectral.v033_reinterpretation import reinterpret_v033


def test_reinterpretation_is_additive_and_does_not_change_old_decision(tmp_path):
    old = tmp_path / "V033_RESOLUTION_CAPACITY_DECISION.md"
    old.write_text("NEXT_ALLOWED_STAGE: STOP_STRUCTURAL_SPACE_CAPACITY\n")
    for experiment, status in (
        ("E1B", "E1B_RESOLUTION_ROLES_CERTIFIED"),
        ("E2A0", "E2A0_IMPLEMENTATION_CLOSURE_PASS"),
        ("E2A_M_SPACE", "E2A_M_SPACE_CAPACITY_PASS"),
    ):
        path = tmp_path / experiment
        path.mkdir()
        (path / "summary.json").write_text(json.dumps({"status": status}))
    structural = tmp_path / "E2A_S_SPACE"
    structural.mkdir()
    with (structural / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "validation_contribution_r2",
                "empirical_operator_nrmse",
                "core_surface_nrmse",
                "e1b_projection_core_nrmse",
                "kkt_relative_residual",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "validation_contribution_r2": 0.999,
                "empirical_operator_nrmse": 0.01,
                "core_surface_nrmse": 0.02,
                "e1b_projection_core_nrmse": 0.02,
                "kkt_relative_residual": 1e-12,
            }
        )
    before = old.read_bytes()
    result = reinterpret_v033(tmp_path)
    assert result["E2A_STRUCTURAL_FULL_SURFACE_CAPACITY"] == "PASS"
    assert result["UNIVERSAL_RANK2_COMPRESSION"] == "REJECTED"
    assert old.read_bytes() == before
