import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODULE_PATH = PROJECT_ROOT / "tools" / "run_phase1_m8.py"
SPEC = importlib.util.spec_from_file_location("phase1_m8_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def frozen_config() -> dict:
    return {
        "phase1_model_selection": {
            "M8": {
                "lag_spline_grid_sizes": [5, 8, 12],
                "residual_smoothness_weights": [1.0e-4, 1.0e-3, 1.0e-2],
                "pilot_smoothing_weight": 1.0e-3,
                "lag_grid_stability_rule": {
                    "minimum_one_se_memberships": 2
                },
            }
        }
    }


def rows_from_means(means: dict[tuple[int, float], float]):
    rows = []
    lookup = {}
    offsets = [-0.001, 0.0, 0.001]
    for (grid, weight), mean in means.items():
        identifier = MODULE.config_id(grid, weight)
        lookup[identifier] = (grid, weight)
        for seed, offset in enumerate(offsets):
            rows.append(
                {
                    "config_id": identifier,
                    "unit_id": f"seed_{seed}",
                    "validation_loss": mean + offset,
                }
            )
    return rows, lookup


def test_m8_stable_branch_freezes_grid_then_prefers_smoothing(
    tmp_path, monkeypatch
) -> None:
    means = {
        (5, 1.0e-4): 1.20,
        (8, 1.0e-4): 1.00,
        (12, 1.0e-4): 1.10,
        (5, 1.0e-3): 1.20,
        (8, 1.0e-3): 1.00,
        (12, 1.0e-3): 1.10,
        (5, 1.0e-2): 1.20,
        (8, 1.0e-2): 1.00,
        (12, 1.0e-2): 1.10,
    }
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(
        MODULE, "load_protocol_config", lambda require_phase1_frozen: frozen_config()
    )
    monkeypatch.setattr(MODULE, "candidate_rows", lambda config: rows_from_means(means))
    selected = MODULE.select_configuration()
    assert selected["selection_branch"] == "stable_sequential"
    assert selected["M8_GRID_SMOOTHING_INTERACTION"] is False
    assert selected["selected_config_id"] == "gtau=8;lambda=0.01"
    persisted = json.loads(
        (tmp_path / "validation_selection.json").read_text(encoding="utf-8")
    )
    assert persisted["rank_inputs_used_for_selection"] is False


def test_m8_unstable_branch_triggers_declared_cartesian_fallback(
    tmp_path, monkeypatch
) -> None:
    means = {
        (5, 1.0e-4): 1.00,
        (8, 1.0e-4): 1.30,
        (12, 1.0e-4): 1.20,
        (5, 1.0e-3): 1.20,
        (8, 1.0e-3): 1.00,
        (12, 1.0e-3): 1.10,
        (5, 1.0e-2): 1.20,
        (8, 1.0e-2): 1.30,
        (12, 1.0e-2): 1.00,
    }
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(
        MODULE, "load_protocol_config", lambda require_phase1_frozen: frozen_config()
    )
    monkeypatch.setattr(MODULE, "candidate_rows", lambda config: rows_from_means(means))
    selected = MODULE.select_configuration()
    assert selected["selection_branch"] == "full_cartesian_fallback"
    assert selected["M8_GRID_SMOOTHING_INTERACTION"] is True
    assert selected["pilot_grid_one_se_membership_count"] == 1
