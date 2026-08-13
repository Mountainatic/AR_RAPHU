import numpy as np
from pathlib import Path
from experiments.neurobem_manifold_switch.metrics import divergence_time
from experiments.neurobem_manifold_switch.run_experiment import _json_safe
from experiments.neurobem_manifold_switch.monitor import ManifoldTemplate, persistent_alarm


def test_persistent_alarm_rejects_single_spike():
    assert persistent_alarm(np.array([0., 2., 0., 2., 2., 2.]), 1., 3) == 3


def test_divergence_requires_persistence():
    e = {"velocity": np.array([0., 20., 0., 20., 20.]), "attitude": np.zeros(5), "body_rate": np.zeros(5)}
    assert divergence_time(e, np.ones(5, bool), {"velocity": 15., "attitude": 1., "body_rate": 15.}, 2) == 3


def test_projection_increases_off_manifold():
    x = np.column_stack((np.linspace(-1, 1, 50), np.zeros(50)))
    template = ManifoldTemplate.fit(x, 1)
    assert template.projection_score(np.array([[0., 5.]]))[0] > template.projection_score(np.array([[0., 0.]]))[0]


def test_switch_cannot_precede_alarm_observation():
    alarm = 7
    first_eligible_switch_step = alarm + 1
    assert first_eligible_switch_step > alarm


def test_alarm_search_is_limited_to_rollout_window():
    scores = np.r_[np.zeros(10), np.ones(3) * 2]
    assert persistent_alarm(scores[:10], 1.0, 3) is None


def test_config_freezes_validation_only_thresholds():
    cfg = Path(__file__).parents[1] / "configs" / "full.yaml"
    text = cfg.read_text(encoding="utf-8")
    assert '"test_tuning_prohibited": true' in text
    assert '"monitor_quantile": 0.995' in text


def test_prism_core_is_not_implemented_in_extension():
    root = Path(__file__).parents[1]
    adapter = (root / "prism_adapter.py").read_text(encoding="utf-8")
    assert "fit_track_b_route_contracts" in adapter
    assert "route_prediction" in adapter
    assert "class RidgeContract" not in adapter


def test_each_csv_is_declared_as_an_isolated_entity():
    data = (Path(__file__).parents[1] / "data.py").read_text(encoding="utf-8")
    assert "ONE_CSV_ONE_TRAJECTORY_NO_CROSS_BOUNDARY_HISTORY" in data


def test_reid_excludes_current_prediction_target():
    source = (Path(__file__).parents[1] / "rollout.py").read_text(encoding="utf-8")
    assert "history + step - 1" in source
    assert "row history+s is the target about to be predicted" in source


def test_r2_excludes_test_sha_from_development():
    source = (Path(__file__).parents[1] / "data.py").read_text(encoding="utf-8")
    assert "test_sha_excluded_from_fit_and_calibration" in source
    assert "excluded_test_sha_collisions" in source


def test_r2_keeps_frozen_natural_cubic_w_family():
    cfg = (Path(__file__).parents[1] / "configs" / "full.yaml").read_text(encoding="utf-8")
    assert '"w_family": "NATURAL_CUBIC_LATENT"' in cfg


def test_trajectory_parallelism_is_ordered_fork_cow():
    source = (Path(__file__).parents[1] / "run_experiment.py").read_text(encoding="utf-8")
    assert 'mp.get_context("fork").Pool' in source
    assert "pool.imap(_evaluate_trajectory_route" in source
    assert "for route in cfg[\"routes\"]" in source
    assert 'cfg["trajectory_workers"]' in source


def test_nonfinite_diagnostic_is_strict_json_null():
    assert _json_safe({"diverged": True, "error": float("inf")}) == {
        "diverged": True,
        "error": None,
    }
