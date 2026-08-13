import numpy as np

from experiments.neurobem_recursive_stability.core import (
    ABLATION_BLOCKS, direction_vector, effective_delta, perturb_latest,
)


def test_deterministic_random_direction():
    assert np.array_equal(direction_vector("random_0", 17), direction_vector("random_0", 17))
    assert np.isclose(np.linalg.norm(direction_vector("random_0", 17)), 1.0)


def test_basis_directions_cover_effective_state():
    names = ["velocity_x", "velocity_y", "velocity_z", "attitude_x", "attitude_y", "attitude_z", "body_rate_x", "body_rate_y", "body_rate_z"]
    matrix = np.stack([direction_vector(name, 0) for name in names])
    assert np.array_equal(matrix, np.eye(9))


def test_attitude_perturbation_is_unit_quaternion_and_local():
    history = np.zeros((20, 10)); history[:, 3] = 1.0
    changed = perturb_latest(history, direction_vector("attitude_y", 0), 1e-5)
    assert np.allclose(changed[:-1], history[:-1])
    assert np.isclose(np.linalg.norm(changed[-1, 3:7]), 1.0)
    delta = effective_delta(history[-1], changed[-1])
    assert np.allclose(delta, direction_vector("attitude_y", 0) * 1e-5, atol=1e-12)


def test_channel_ablation_set_is_registered_exactly():
    assert set(ABLATION_BLOCKS) == {"velocity_only", "attitude_only", "body_rate_only", "velocity_attitude", "attitude_body_rate", "full_recursive"}


def test_position_is_not_fabricated_in_track_b_state():
    assert all("position" not in blocks for blocks in ABLATION_BLOCKS.values())


def test_no_future_observation_in_free_rollout_source():
    source = (__import__("pathlib").Path(__file__).parents[1] / "core.py").read_text()
    assert "predicted_history = state[:history].copy()" in source
    assert "control[step:step + history]" in source
    assert "if resync_interval is not None" in source


def test_resynchronization_is_explicit_after_observation():
    source = (__import__("pathlib").Path(__file__).parents[1] / "core.py").read_text()
    assert "(step + 1) % resync_interval == 0" in source
    assert "state[step + 1:step + history + 1]" in source


def test_fork_parallelism_is_ordered():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert 'mp.get_context("fork").Pool' in source
    assert "pool.imap(_task, tasks, chunksize=1)" in source


def test_test_stage_requires_calibration_freeze():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert "TEST_REQUIRES_CALIBRATION_FREEZE" in source
    assert "R3_CONFIG_HASH_MISMATCH" in source
    assert "R3_ADAPTER_HASH_MISMATCH" in source


def test_reliability_is_nested_by_trajectory_then_across_trajectories():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert "trajectory_reliable=" in source
    assert ".groupby(\"resync_interval\").trajectory_reliable.mean()" in source


def test_r2_baseline_is_a_hard_gate():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert "BLOCKED_R2_BASELINE_REPRODUCTION_MISMATCH" in source
    assert "new_tracks_executed\": False" in source


def test_expansion_threshold_uses_same_anchor_max_statistic_as_event():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert 'groupby(["trajectory_id", "anchor_step"]).max_growth.max()' in source


def test_growth_curves_and_channel_growth_are_saved():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    for field in ("growth_curve", "velocity_growth_curve", "attitude_growth_curve", "body_rate_growth_curve"):
        assert field in source


def test_registered_component_routes_are_audited_without_refit():
    config = (__import__("json").loads((__import__("pathlib").Path(__file__).parents[1] / "configs" / "calibration.yaml").read_text()))
    assert config["component_routes"] == ["PF_KC", "PF_KCW", "J_KC", "J_KCW"]
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert "component_route_attribution.csv" in source


def test_jacobian_product_scope_is_not_overclaimed():
    source = (__import__("pathlib").Path(__file__).parents[1] / "run_experiment.py").read_text()
    assert "newest_block_product_growth_rate" in source
    assert "NEWEST_STATE_9X9_BLOCK_NOT_FULL_20_STEP_AUGMENTED_JACOBIAN" in source
