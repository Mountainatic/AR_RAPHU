import numpy as np

from prism_benchmark.prism_ct import (
    CTBasisConfig,
    aligned_temporal_blocks,
    ct_multires_features,
    feature_audit,
    stable_ct_states,
)


def test_constant_signal_is_fixed_point():
    config = CTBasisConfig(dt_seconds=2.0, taus_seconds=(10.0, 60.0, 600.0))
    x = np.full((100, 2), 3.25)
    z = stable_ct_states(x, config)
    assert np.allclose(z, 3.25)


def test_step_response_is_stable_and_monotone():
    config = CTBasisConfig(dt_seconds=1.0, taus_seconds=(10.0,))
    x = np.zeros((200, 1))
    x[20:] = 1.0
    z = stable_ct_states(x, config)[:, 0, 0]
    assert np.all(np.diff(z[20:]) >= -1e-12)
    assert 0.999999 < z[-1] < 1.0


def test_multires_telescopes_to_current_minus_slowest_state():
    config = CTBasisConfig(dt_seconds=2.0, taus_seconds=(10.0, 30.0, 60.0))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 2))
    z = stable_ct_states(x, config)
    multires = ct_multires_features(x, config, include_current=False)
    channels = x.shape[1]
    reconstructed = sum(
        multires[:, index * channels : (index + 1) * channels]
        for index in range(len(config.taus_seconds))
    )
    assert np.allclose(reconstructed, x - z[:, :, -1], atol=1e-10)


def test_causal_features_do_not_change_when_future_changes():
    config = CTBasisConfig(dt_seconds=2.0, taus_seconds=(10.0, 30.0, 60.0))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 2))
    time_index = 150
    first, start = aligned_temporal_blocks(x, config)
    changed = x.copy()
    changed[time_index + 1 :] += 1000.0
    second, second_start = aligned_temporal_blocks(changed, config)
    assert start == second_start
    row = time_index - start
    for name in first:
        assert np.allclose(first[name][row], second[name][row])


def test_feature_audit_detects_exact_rank_deficiency():
    config = CTBasisConfig(condition_hard_fail=1e8)
    x = np.arange(100.0, dtype=float)[:, None]
    duplicate = np.concatenate([x, x], axis=1)
    audit = feature_audit(duplicate, config)
    assert not audit.passed_conditioning
    assert audit.matrix_rank < audit.n_features
