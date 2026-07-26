import numpy as np

from ar_raphu.spectral.rank_ladder import (
    build_rank_ladder,
    normalized_excess_rmse,
)


def test_modal_gain_ladder_keeps_modes_fixed_and_full_excess_is_zero():
    rng = np.random.default_rng(34)
    train = rng.normal(size=(300, 12))
    validation = rng.normal(size=(100, 12))
    modes = []
    for index in range(3):
        coefficient = np.zeros((3, 4))
        coefficient.flat[index] = 1.0
        modes.append(coefficient)
    true_gains = np.array([2.0, -0.5, 0.25])
    train_target = 0.7 + train[:, :3] @ true_gains
    validation_target = 0.7 + validation[:, :3] @ true_gains
    ladder = build_rank_ladder(
        train_design=train,
        validation_design=validation,
        train_target=train_target,
        validation_target=validation_target,
        mode_coefficients=modes,
        full_validation_prediction=validation_target,
        rank_max=3,
    )
    assert ladder[-1].normalized_excess_rmse < 1e-12
    np.testing.assert_allclose(ladder[-1].modal_gains, true_gains)
    assert normalized_excess_rmse(
        validation_target, validation_target, validation_target
    ) == 0.0
