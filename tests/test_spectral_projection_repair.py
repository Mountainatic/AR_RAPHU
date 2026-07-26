import json
from pathlib import Path

import numpy as np

from ar_raphu.spectral.projection import (
    identity_lag_basis,
    project_tensor_surface,
)
from ar_raphu.spectral.spline_basis import (
    CenteredSplineBasis,
    clamped_knots,
    evaluate_basis,
)
from ar_raphu.spectral.synthetic_components import true_kernel_surface
from ar_raphu.synthetic import generate_synthetic_sequence


ROOT = Path(__file__).resolve().parents[1]


def test_identity_lag_basis_is_exact_64_by_64_identity():
    basis = identity_lag_basis(64)
    assert basis.shape == (64, 64)
    assert basis.dtype == np.float64
    assert np.array_equal(basis, np.eye(64))


def test_two_sided_projection_matches_explicit_kronecker_projection():
    rng = np.random.default_rng(42)
    lag = rng.normal(size=(8, 5))
    amplitude = rng.normal(size=(8, 4))
    truth = rng.normal(size=(8, 8))
    result = project_tensor_surface(truth, lag, amplitude)
    explicit = np.einsum("ia,jb->ijab", lag, amplitude).reshape(64, -1)
    coefficients = np.linalg.lstsq(explicit, truth.reshape(-1), rcond=None)[0]
    estimate = (explicit @ coefficients).reshape(8, 8)
    assert np.max(np.abs(result.estimate - estimate)) <= 1e-10


def _scenario_worst_nrmse(scenario: str, lag_count: int) -> float:
    config = json.loads(
        (ROOT / "configs" / "spectral_v031.yaml").read_text(encoding="utf-8")
    )
    amplitude_config = config["amplitude_basis"]
    worst = 0.0
    for seed in config["common"]["development_seeds"]:
        sequence = generate_synthetic_sequence(
            scenario,
            seed=seed,
            n_samples=config["common"]["n_samples"],
            external_variables=config["common"]["external_variables"],
        )
        _, train_stop = sequence.split_target_intervals["train"]
        for variable in sequence.truth["active_support"]:
            train_values = sequence.x[:train_stop, variable]
            lower, upper = np.quantile(
                train_values, amplitude_config["quantiles"]
            )
            amplitudes = np.linspace(
                lower, upper, amplitude_config["evaluation_grid_points"]
            )
            amp_basis = CenteredSplineBasis.fit(
                train_values,
                n_basis=amplitude_config["basis_count"],
                degree=amplitude_config["degree"],
                quantiles=tuple(amplitude_config["quantiles"]),
            )
            amp_eval = amp_basis.transform(amplitudes)
            truth = true_kernel_surface(sequence, variable, amplitudes)
            empirical_mean = true_kernel_surface(
                sequence, variable, train_values
            ).mean(axis=1, keepdims=True)
            centered_truth = truth - empirical_mean
            if lag_count == config["lag_representation_repair"]["identity_reference"]:
                lag_basis = identity_lag_basis(config["common"]["L_x"])
            else:
                knots = clamped_knots(
                    0.0,
                    float(config["common"]["L_x"] - 1),
                    lag_count,
                    amplitude_config["degree"],
                )
                lag_basis = evaluate_basis(
                    np.arange(config["common"]["L_x"]),
                    knots,
                    amplitude_config["degree"],
                )
            result = project_tensor_surface(centered_truth, lag_basis, amp_eval)
            worst = max(worst, result.nrmse)
    return worst


def test_e1r_regression_table_matches_frozen_values():
    config = json.loads(
        (ROOT / "configs" / "spectral_v031.yaml").read_text(encoding="utf-8")
    )
    repair = config["lag_representation_repair"]
    for scenario, expected in repair["expected_worst_nrmse"].items():
        for lag_count_text, target in expected.items():
            observed = _scenario_worst_nrmse(scenario, int(lag_count_text))
            assert abs(observed - target) <= repair["regression_tolerance"]
