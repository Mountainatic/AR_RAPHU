import numpy as np
import pytest

from prism_benchmark.level_reconstruction import (
    metric_bundle_delta_and_level,
    reconstruct_registered_level,
    reconstruction_identity_audit,
    support_hash,
)


def test_reconstruction_and_metrics_share_residuals() -> None:
    current = np.array([10.0, 11.0, 12.0])
    delta_true = np.array([1.0, -2.0, 0.5])
    delta_pred = np.array([0.5, -1.0, 0.0])

    result = metric_bundle_delta_and_level(delta_true, delta_pred, current)

    np.testing.assert_allclose(
        result["future_level_pred"], current + delta_pred
    )
    assert result["mse"] == pytest.approx(result["mse_delta"])
    assert result["rmse"] == pytest.approx(result["rmse_delta"])
    assert result["mae"] == pytest.approx(result["mae_delta"])
    assert result["r2_level_reconstructed"] != result["r2_delta"]


def test_reconstruction_validates_independently_observed_future_level() -> None:
    current = np.array([10.0, 11.0, 12.0])
    delta_true = np.array([1.0, -2.0, 0.5])
    future = np.array([11.0, 9.0, 12.5])

    result = metric_bundle_delta_and_level(
        delta_true,
        [0.5, -1.0, 0.0],
        current,
        future_level_true=future,
    )

    np.testing.assert_array_equal(result["future_level_true"], future)
    assert result["target_identity_max_abs_error"] == pytest.approx(0.0)
    assert result["residual_identity_max_abs_error"] == pytest.approx(0.0)


def test_reconstruction_rejects_inconsistent_registered_future_level() -> None:
    with pytest.raises(
        AssertionError,
        match="STOP_LEVEL_RECONSTRUCTION_TARGET_IDENTITY_FAILED",
    ):
        metric_bundle_delta_and_level(
            [1.0, -2.0],
            [0.5, -1.0],
            [10.0, 11.0],
            future_level_true=[11.0, 9.1],
        )


def test_reconstruction_rejects_mismatched_support_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        reconstruct_registered_level([1.0], [1.0, 2.0])


def test_identity_audit_is_reporting_only() -> None:
    audit = reconstruction_identity_audit(
        [0.0, 1.0],
        [0.0, 0.5],
        [10.0, 11.0],
    )

    assert audit["status"] == "PASS"
    assert audit["model_retrained"] is False
    assert audit["model_reselected"] is False
    assert audit["sample_support_changed"] is False


def test_support_hash_is_deterministic_and_order_sensitive() -> None:
    left = support_hash(["a", "b"])
    right = support_hash(["a", "b"])
    reversed_hash = support_hash(["b", "a"])

    assert left == right
    assert left != reversed_hash
