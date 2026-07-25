import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.statistics import (
    benjamini_hochberg,
    moving_block_indices,
    paired_moving_block_rmse_difference,
    residual_acf_block_length,
)


def test_white_noise_block_length_hits_registered_minimum() -> None:
    residual = np.array([1.0, -1.0] * 1000)
    # Alternating residuals remain significant until the cap.
    assert residual_acf_block_length(residual, search_cap=7) == 7
    white = np.random.default_rng(2).normal(size=5000)
    assert residual_acf_block_length(white) == 2


def test_moving_blocks_are_contiguous_and_reproducible() -> None:
    first = moving_block_indices(
        20, 4, replicates=10, rng=np.random.default_rng(3)
    )
    second = moving_block_indices(
        20, 4, replicates=10, rng=np.random.default_rng(3)
    )
    np.testing.assert_array_equal(first, second)
    reshaped = first.reshape(10, 5, 4)
    np.testing.assert_array_equal(np.diff(reshaped, axis=-1), 1)


def test_paired_bootstrap_detects_uniformly_better_candidate() -> None:
    rng = np.random.default_rng(4)
    observed = rng.normal(size=1000)
    reference = observed + rng.normal(scale=1.0, size=1000)
    candidate = observed + 0.2 * (reference - observed)
    result = paired_moving_block_rmse_difference(
        observed,
        reference,
        candidate,
        replicates=500,
        rng=np.random.default_rng(5),
    )
    assert result.observed_loss_difference > 0
    assert result.confidence_interval_95[0] > 0
    assert result.one_sided_p_value <= 1 / 501


def test_bh_adjustment_is_monotone_in_ranked_order() -> None:
    p = np.array([0.001, 0.03, 0.01, 0.9])
    reject, adjusted = benjamini_hochberg(p, q=0.05)
    np.testing.assert_array_equal(reject, [True, True, True, False])
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)

