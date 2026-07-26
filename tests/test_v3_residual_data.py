from __future__ import annotations

import numpy as np
import torch

from ar_raphu.diagnostics.residual_data import PreparedExternalResidualData
from ar_raphu.sequence_data import PreparedDirectForecastData
from ar_raphu.synthetic import generate_synthetic_sequence


def test_residual_dataset_preserves_target_origin_and_x_alignment() -> None:
    sequence = generate_synthetic_sequence(
        "AR-S3", seed=2, n_samples=160, external_variables=10, snr_db=40
    )
    base = PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track="XAR",
        horizon=1,
        L_x=64,
        L_y=32,
        split_target_intervals=sequence.split_target_intervals,
    )
    predictions = {}
    for partition, (start, stop) in base.split_target_intervals.items():
        predictions[partition] = base.y_scaled[start:stop] - 0.25
    residual = PreparedExternalResidualData.from_ar_predictions(base, predictions)

    base_batch = next(
        base.iter_contiguous_batches(
            "validation", batch_size=23, device=torch.device("cpu")
        )
    )
    residual_batch = next(
        residual.iter_contiguous_batches(
            "validation", batch_size=23, device=torch.device("cpu")
        )
    )
    assert torch.equal(
        residual_batch["target_index"], base_batch["target_index"]
    )
    assert torch.equal(
        residual_batch["origin_index"], base_batch["origin_index"]
    )
    assert torch.equal(
        residual_batch["target_index"], residual_batch["origin_index"] + 1
    )
    torch.testing.assert_close(
        residual_batch["x_sequence"], base_batch["x_sequence"]
    )
    torch.testing.assert_close(
        residual_batch["target"], torch.full((23,), 0.25)
    )
    assert not hasattr(residual, "y_window_view")
    assert np.isclose(residual.scaler.y_scale, base.scaler.y_scale)
