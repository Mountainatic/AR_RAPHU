from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.model import ARRAPHURank1  # noqa: E402
from ar_raphu.sequence_data import PreparedDirectForecastData  # noqa: E402
from ar_raphu.synthetic import generate_synthetic_sequence  # noqa: E402
from ar_raphu.training import (  # noqa: E402
    evaluate_rmse,
    free_lag_logit_roughness,
    prune_external_path,
    refit_fixed_external_support,
    seed_everything,
    train_dense_warmup,
)


def test_free_lag_logit_roughness_zero_for_linear_logits() -> None:
    data = prepared_small()
    x_ranges, y_range = data.scaler.input_grid_ranges(
        data.x_scaled * data.scaler.x_scale + data.scaler.x_mean,
        data.y_scaled * data.scaler.y_scale + data.scaler.y_mean,
    )
    model = ARRAPHURank1(
        track="X",
        horizon=1,
        external_channels=10,
        inactive_external_channels=(),
        L_x=64,
        input_grid_ranges_x=x_ranges,
        hidden_kan=4,
        grid_size=5,
        external_delay_mode="free_static_logits",
    )
    with torch.no_grad():
        line = torch.linspace(-1.0, 1.0, 64)
        model.external_branch.delay_logits.copy_(line.expand(10, -1))
    torch.testing.assert_close(
        free_lag_logit_roughness(model), torch.tensor(0.0), atol=1e-10, rtol=0
    )


def prepared_small(track: str = "XAR") -> PreparedDirectForecastData:
    sequence = generate_synthetic_sequence(
        "AR-S1", seed=11, n_samples=160, external_variables=10, snr_db=40
    )
    return PreparedDirectForecastData.from_sequence(
        sequence.x,
        sequence.y_observed,
        track=track,
        horizon=1,
        L_x=64,
        L_y=32,
        split_target_intervals=sequence.split_target_intervals,
    )


def build_small(data: PreparedDirectForecastData) -> ARRAPHURank1:
    x_ranges, y_range = data.scaler.input_grid_ranges(
        data.x_scaled * data.scaler.x_scale + data.scaler.x_mean,
        data.y_scaled * data.scaler.y_scale + data.scaler.y_mean,
    )
    return ARRAPHURank1(
        track=data.track,
        horizon=1,
        external_channels=10,
        inactive_external_channels=(),
        L_x=64 if data.track in {"X", "XAR"} else None,
        L_y=32 if data.track in {"AR", "XAR"} else None,
        input_grid_ranges_x=x_ranges if data.track in {"X", "XAR"} else None,
        input_grid_range_y=y_range if data.track in {"AR", "XAR"} else None,
        hidden_kan=4,
        grid_size=5,
        response_execution_mode="vectorized",
    )


def test_prepared_batches_end_at_origin_before_target() -> None:
    data = prepared_small()
    batch = next(
        data.iter_batches("validation", batch_size=32, device=torch.device("cpu"))
    )

    assert batch["x_window"].shape[1:] == (10, 64)
    assert batch["y_window"].shape[1:] == (32,)
    assert torch.equal(batch["target_index"], batch["origin_index"] + 1)
    fast_batch = next(
        data.iter_contiguous_batches(
            "validation", batch_size=32, device=torch.device("cpu")
        )
    )
    assert fast_batch["x_sequence"].shape == (10, 32 + 64 - 1)
    assert fast_batch["y_sequence"].shape == (32 + 32 - 1,)


def test_dense_training_and_external_pruning_protocol_execute() -> None:
    data = prepared_small()
    device = torch.device("cpu")
    seed_everything(0, deterministic=True)
    model = build_small(data).to(device)
    initial = evaluate_rmse(
        model, data, "validation", batch_size=128, device=device
    )
    warmup = train_dense_warmup(
        model,
        data,
        epochs=3,
        learning_rate=0.003,
        patience=3,
        batch_size=128,
        device=device,
        validation_interval=1,
    )
    assert np.isfinite(warmup.best_validation_rmse)
    assert warmup.best_validation_rmse <= initial

    norms = model.external_branch.response_branches.compute_branch_norms()
    pruning_model = build_small(data).to(device)
    pruned = prune_external_path(
        pruning_model,
        data,
        warmup.best_state,
        requested_scale=0.003,
        median_warmup_branch_norm=float(norms.median()),
        epochs=3,
        learning_rate=0.003,
        ramp_epochs=1,
        full_penalty_min_epochs=1,
        stable_epochs=1,
        batch_size=128,
        device=device,
        validation_interval=1,
    )
    assert np.isfinite(pruned.best_validation_rmse)
    assert set(pruned.terminal_support).issubset(set(range(10)))
    refit_model = build_small(data).to(device)
    refit = refit_fixed_external_support(
        refit_model,
        data,
        pruned.terminal_state,
        pruned.terminal_support,
        epochs=2,
        learning_rate=0.003,
        patience=2,
        batch_size=128,
        device=device,
        validation_interval=1,
    )
    assert refit.terminal_support == sorted(pruned.terminal_support)
    assert np.isfinite(refit.best_validation_rmse)
