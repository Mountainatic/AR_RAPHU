from __future__ import annotations

import numpy as np

from ar_raphu.datasets.base import DynamicDataset
from ar_raphu.spectral.pb1_development import (
    bootstrap_external_rank_spectrum,
    fit_pb1_shared_history_spectral,
    simulate_pb1_free_run,
)


def _dataset() -> DynamicDataset:
    rng = np.random.default_rng(4)
    records = []
    for phase in range(6):
        x = rng.uniform(-1.0, 1.0, size=140)
        y = np.zeros(140)
        for t in range(2, len(y)):
            y[t] = (
                0.6 * y[t - 1]
                - 0.1 * y[t - 2]
                + 0.5 * x[t - 1]
                + 0.2 * x[t - 2] ** 2
                + 0.02 * rng.normal()
            )
        records.append((x, y))
    n = len(records[0][0])
    return DynamicDataset(
        x=np.concatenate([record[0] for record in records])[:, None],
        y=np.concatenate([record[1] for record in records])[:, None],
        timestamps=np.arange(len(records) * n, dtype=np.float64),
        sequence_id=np.concatenate(
            [
                np.full(
                    n,
                    f"{phase:04d}:Est-phase-{phase}-amp-0",
                    dtype=object,
                )
                for phase in range(len(records))
            ]
        ),
        split=np.concatenate(
            [
                np.full(
                    n,
                    "train" if phase < 4 else "validation",
                    dtype=object,
                )
                for phase in range(len(records))
            ]
        ),
        label_mask=np.ones((len(records) * n, 1), dtype=bool),
        quality_mask=np.ones((len(records) * n, 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={"dataset_id": "pwh"},
    )


def test_pb1_spectral_adapter_is_no_test_cpu_fp64_and_rank_after_selection() -> None:
    fit = fit_pb1_shared_history_spectral(
        _dataset(),
        L_x=4,
        L_y=3,
        amplitude_count=5,
        grid_points=3,
        maximum_expansions=0,
    )
    assert fit.selected.coefficients.dtype == np.float64
    assert fit.selected.relative_kkt_residual <= 1.0e-8
    assert len(fit.candidates) == 64
    assert fit.rank_audit["structural_rank_claim_allowed"] is False
    assert fit.rank_audit["predictive_svd_rank_claim_allowed"] is True
    assert "relative_loss_inflation" in fit.rank_audit
    assert fit.rank_audit["selected_rank_5pct"] >= 1
    bootstrap = bootstrap_external_rank_spectrum(
        fit, replicates=5, seed=12
    )
    assert bootstrap.replicates == 5
    assert bootstrap.maximum_relative_kkt_residual <= 1.0e-8
    assert sum(
        bootstrap.spectral_tail_budget_rank_frequencies["0.05"].values()
    ) == 5


def test_pb1_spectral_track_isolation() -> None:
    for track in ("X", "AR"):
        fit = fit_pb1_shared_history_spectral(
            _dataset(),
            L_x=4,
            L_y=3,
            amplitude_count=5,
            grid_points=3,
            maximum_expansions=0,
            track=track,
        )
        assert fit.track == track
        assert (fit.x_block is not None) is (track == "X")
        assert (fit.ar_block is not None) is (track == "AR")
        if track == "AR":
            assert fit.rank_audit["status"] == "NOT_APPLICABLE"


def test_pb1_spectral_free_run_is_separate_and_validation_only() -> None:
    dataset = _dataset()
    fit = fit_pb1_shared_history_spectral(
        dataset,
        L_x=4,
        L_y=3,
        amplitude_count=5,
        grid_points=3,
        maximum_expansions=0,
    )
    result = simulate_pb1_free_run(dataset, fit)
    assert result.status == "COMPLETED"
    assert result.initialization_length == 3
    assert result.scored_samples > 0
    assert np.isfinite(result.rmse_standardized)
