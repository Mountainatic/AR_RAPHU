from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from STAGE1_DUAL_SOLVER_V20_bundle.stage1.model import (  # noqa: E402
    Stage1TargetDelayKAN,
)
from ar_raphu.model import (  # noqa: E402
    ARGON_FLOW_CHANNEL,
    ARRAPHURank1,
)


X_RANGES = [(-2.0, 2.0)] * 9
Y_RANGE = (-3.0, 3.0)


def build_xar(*, L_x: int = 64, L_y: int = 8) -> ARRAPHURank1:
    return ARRAPHURank1(
        track="XAR",
        horizon=5,
        L_x=L_x,
        L_y=L_y,
        input_grid_ranges_x=X_RANGES,
        input_grid_range_y=Y_RANGE,
        response_execution_mode="legacy",
    )


def test_m6_free_static_external_kernel_is_normalized_and_ar_stays_gamma() -> None:
    model = ARRAPHURank1(
        track="XAR",
        horizon=1,
        L_x=64,
        L_y=8,
        input_grid_ranges_x=X_RANGES,
        input_grid_range_y=Y_RANGE,
        external_delay_mode="free_static_logits",
        response_execution_mode="legacy",
    )
    assert model.external_branch.delay_prior is None
    assert model.external_branch.delay_logits.shape == (9, 64)
    assert model.ar_branch.delay_prior is not None
    q = model.external_branch._static_q()
    torch.testing.assert_close(q.sum(dim=1), torch.ones(9))


def test_xar_uses_two_exact_v20_cores_with_unequal_windows() -> None:
    model = build_xar(L_x=64, L_y=8)

    assert isinstance(model.external_branch, Stage1TargetDelayKAN)
    assert isinstance(model.ar_branch, Stage1TargetDelayKAN)
    assert model.external_branch.max_lag == 64
    assert model.ar_branch.max_lag == 8
    assert model.external_branch.delay_prior.max_lag == 64
    assert model.ar_branch.delay_prior.max_lag == 8
    assert model.external_branch.scorer is None
    assert model.ar_branch.scorer is None
    assert not model.external_branch.bias.requires_grad
    assert not model.ar_branch.bias.requires_grad

    x = torch.randn(4, 9, 64)
    y = torch.randn(4, 8)
    prediction, aux = model(x, y)

    assert prediction.shape == (4, 1)
    assert aux["external"]["q"].shape == (4, 9, 64)
    assert aux["ar"]["q"].shape == (4, 1, 8)
    assert torch.allclose(
        aux["external"]["q"].sum(dim=-1), torch.ones(4, 9)
    )
    assert torch.allclose(aux["ar"]["q"].sum(dim=-1), torch.ones(4, 1))


def test_non_cz_tracks_can_use_other_channel_counts_without_argon_mask() -> None:
    model = ARRAPHURank1(
        track="X",
        horizon=1,
        external_channels=10,
        inactive_external_channels=(),
        L_x=32,
        input_grid_ranges_x=[(-2.0, 2.0)] * 10,
        response_execution_mode="legacy",
    )
    _, aux = model(torch.randn(2, 10, 32))

    assert model.external_branch.num_variables == 10
    assert aux["component_contribution"].shape == (2, 11)
    assert torch.all(aux["active_mask"][:10])


@pytest.mark.parametrize("track", ["X", "AR", "XAR"])
def test_contribution_closes_exactly_to_prediction(track: str) -> None:
    kwargs = {
        "track": track,
        "horizon": 10,
        "response_execution_mode": "legacy",
    }
    x = y = None
    if track in {"X", "XAR"}:
        kwargs.update(L_x=32, input_grid_ranges_x=X_RANGES)
        x = torch.randn(3, 9, 32)
    if track in {"AR", "XAR"}:
        kwargs.update(L_y=4, input_grid_range_y=Y_RANGE)
        y = torch.randn(3, 4)
    model = ARRAPHURank1(**kwargs)
    with torch.no_grad():
        model.bias.fill_(1.25)

    prediction, aux = model(x, y)
    reconstructed = model.bias + aux["component_contribution"].sum(dim=1)

    assert aux["component_contribution"].shape == (3, 10)
    assert torch.equal(prediction.squeeze(-1), reconstructed)


def test_constant_argon_branch_is_hard_masked_in_output_and_gradient() -> None:
    model = ARRAPHURank1(
        track="X",
        horizon=1,
        L_x=32,
        input_grid_ranges_x=X_RANGES,
        response_execution_mode="legacy",
    )
    x = torch.randn(5, 9, 32, requires_grad=True)
    altered = x.detach().clone()
    altered[:, ARGON_FLOW_CHANNEL, :] = 1.0e6

    first, first_aux = model(x, return_aux=True)
    second, second_aux = model(altered, return_aux=True)
    first.sum().backward()

    assert torch.equal(first, second)
    assert torch.count_nonzero(
        first_aux["external_contribution"][:, ARGON_FLOW_CHANNEL]
    ) == 0
    assert torch.count_nonzero(
        second_aux["external_contribution"][:, ARGON_FLOW_CHANNEL]
    ) == 0
    assert torch.count_nonzero(x.grad[:, ARGON_FLOW_CHANNEL, :]) == 0
    assert (
        model.external_branch.delay_prior.raw_mean.grad[ARGON_FLOW_CHANNEL]
        == 0
    )
    assert (
        model.external_branch.delay_prior.raw_std.grad[ARGON_FLOW_CHANNEL]
        == 0
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.external_branch.response_branches.branches[
            ARGON_FLOW_CHANNEL
        ].parameters()
    )


def test_track_inputs_are_isolated_and_irrelevant_branches_do_not_exist() -> None:
    x_model = ARRAPHURank1(
        track="X",
        horizon=30,
        L_x=32,
        input_grid_ranges_x=X_RANGES,
    )
    ar_model = ARRAPHURank1(
        track="AR",
        horizon=30,
        L_y=16,
        input_grid_range_y=Y_RANGE,
    )

    assert x_model.external_branch is not None
    assert x_model.ar_branch is None
    assert ar_model.external_branch is None
    assert ar_model.ar_branch is not None

    with pytest.raises(ValueError, match="must not receive y_window"):
        x_model(torch.randn(2, 9, 32), torch.randn(2, 16))
    with pytest.raises(ValueError, match="must not receive x_window"):
        ar_model(torch.randn(2, 9, 32), torch.randn(2, 16))


def test_train_only_grid_ranges_are_mandatory() -> None:
    with pytest.raises(ValueError, match="fitted on train only"):
        ARRAPHURank1(track="X", horizon=1, L_x=32)
    with pytest.raises(ValueError, match="fitted on train only"):
        ARRAPHURank1(track="AR", horizon=1, L_y=8)


def test_only_frozen_horizons_and_capacity_candidates_are_accepted() -> None:
    with pytest.raises(ValueError, match="horizon"):
        ARRAPHURank1(
            track="X",
            horizon=2,
            L_x=32,
            input_grid_ranges_x=X_RANGES,
        )
    with pytest.raises(ValueError, match="hidden_kan"):
        ARRAPHURank1(
            track="X",
            horizon=1,
            L_x=32,
            input_grid_ranges_x=X_RANGES,
            hidden_kan=16,
        )
    with pytest.raises(ValueError, match="grid_size"):
        ARRAPHURank1(
            track="AR",
            horizon=1,
            L_y=8,
            input_grid_range_y=Y_RANGE,
            grid_size=9,
        )


def _assert_legacy_vectorized_equivalence(device: torch.device) -> None:
    torch.manual_seed(20260725)
    legacy = ARRAPHURank1(
        track="XAR",
        horizon=60,
        L_x=32,
        L_y=8,
        input_grid_ranges_x=X_RANGES,
        input_grid_range_y=Y_RANGE,
        response_execution_mode="legacy",
    ).to(device)
    vectorized = ARRAPHURank1(
        track="XAR",
        horizon=60,
        L_x=32,
        L_y=8,
        input_grid_ranges_x=X_RANGES,
        input_grid_range_y=Y_RANGE,
        response_execution_mode="vectorized",
    ).to(device)
    vectorized.load_state_dict(legacy.state_dict())

    x_legacy = torch.randn(3, 9, 32, device=device, requires_grad=True)
    y_legacy = torch.randn(3, 8, device=device, requires_grad=True)
    x_vectorized = x_legacy.detach().clone().requires_grad_(True)
    y_vectorized = y_legacy.detach().clone().requires_grad_(True)

    prediction_legacy, aux_legacy = legacy(x_legacy, y_legacy)
    prediction_vectorized, aux_vectorized = vectorized(
        x_vectorized, y_vectorized
    )
    prediction_legacy.square().sum().backward()
    prediction_vectorized.square().sum().backward()

    assert torch.allclose(
        prediction_legacy, prediction_vectorized, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        aux_legacy["component_contribution"],
        aux_vectorized["component_contribution"],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(x_legacy.grad, x_vectorized.grad, atol=3e-5, rtol=3e-5)
    assert torch.allclose(y_legacy.grad, y_vectorized.grad, atol=3e-5, rtol=3e-5)


def test_cpu_vectorized_path_matches_legacy_prediction_and_gradient() -> None:
    _assert_legacy_vectorized_equivalence(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_vectorized_path_matches_legacy_prediction_and_gradient() -> None:
    _assert_legacy_vectorized_equivalence(torch.device("cuda"))


def test_dual_branch_sequence_fast_path_matches_materialized_windows() -> None:
    model = build_xar(L_x=64, L_y=8)
    raw_x = torch.randn(9, 70)
    raw_y = torch.randn(14)
    x_windows = raw_x.unfold(-1, 64, 1).permute(1, 0, 2)
    y_windows = raw_y.unfold(-1, 8, 1)

    ordinary, ordinary_aux = model(x_windows, y_windows)
    fast, fast_aux = model.forward_contiguous(raw_x, raw_y)

    assert torch.allclose(ordinary, fast, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        ordinary_aux["component_contribution"],
        fast_aux["component_contribution"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert fast_aux["sequence_fast_path"] is True
