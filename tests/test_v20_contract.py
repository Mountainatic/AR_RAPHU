from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V20_ROOT = PROJECT_ROOT / "STAGE1_DUAL_SOLVER_V20_bundle"
sys.path.insert(0, str(V20_ROOT))

from stage1.delay_prior import DiscreteGammaPrior  # noqa: E402
from stage1.model import Stage1TargetDelayKAN  # noqa: E402
from stage1.synthetic import SyntheticDataGenerator  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_all_v20_bundle_checksums_match() -> None:
    lines = (V20_ROOT / "SHA256SUMS.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    checked = 0
    for line in lines:
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        assert sha256_file(V20_ROOT / relative_path) == expected
        checked += 1
    assert checked == 64


def test_v20_gamma_is_point_discrete_shape_scale_with_softmax() -> None:
    prior = DiscreteGammaPrior(
        2,
        32,
        delta=1e-3,
        parametrization="mean_std",
    )
    alpha, beta_scale = prior.get_alpha_beta()
    tau = torch.arange(32, dtype=alpha.dtype)
    logits = (
        (alpha[:, None] - 1.0) * torch.log(tau[None, :] + 1e-3)
        - (tau[None, :] + 1e-3) / beta_scale[:, None]
    )
    expected = torch.softmax(logits, dim=-1)

    assert prior.parametrization == "mean_std"
    assert prior.delta == 1e-3
    assert prior.alpha_min == 0.5
    assert prior.alpha_max == 10.0
    assert prior.beta_min == 0.1
    assert prior.beta_max == 32.0
    assert prior.mean_min == 0.0
    assert prior.mean_max == 31.0
    assert prior.std_min == 0.5
    assert prior.std_max == 16.0
    assert torch.allclose(prior(), expected)
    assert torch.allclose(prior().sum(dim=-1), torch.ones(2))


def test_v20_formal_fast_kan_architecture_contract() -> None:
    ranges = [(-2.0, 2.0)] * 10
    model = Stage1TargetDelayKAN(
        10,
        32,
        hidden_kan=8,
        kan_grid_size=7,
        kan_spline_order=3,
        epsilon=0.0,
        input_grid_ranges=ranges,
    )

    assert model.delay_mode == "static_gamma"
    assert model.scorer is None
    assert model.delay_prior is not None
    assert model.delay_prior.parametrization == "mean_std"
    assert len(model.response_branches.branches) == 10
    assert model.response_branches.input_grid_ranges == ranges
    assert model.response_branches.second_layer_grid_range == (-3.0, 3.0)

    branch = model.response_branches.branches[0]
    first, second = branch[0], branch[1]
    assert (first.in_features, first.out_features) == (1, 8)
    assert (second.in_features, second.out_features) == (8, 1)
    assert first.grid_size == second.grid_size == 7
    assert first.spline_order == second.spline_order == 3
    assert first.scale_base == second.scale_base == 1.0
    assert first.scale_spline == second.scale_spline == 1.0


def test_v20_synthetic_alignment_is_horizon_zero_and_not_v2_direct_forecast() -> None:
    generator = SyntheticDataGenerator(
        n_active=3,
        n_inactive=0,
        max_lag=8,
        n_samples=16,
        noise_std=0.0,
        seed=7,
        scenario="S0_oracle",
        burn_in=20,
    )
    x, _, truth = generator.generate(return_debug=True)

    assert truth["horizon"] == 0
    assert np.array_equal(
        truth["target_raw_indices"],
        truth["window_raw_indices"][:, -1],
    )
    assert x.shape[-1] == 8


def test_v20_model_cannot_represent_unequal_Lx_Ly_without_wrapper() -> None:
    model = Stage1TargetDelayKAN(10, 32)

    assert model.max_lag == 32
    assert model._fixed_q.shape == (10, 32)
    assert model.delay_prior.max_lag == 32
