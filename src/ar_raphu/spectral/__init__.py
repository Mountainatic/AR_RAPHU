"""Spectral Predictive-State AR-RAPHU v0.3 core operators."""

from .contracts import ExperimentContract
from .crossfit import CrossFitResult, forward_crossfit
from .design import SpectralDesign, build_spectral_design
from .gram_svd import GramSpectrum, gram_whitened_svd
from .projection import (
    SurfaceProjectionResult,
    identity_lag_basis,
    project_tensor_surface,
)
from .solver import SpectralFit, solve_full_kernel, solve_full_kernel_pcg
from .synthetic_components import (
    SyntheticComponents,
    e2a_component_target,
    e2b_total_external_target,
    replay_synthetic_components,
)

__all__ = [
    "CrossFitResult",
    "ExperimentContract",
    "GramSpectrum",
    "SpectralDesign",
    "SpectralFit",
    "SurfaceProjectionResult",
    "SyntheticComponents",
    "build_spectral_design",
    "e2a_component_target",
    "e2b_total_external_target",
    "forward_crossfit",
    "gram_whitened_svd",
    "identity_lag_basis",
    "project_tensor_surface",
    "replay_synthetic_components",
    "solve_full_kernel",
    "solve_full_kernel_pcg",
]
