"""Spectral Predictive-State AR-RAPHU v0.3 core operators."""

from .contracts import ExperimentContract
from .crossfit import CrossFitResult, forward_crossfit
from .design import SpectralDesign, build_spectral_design
from .gram_svd import GramSpectrum, gram_whitened_svd
from .solver import SpectralFit, solve_full_kernel
from .synthetic_components import SyntheticComponents, replay_synthetic_components

__all__ = [
    "CrossFitResult",
    "ExperimentContract",
    "GramSpectrum",
    "SpectralDesign",
    "SpectralFit",
    "SyntheticComponents",
    "build_spectral_design",
    "forward_crossfit",
    "gram_whitened_svd",
    "replay_synthetic_components",
    "solve_full_kernel",
]
