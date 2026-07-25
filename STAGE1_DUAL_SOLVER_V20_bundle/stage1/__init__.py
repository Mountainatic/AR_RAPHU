from .delay_prior import DiscreteGammaPrior
from .scorer import BoundedLagScorer
from .response_kan import UnivariateKANResponse
from .proximal import ProximalGroupLasso, apply_group_proximal_step
from .losses import smoothness_loss_from_coeffs, compute_smoothness_loss, total_loss
from .model import Stage1TargetDelayKAN
from .utils import Stage1Config, set_seed_stage1
from .synthetic import SyntheticDataGenerator, generate_all_scenarios
from .train import Stage1Trainer
from .evaluate import evaluate_model, rolling_window_validate
from .analysis import Stage1Analyzer

# V20 dual-solver additions
from .sequence_ops import ContiguousWindowBatch
from .independent_path import run_independent_fork
from .variational_v20 import (
    VariationalDistributedLagSplineV20,
    solve_fixed_q_fista_v20,
    fit_variational_v20,
)
