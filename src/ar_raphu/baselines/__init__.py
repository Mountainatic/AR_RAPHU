"""Literature-grounded public-benchmark baselines."""

from .arx_champneys2024 import (
    ARXHistorySelection,
    fit_and_select_arx_history,
)
from .pnarx_champneys2024 import (
    PNARXSelection,
    fit_and_select_pnarx,
)
from .mlp_narx_champneys2024 import (
    MLPTrainingResult,
    MLPWeights,
    train_mlp_narx,
)

__all__ = [
    "ARXHistorySelection",
    "PNARXSelection",
    "MLPTrainingResult",
    "MLPWeights",
    "fit_and_select_arx_history",
    "fit_and_select_pnarx",
    "train_mlp_narx",
]
