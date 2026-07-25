import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODULE_PATH = PROJECT_ROOT / "tools" / "run_phase1_m8_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("phase1_m8_bootstrap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rank_statistic_is_zero_for_rank1_anchor_only() -> None:
    q = np.array([[0.4, 0.6]])
    anchor = np.array([[1.0, -0.5, 0.25]])
    residual_lag_basis = np.array([[[0.5], [-0.5]]])
    residual_coefficients = np.zeros((1, 1, 3))
    observed = MODULE.rank_statistic(
        q=q,
        m7_coefficients=anchor,
        lag_residual_basis=residual_lag_basis,
        residual_coefficients=residual_coefficients,
        amplitude_grams=[np.eye(3)],
    )
    assert observed.shape == (1,)
    assert observed[0] < 1.0e-12
