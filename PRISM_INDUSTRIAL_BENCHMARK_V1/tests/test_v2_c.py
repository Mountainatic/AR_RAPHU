from __future__ import annotations

import numpy as np

from prism_benchmark.v2_c import _pair_columns, _qr_columns


def test_joint_qr_cap_and_pair_anova_orthogonality() -> None:
    rng = np.random.default_rng(8)
    design = rng.normal(size=(1000, 30))
    selected = _qr_columns(design, 12, 1e-8)
    assert len(selected) == 12
    scalar = rng.normal(size=(1000, 3))
    pair, evaluation, contract = _pair_columns(scalar, scalar, (0, 1), 4)
    assert pair.shape[1] <= 16
    assert evaluation.shape == pair.shape
    assert contract["orthogonality"]["pass"]
