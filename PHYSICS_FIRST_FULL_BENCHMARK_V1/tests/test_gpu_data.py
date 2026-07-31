from __future__ import annotations
import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_data import Standardizer, TargetScaler, chronological_folds


def test_standardizers_fit_train_only_shape():
    x = np.arange(10 * 4 * 3, dtype=np.float32).reshape(10, 4, 3)
    scaler = Standardizer.fit(x[:6], axes=(0, 1))
    transformed = scaler.transform(x)
    assert transformed.shape == x.shape
    np.testing.assert_allclose(transformed[:6].mean(axis=(0, 1)), 0.0, atol=1e-6)
    y = np.linspace(-2, 3, 10)
    target = TargetScaler.fit(y[:6])
    restored = target.inverse(target.transform(y))
    np.testing.assert_allclose(restored, y, atol=1e-6)


def test_chronological_folds_have_purge_gap():
    folds = list(chronological_folds(1000, [[0.5, 0.6], [0.6, 0.7]], 50))
    assert len(folds) == 2
    for train, valid in folds:
        assert train.max() + 50 < valid.min() + 1
