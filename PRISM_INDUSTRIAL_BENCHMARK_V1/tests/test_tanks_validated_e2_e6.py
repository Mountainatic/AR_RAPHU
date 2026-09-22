from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tanks_validated_e2_e6.py"
    spec = importlib.util.spec_from_file_location("tanks_validated_e2_e6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_tanks_sample_uses_exact_h16_target() -> None:
    module = _module()
    values = np.arange(1200, dtype=np.float64)
    samples = module.build_samples(values / 10.0, values)
    assert np.all(samples.target_times - samples.origins == 16)
    assert np.array_equal(samples.target, values[samples.origins + 16])
    assert np.array_equal(samples.blocks["K:16"][:, 0], values[samples.origins])


def test_tanks_outer_folds_have_disjoint_dependency_intervals() -> None:
    module = _module()
    values = np.sin(np.arange(1200, dtype=np.float64) / 20.0)
    samples = module.build_samples(values, values)
    for train, validation in module.folds(samples):
        train_dependency_stop = samples.target_times[train].max() + 1
        validation_dependency_start = (
            samples.origins[validation].min() - module.MAX_HISTORY + 1
        )
        assert train_dependency_stop <= validation_dependency_start


def test_family_designs_are_not_reused_k_only_metrics() -> None:
    module = _module()
    rng = np.random.default_rng(4)
    u = rng.normal(size=1200)
    y = np.cumsum(0.1 * u + rng.normal(scale=0.02, size=1200))
    samples = module.build_samples(u, y)
    k = module.design(samples, module.Route(("K",), 16, 1e-3))
    kc = module.design(samples, module.Route(("K", "C"), 16, 1e-3))
    kcw = module.design(samples, module.Route(("K", "C", "W"), 16, 1e-3))
    assert k.shape[1] < kc.shape[1] < kcw.shape[1]
    assert not np.array_equal(kc, kcw[:, : kc.shape[1] + 1])
