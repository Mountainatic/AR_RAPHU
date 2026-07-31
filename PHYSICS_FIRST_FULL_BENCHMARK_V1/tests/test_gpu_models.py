from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.gpu_models import build_model, parameter_count, unwrap_output
from src.gpu_data import matured_residual_history


def test_all_configured_architectures_forward():
    config = json.loads((ROOT / 'configs' / 'gpu_models.yaml').read_text())
    seen = set()
    for spec in config['models']:
        key = (spec['architecture'], spec['mode'], json.dumps(spec.get('parameters', {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        input_dim = 1 if spec['mode'] == 'residual' else (5 if spec['mode'] == 'dynamic' else 4)
        model = build_model(
            spec['architecture'],
            sequence_length=240,
            input_dim=input_dim,
            parameters=spec.get('parameters', {}),
        )
        assert parameter_count(model) <= spec.get('max_parameters', config['max_parameters'])
        x = torch.randn(2, 240, input_dim)
        output = unwrap_output(model(x))
        assert output.prediction.shape == (2,)
        assert torch.isfinite(output.prediction).all()
        if output.auxiliary_loss is not None:
            assert torch.isfinite(output.auxiliary_loss)


def test_residual_history_uses_only_matured_rows():
    target = np.arange(20, dtype=float)
    prediction = np.zeros(20, dtype=float)
    history, available = matured_residual_history(
        target, prediction, maturity_rows=3, history_rows=4
    )
    assert not available[:6].any()
    assert available[6]
    np.testing.assert_array_equal(history[6, :, 0], np.array([0., 1., 2., 3.], dtype=np.float32))
