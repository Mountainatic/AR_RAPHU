import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.baselines import (
    LinearDirectForecaster,
    linear_design,
    persistence_predict,
)
from ar_raphu.sequence_data import PreparedDirectForecastData


def make_data(horizon: int = 1) -> PreparedDirectForecastData:
    time = np.arange(160, dtype=np.float64)
    x = np.column_stack((time, 2.0 * time + 1.0))
    y = 3.0 * time - 4.0
    return PreparedDirectForecastData.from_sequence(
        x,
        y,
        track="XAR",
        horizon=horizon,
        L_x=4,
        L_y=3,
        split_target_intervals={
            "train": (20, 100),
            "validation": (100, 130),
            "test": (130, 160),
        },
    )


def test_persistence_uses_forecast_origin_not_target() -> None:
    data = make_data(horizon=5)
    prediction = persistence_predict(data, "test")
    targets = np.arange(130, 160)
    expected = data.y_scaled[targets - 5]
    np.testing.assert_allclose(prediction, expected)


def test_linear_design_is_current_to_past_and_target_aligned() -> None:
    data = make_data(horizon=5)
    design, target, indices = linear_design(data, "validation", kind="ARX")
    origin = indices[0] - 5
    np.testing.assert_allclose(
        design[0, :3], data.y_scaled[origin - 2 : origin + 1][::-1]
    )
    np.testing.assert_allclose(target, data.y_scaled[indices])
    assert not np.any(indices <= origin)


def test_linear_ar_fits_training_only_and_predicts_affine_sequence() -> None:
    data = make_data()
    model = LinearDirectForecaster.fit(data, kind="AR")
    prediction, indices = model.predict(data, "test")
    np.testing.assert_allclose(
        prediction, data.y_scaled[indices], atol=2e-7, rtol=2e-7
    )
    assert model.rank <= data.L_y + 1
