import numpy as np

from ar_raphu.datasets.scaling import TrainOnlyStandardizer
from ar_raphu.datasets.windowing import build_windowed_task

from public_dataset_fixtures import make_public_dataset


def test_scaler_ignores_validation_and_test_values():
    dataset = make_public_dataset()
    x = dataset.x.copy()
    x[10:, :] += 1_000_000.0
    changed = type(dataset)(
        x=x,
        y=dataset.y,
        timestamps=dataset.timestamps,
        sequence_id=dataset.sequence_id,
        split=dataset.split,
        label_mask=dataset.label_mask,
        quality_mask=dataset.quality_mask,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        metadata=dataset.metadata,
    )
    scaler = TrainOnlyStandardizer.fit(changed)
    np.testing.assert_allclose(scaler.x_mean, dataset.x[:10].mean(axis=0))
    np.testing.assert_allclose(scaler.y_mean, dataset.y[:10].mean(axis=0))
    assert scaler.fitted_row_count == 10


def test_test_range_exceedance_is_flagged_without_clipping():
    dataset = make_public_dataset()
    scaler = TrainOnlyStandardizer.fit(dataset)
    task = build_windowed_task(
        dataset,
        target=0,
        horizon=1,
        L_x=2,
        L_y=2,
        standardizer=scaler,
    )
    assert np.any(task.ood_mask[task.split == "test"])
    transformed = scaler.transform(dataset)
    assert transformed.x[-1, 0] > transformed.x[9, 0]
