import numpy as np

from ar_raphu.datasets.windowing import build_windowed_task

from public_dataset_fixtures import make_public_dataset


def test_direct_windows_end_exactly_at_origin_before_target():
    dataset = make_public_dataset()
    horizon = 3
    task = build_windowed_task(
        dataset,
        target=0,
        horizon=horizon,
        L_x=5,
        L_y=4,
    )
    assert np.all(task.origin_index + horizon == task.target_index)
    assert np.all(task.origin_index < task.target_index)
    for row, origin in enumerate(task.origin_index):
        np.testing.assert_array_equal(
            task.x_history[row, 0],
            dataset.x[origin - np.arange(5), 0],
        )


def test_unobserved_target_rows_are_never_training_examples():
    dataset = make_public_dataset()
    label_mask = dataset.label_mask.copy()
    label_mask[12, 0] = False
    replaced = type(dataset)(
        x=dataset.x,
        y=dataset.y,
        timestamps=dataset.timestamps,
        sequence_id=dataset.sequence_id,
        split=dataset.split,
        label_mask=label_mask,
        quality_mask=dataset.quality_mask,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        metadata=dataset.metadata,
    )
    task = build_windowed_task(
        replaced,
        target=0,
        horizon=1,
        L_x=2,
        L_y=2,
    )
    assert 12 not in task.target_index
