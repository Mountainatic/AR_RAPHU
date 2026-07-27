import numpy as np

from ar_raphu.datasets.windowing import build_windowed_task

from public_dataset_fixtures import make_public_dataset


def test_windows_use_current_to_past_orientation_and_target_split():
    dataset = make_public_dataset()
    task = build_windowed_task(
        dataset,
        target=0,
        horizon=2,
        L_x=3,
        L_y=2,
    )
    row = int(np.flatnonzero(task.target_index == 10)[0])
    assert task.origin_index[row] == 8
    np.testing.assert_array_equal(task.x_history[row, 0], [8.0, 7.0, 6.0])
    np.testing.assert_array_equal(task.y_history[row], [108.0, 107.0])
    assert task.target[row] == 110.0
    assert task.split[row] == "validation"


def test_windows_never_cross_sequence_id():
    sequence = np.array([0] * 8 + [1] * 8)
    split = np.array(["train"] * 6 + ["validation"] * 2 + ["test"] * 8)
    dataset = make_public_dataset(split=split, sequence_id=sequence)
    task = build_windowed_task(
        dataset,
        target=0,
        horizon=1,
        L_x=4,
        L_y=4,
    )
    assert not np.any(np.isin(task.target_index, [8, 9, 10, 11]))
    for target_index, origin, sequence_value in zip(
        task.target_index,
        task.origin_index,
        task.sequence_id,
        strict=True,
    ):
        assert dataset.sequence_id[target_index] == sequence_value
        assert np.all(
            dataset.sequence_id[origin - np.arange(4)] == sequence_value
        )
