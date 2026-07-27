import numpy as np
import pytest

from ar_raphu.datasets.base import DynamicDataset

from public_dataset_fixtures import make_public_dataset


def test_dynamic_dataset_contract_accepts_multiple_contiguous_records():
    sequence = np.array([0] * 8 + [1] * 8)
    dataset = make_public_dataset(sequence_id=sequence)
    assert dataset.n_time == 16
    assert dataset.n_features == 2
    assert dataset.n_targets == 1


def test_dynamic_dataset_rejects_noncontiguous_sequence_identifier():
    sequence = np.array([0] * 7 + [1] * 8 + [0])
    with pytest.raises(ValueError, match="not contiguous"):
        make_public_dataset(sequence_id=sequence)


def test_dynamic_dataset_rejects_quality_valid_nonfinite_value():
    base = make_public_dataset()
    x = base.x.copy()
    x[2, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        DynamicDataset(
            x=x,
            y=base.y,
            timestamps=base.timestamps,
            sequence_id=base.sequence_id,
            split=base.split,
            label_mask=base.label_mask,
            quality_mask=base.quality_mask,
            feature_names=base.feature_names,
            target_names=base.target_names,
            metadata=base.metadata,
        )
