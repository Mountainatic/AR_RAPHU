import numpy as np
import pytest

from ar_raphu.datasets.splits import (
    development_tail_split,
    validate_split_integrity,
)

from public_dataset_fixtures import make_public_dataset


def test_ordered_official_split_is_verified():
    result = validate_split_integrity(make_public_dataset())
    assert result["split_verified"] is True
    assert result["train_count"] == 10
    assert result["validation_count"] == 3
    assert result["test_count"] == 3


def test_split_regression_inside_record_is_rejected():
    split = np.array(["train"] * 8 + ["test"] * 3 + ["validation"] * 5)
    with pytest.raises(ValueError, match="regresses"):
        validate_split_integrity(make_public_dataset(split=split))


def test_development_tail_split_is_time_ordered():
    split = development_tail_split(20, validation_fraction=0.2)
    assert list(split[:16]) == ["train"] * 16
    assert list(split[16:]) == ["validation"] * 4
