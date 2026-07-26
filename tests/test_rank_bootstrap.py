import numpy as np

from ar_raphu.spectral.rank_inference import (
    bootstrap_upper_tail_pvalue,
    circular_moving_block_indices,
)


def test_circular_blocks_preserve_within_block_adjacency():
    indices = circular_moving_block_indices(
        101, block_length=16, rng=np.random.default_rng(2)
    )
    assert indices.shape == (101,)
    for start in range(0, 96, 16):
        assert np.all((np.diff(indices[start : start + 16]) % 101) == 1)


def test_bootstrap_pvalue_has_plus_one_correction():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert bootstrap_upper_tail_pvalue(0.35, values) == 2 / 5
