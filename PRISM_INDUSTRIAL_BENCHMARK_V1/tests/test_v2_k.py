from __future__ import annotations

from prism_benchmark.v2_k import profile_intervals


def test_v2_lag_blocks_are_causal_cover_history_and_grow() -> None:
    intervals = profile_intervals(2, 128, 8)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 128
    assert all(left < right for left, right in intervals)
    assert all(intervals[index][1] == intervals[index + 1][0] for index in range(len(intervals) - 1))
    widths = [right - left for left, right in intervals]
    assert widths[-1] >= widths[0]
