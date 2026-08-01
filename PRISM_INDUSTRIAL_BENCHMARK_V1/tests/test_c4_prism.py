from __future__ import annotations

from prism_benchmark.c4_prism import channel_class, profile_intervals


def test_channel_class_rules_are_semantic_and_deterministic() -> None:
    assert channel_class("tep", "xmv_1") == "FAST"
    assert channel_class("tep", "xmeas_1") == "MEDIUM"
    assert channel_class("pmsm", "ambient") == "SLOW"
    assert channel_class("pmsm", "motor_speed") == "FAST"
    assert channel_class("metropt", "Oil_temperature") == "SLOW"
    assert channel_class("metropt", "Motor_current") == "FAST"


def test_profile_intervals_are_causal_contiguous_and_delta_aligned() -> None:
    intervals = profile_intervals(delta=4, history=64, m_tau=8)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 64
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert all(near % 4 == 0 and far % 4 == 0 for near, far in intervals)
