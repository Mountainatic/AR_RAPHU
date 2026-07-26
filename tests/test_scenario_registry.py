from ar_raphu.spectral.scenario_registry import (
    s4c_mixed_difference,
    scenario_spec,
)


def test_s4c_is_excluded_and_s4u_is_eligible_for_2d_capacity():
    s4c = scenario_spec("AR-S4")
    s4u = scenario_spec("AR-S4U")
    assert s4c.canonical_name == "AR-S4C"
    assert s4c.model_class == "conditional_urysohn_3d"
    assert not s4c.eligible_for_2d_capacity
    assert s4u.eligible_for_2d_capacity


def test_s4c_has_nonzero_cross_lag_mixed_difference():
    assert abs(s4c_mixed_difference()) > 1e-8
