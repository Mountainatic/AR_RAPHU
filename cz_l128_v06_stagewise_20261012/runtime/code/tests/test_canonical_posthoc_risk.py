import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.risk import *
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError,NumericallyInvalid


def test_routing_and_model_ties_use_identical_numeric_bound():
    y=np.linspace(-2.,2.,100)
    p=np.ones(100)
    q=np.nextafter(p,np.zeros(100))
    a=mse_evidence(y,p);b=mse_evidence(y,q)
    route=paired_route(y,p,q)
    assert route['epsilon_fp']==equivalence_bound(a,b)
    assert registered_best({'registered_first':a,'lower_float_only':b})=='registered_first'
    assert route['routing_status']=='BYPASSED'


def test_small_real_gain_is_not_rejected_by_a_statistical_or_practical_threshold():
    y=np.ones(100);parent=np.zeros(100);child=np.full(100,1e-8)
    route=paired_route(y,parent,child)
    assert route['incremental_gain']>route['epsilon_fp']
    assert route['routing_status']=='ADMITTED'
    assert registered_best({'parent_like':mse_evidence(y,parent),'child':mse_evidence(y,child)})=='child'


def test_numeric_bounds_scale_with_observed_values_without_unit_floor():
    y=np.array([1.,2.,3.]);p=np.array([.5,1.,1.5])
    large=mse_evidence(y,p);small=mse_evidence(y*1e-12,p*1e-12)
    assert np.isclose(small.value/large.value,1e-24,rtol=1e-12,atol=0.)
    assert np.isclose(small.bound/large.bound,1e-24,rtol=1e-12,atol=0.)


def test_recipe_invalidity_is_distinct_from_bypass_or_equivalent_tie():
    with pytest.raises(NumericallyInvalid):registered_best({})
    with pytest.raises(NumericallyInvalid):mse_evidence([1.],[np.nan])
    with pytest.raises(ProtocolError):
        registered_best({'a':RiskEvidence(1.,1e-15,10),'b':RiskEvidence(1.,1e-15,9)})


def test_aggregate_weights_rows_and_retains_na_identity_observations():
    identity=mse_evidence([1.,2.],[0.,0.]);active=mse_evidence([3.],[2.])
    combined=aggregate([identity,active])
    assert combined.rows==3 and combined.value==(1.+4.+1.)/3.
