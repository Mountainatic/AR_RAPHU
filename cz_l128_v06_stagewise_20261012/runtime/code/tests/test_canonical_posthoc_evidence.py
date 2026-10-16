import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.evidence import report_evidence
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError


def procedure(seed=20260804):
    return dict(replicates=500,quantiles=[.025,.975],block_lengths_steps=[2,4,64],
        seed=seed,complete_entity_resampling=True,selection_authority=False)


def pointwise(n=320):
    origin=np.arange(1,2*n+1)
    child=np.r_[np.zeros(n),np.ones(n)*.1]
    return pd.DataFrame(dict(entity_id='Rod_1_segment_0',origin=origin,
        fold=np.repeat([0,1],n),y_true=np.ones(2*n),parent=np.zeros(2*n),MODEL_child=child))


def test_evidence_keeps_na_identity_fold_and_never_changes_routing():
    frame=pointwise()
    first=report_evidence(frame,procedure())
    other=report_evidence(frame,procedure(seed=20261003))
    assert first['paired_rows']==640 and len(first['fold_gains'])==2
    assert first['fold_gains'][0]['gain']==0.
    assert first['paired_risk']==other['paired_risk']
    assert not first['route_changed_by_report'] and first['reporting_only']
    assert all(result['replicates']==500 for result in first['bootstrap'])


def test_short_scoring_run_is_reported_na_without_changing_frozen_block():
    report=report_evidence(pointwise(10),procedure())
    assert report['bootstrap'][-1]['block_steps']==64
    assert report['bootstrap'][-1]['status']=='NOT_APPLICABLE'
    assert report['paired_rows']==20


def test_reporter_cannot_claim_selection_authority():
    spec=procedure();spec['selection_authority']=True
    with pytest.raises(ProtocolError,match='reporting-only'):report_evidence(pointwise(),spec)
