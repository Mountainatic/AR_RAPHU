import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.temporal_support import *
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError


def metadata(n=1000,entity='Rod_1_segment_0'):
    origin=np.arange(1,n+1)
    return pd.DataFrame(dict(entity_id=entity,origin=origin,split='train',
        dependency_start=np.maximum(0,origin-256),dependency_stop_exclusive=origin+1))


def test_replay_has_no_global_lmax_cut_or_fixed_burnin():
    frame=metadata()
    blocks=final_replay_blocks(frame,320)
    assert len(blocks)==4 and len(blocks[-1][1])==40
    assert len(blocks[0][0])==0 and blocks[0][1][0]==0
    assert np.array_equal(np.concatenate([ev for _,ev in blocks]),np.arange(1000))
    assert frame.iloc[blocks[0][1]].origin.min()==1


def test_entity_reset_keeps_prior_independent_past_without_future_rows():
    frame=pd.concat([metadata(1000),metadata(500,'Rod_1_segment_1')],ignore_index=True)
    blocks=final_replay_blocks(frame,320)
    fit,ev=next((f,e) for f,e in blocks if e[0]==1000)
    assert np.array_equal(fit,np.arange(1000))
    assert len(set(frame.iloc[ev].entity_id))==1
    for fi,ei in blocks:
        assert not len(fi) or fi[-1]<ei[0]
        for entity,g in frame.iloc[ei].groupby('entity_id'):
            matching=frame.iloc[fi].query('entity_id==@entity')
            assert not len(matching) or matching.dependency_stop_exclusive.max()<g.dependency_start.min()


def test_replay_block_length_and_train_partition_are_frozen():
    frame=metadata()
    with pytest.raises(ProtocolError,match='320'):final_replay_blocks(frame,8)
    frame.loc[0,'split']='test'
    with pytest.raises(ProtocolError,match='train'):final_replay_blocks(frame,320)
