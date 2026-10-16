import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.data import load_development,load_inference
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError


@pytest.mark.parametrize('operation',['source_validation','target_test'])
def test_direct_label_helper_cannot_bypass_current_execution_stop(operation):
    context=dict(protocol={'execution_authorization':dict(source_validation_reporting=False,target_test_inference=False)})
    with pytest.raises(ProtocolError,match='stopped'):load_inference(context,operation)


@pytest.mark.parametrize('operation',['source_validation','target_test'])
def test_authorized_inference_still_needs_verified_checkpoint_before_any_read(operation):
    context=dict(mode='inference_only',protocol={'execution_authorization':dict(source_validation_reporting=True,target_test_inference=True)})
    with pytest.raises(ProtocolError,match='verified deployment'):load_inference(context,operation)


@pytest.mark.parametrize('mode',['metadata_only','inference_only',None])
def test_development_labels_cannot_be_loaded_in_a_nonfitting_mode(mode):
    with pytest.raises(ProtocolError,match='fitting mode'):load_development({'mode':mode})
