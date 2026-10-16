import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.a_registry import load_a_registry
from canonical.cz_h1_posthoc_v22.invariants import file_sha,dump,ProtocolError


def test_explicit_a_registry_preserves_values_order_and_honest_posthoc_provenance():
    path=Path(__file__).resolve().parents[1]/'canonical/cz_h1_posthoc_v22/A_TEMPORAL_REGISTRY.json'
    registry,profiles=load_a_registry(path,file_sha(path))
    assert profiles==[(1,h) for h in [8,16,32,64,96,128]]
    assert registry['maximum_lags']==registry['maximum_history_steps']==128
    assert registry['scientific_provenance']['revision_status']=='POST_HOC_PROTOCOL_REVISION'
    assert not registry['scientific_provenance']['virgin_test_confirmatory_claim_permitted']


def test_a_registry_changes_cannot_reuse_old_pin(tmp_path):
    original=Path(__file__).resolve().parents[1]/'canonical/cz_h1_posthoc_v22/A_TEMPORAL_REGISTRY.json'
    data=json.loads(original.read_text(encoding='utf-8'))
    data['profiles_in_registered_order'].append([1,256])
    path=tmp_path/'changed.json';dump(path,data)
    with pytest.raises(ProtocolError,match='SHA256'):load_a_registry(path,file_sha(original))
    with pytest.raises(ProtocolError,match='maximum support'):load_a_registry(path,file_sha(path))
