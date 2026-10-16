"""Load explicit A candidates without changing the inherited family/penalties."""
import json
from pathlib import Path
from .invariants import ProtocolError,file_sha


def load_a_registry(path,expected_sha):
    path=Path(path)
    if not expected_sha or file_sha(path)!=expected_sha:
        raise ProtocolError('A temporal registry SHA256 changed')
    registry=json.loads(path.read_text(encoding='utf-8'))
    if registry.get('schema')!='CZ_H1_A_TEMPORAL_REGISTRY_POSTHOC_V1':
        raise ProtocolError('Unknown A registry schema')
    profiles=[tuple(map(int,profile)) for profile in registry['profiles_in_registered_order']]
    if not profiles or len(profiles)!=len(set(profiles)) or any(d<1 or h<d for d,h in profiles):
        raise ProtocolError('A registry profiles are invalid or duplicated')
    if registry['maximum_history_steps']!=max(h for _,h in profiles) or registry['maximum_lags']<1:
        raise ProtocolError('A registry maximum support differs from its candidates')
    if registry['missing_history_zero_fill'] is not False:
        raise ProtocolError('Formal A may not zero-fill nonexistent residual histories')
    if registry['scientific_provenance']['historical_target_test_used_for_protocol_design'] is not True:
        raise ProtocolError('A registry must retain historical test-exposure provenance')
    if registry['training_state_role']==registry['inference_state_role']:
        raise ProtocolError('A training/inference residual state roles must differ')
    return registry,profiles
