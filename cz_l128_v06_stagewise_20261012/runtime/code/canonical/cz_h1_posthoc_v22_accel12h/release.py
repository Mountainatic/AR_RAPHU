"""Build a reviewable pinned registration; never fit or authorize a run."""
import json
from pathlib import Path
from .invariants import ProtocolError,file_sha,source_sha,dump,digest
from .environment import fingerprint,verify_environment


def build_release(authority,c1_root,destination,*,accepted_registration,runtime_budget_projection=None):
    authority=Path(authority);c1_root=Path(c1_root);destination=Path(destination)
    if destination.exists():raise ProtocolError('Release namespaces are immutable')
    module=Path(__file__).parent
    sources={p.relative_to(module).as_posix():source_sha(p) for p in module.rglob('*.py') if '__pycache__' not in p.parts}
    protocols={name:file_sha(module/name) for name in ['CANONICAL_PROTOCOL.json','FINAL_PROTOCOL.json','A_TEMPORAL_REGISTRY.json','K_MODEL_REGISTRY.json','C_MODEL_REGISTRY.json','W_MODEL_REGISTRY.json','A_MODEL_REGISTRY.json','IMMUTABLE_BASELINE.json','ACCEL_BASELINE.json','ACCEL_PROTOCOL.json','ACCEL_K_SEARCH_PROTOCOL.json']}
    protocol=json.loads((module/'CANONICAL_PROTOCOL.json').read_text(encoding='utf-8'))
    if protocol['scientific_gate_status']!='FROZEN':raise ProtocolError('Scientific gate is not frozen')
    config=authority/'configs/cz_h1_k_temporal_registry_v2'
    historical=json.loads((config/'AUTHORITY_UNCHANGED_BINDINGS.json').read_text(encoding='utf-8'))
    bindings=dict(historical['files'])
    for relative,expected in bindings.items():
        if source_sha(authority/relative)!=expected:raise ProtocolError('Historical authority binding drift: '+relative)
    oldseal=json.loads((config/'CZ_H1_K_TEMPORAL_REGISTRY_V2_SEAL.json').read_text(encoding='utf-8'))
    for relative,expected in oldseal['runtime_source_bindings'].items():
        if source_sha(authority/relative)!=expected:raise ProtocolError('Frozen K runtime changed: '+relative)
        bindings[relative]=expected
    for path in (authority/'src/prism_benchmark').rglob('*.py'):
        bindings[path.relative_to(authority).as_posix()]=source_sha(path)
    for path in config.glob('*.json'):bindings[path.relative_to(authority).as_posix()]=source_sha(path)
    accepted=json.loads(Path(accepted_registration).read_text(encoding='utf-8'))
    cases=accepted['C1_case_bindings']
    from .runtime_seal import verify_c1
    for case,binding in cases.items():verify_c1(c1_root/case,binding,development_only=True)
    # Reuse sealed inference identities. Do not open or re-hash validation/test bytes.
    environment=fingerprint();verify_environment(environment,exact=True)
    registration=dict(schema='CZ_H1_CANONICAL_POSTHOC_REGISTRATION_V1',
        runner_protocol_relative='CANONICAL_PROTOCOL.json',canonical_source_sha256_lf=sources,
        canonical_protocol_sha256=protocols,authority_bindings_sha256_lf=bindings,C1_case_bindings=cases,
        scientific_provenance=protocol['scientific_provenance'],
        environment_fingerprint=environment,A_initialization=protocol['A_initialization'],
        historical_K_registry_sha256=oldseal['registry_sha256'],
        historical_registry_claims_superseded=True,H4_scope='HISTORICAL_UNCHANGED')
    if runtime_budget_projection is not None:
        projection=json.loads(Path(runtime_budget_projection).read_text(encoding='utf-8'))
        if projection['canonical_source_sha256_lf']!=sources:
            raise ProtocolError('Runtime forecast was measured on different source code')
        registration['runtime_budget_projection']=projection
    destination.mkdir(parents=True,exist_ok=True)
    dump(destination/'ENVIRONMENT_FINGERPRINT.json',environment)
    reg_path=destination/'REGISTRATION.json';dump(reg_path,registration)
    seal=dict(schema='CZ_CANONICAL_POSTHOC_RELEASE_SEAL_V1',registration_sha256=file_sha(reg_path),
        canonical_source_sha256_lf=sources,canonical_protocol_sha256=protocols,
        authority_bindings_sha256_lf=bindings)
    seal_path=destination/'RELEASE_SEAL.json';dump(seal_path,seal)
    pin=file_sha(seal_path)
    dump(destination/'RELEASE_PIN.json',dict(release_sha256=pin,policy_id=digest(registration),
        formal_training_authorized=False,target_test_authorized=False))
    return pin
