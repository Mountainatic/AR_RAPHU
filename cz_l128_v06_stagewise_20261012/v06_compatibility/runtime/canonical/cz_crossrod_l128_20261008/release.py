"""Build a reviewable pinned registration; never fit or authorize a run."""
import json
from pathlib import Path
from .invariants import ProtocolError,file_sha,source_sha,dump,digest
from .environment import fingerprint,verify_environment


def build_release(authority,c1_root,destination):
    authority=Path(authority);c1_root=Path(c1_root);destination=Path(destination)
    if destination.exists():raise ProtocolError('Release namespaces are immutable')
    module=Path(__file__).parent
    sources={p.relative_to(module).as_posix():source_sha(p) for p in module.rglob('*.py') if '__pycache__' not in p.parts}
    protocols={name:file_sha(module/name) for name in ['CANONICAL_PROTOCOL.json','FINAL_PROTOCOL.json','A_TEMPORAL_REGISTRY.json','K_MODEL_REGISTRY.json','C_MODEL_REGISTRY.json','W_MODEL_REGISTRY.json','A_MODEL_REGISTRY.json','IMMUTABLE_BASELINE.json']}
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
    cases={}
    from .maximal_inference import materialize_case
    for direction in ['Rod_1_to_Rod_2','Rod_2_to_Rod_1']:
        manifest_path=c1_root/direction/'C1_MAXIMAL_ANCHOR_MANIFEST.json'
        manifest=json.loads(manifest_path.read_text())
        source=Path(manifest['source_shared'])
        # Hash streams only: no target projection, loss, or descriptive statistics.
        required=['sample_ids/CZ_DIAM_RAW2S_CURRENT_L128_H1_CROSSROD_20261008/dynamic/record_time/primary/'+s+'.parquet'
                  for s in ('train','validation','test')]
        required+=['base_data/cz_czochralski/train.parquet']
        required+=sorted(p.relative_to(source).as_posix() for p in source.rglob('*.json') if p.is_file())
        byte_bindings={relative:file_sha(source/relative) for relative in required}
        optional='base_data/cz_czochralski/validation.parquet'
        byte_bindings[optional]=file_sha(source/optional) if (source/optional).is_file() else None
        inference=materialize_case(source,destination/'maximal_inference'/direction)
        for entry in inference.values():entry['relative']='maximal_inference/'+direction+'/'+entry['relative']
        cases[direction]=dict(manifest_sha256=file_sha(manifest_path),anchor_contract=manifest['contract_id'],
            train_support_hash=manifest['train_support_hash'],
            sample_relative='sample_ids/CZ_DIAM_RAW2S_CURRENT_L128_H1_CROSSROD_20261008/dynamic/record_time/primary/train.parquet',
            inference_source_root=str(source.resolve()),input_bytes_sha256=byte_bindings,maximal_inference_metadata=inference,
            input_access_role='HASH_BYTES_ONLY_NO_TARGET_STATISTICS',task_split_identity_sha256=digest(byte_bindings))
    environment=fingerprint();verify_environment(environment,exact=True)
    registration=dict(schema='CZ_H1_CANONICAL_POSTHOC_REGISTRATION_V1',
        runner_protocol_relative='CANONICAL_PROTOCOL.json',canonical_source_sha256_lf=sources,
        canonical_protocol_sha256=protocols,authority_bindings_sha256_lf=bindings,C1_case_bindings=cases,
        scientific_provenance=protocol['scientific_provenance'],
        environment_fingerprint=environment,A_initialization=protocol['A_initialization'],
        historical_K_registry_sha256=oldseal['registry_sha256'],
        historical_registry_claims_superseded=True,H4_scope='HISTORICAL_UNCHANGED')
    destination.mkdir(parents=True,exist_ok=True)
    dump(destination/'ENVIRONMENT_FINGERPRINT.json',environment)
    reg_path=destination/'REGISTRATION.json';dump(reg_path,registration)
    seal=dict(schema='CZ_CANONICAL_POSTHOC_RELEASE_SEAL_V1',registration_sha256=file_sha(reg_path),
        canonical_source_sha256_lf=sources,canonical_protocol_sha256=protocols,
        authority_bindings_sha256_lf=bindings)
    seal_path=destination/'RELEASE_SEAL.json';dump(seal_path,seal)
    pin=file_sha(seal_path)
    dump(destination/'RELEASE_PIN.json',dict(release_sha256=pin,policy_id=digest(registration),
        formal_training_authorized=protocol['execution_authorization']['formal_development_training'],target_test_authorized=protocol['execution_authorization']['target_test_inference']))
    return pin
