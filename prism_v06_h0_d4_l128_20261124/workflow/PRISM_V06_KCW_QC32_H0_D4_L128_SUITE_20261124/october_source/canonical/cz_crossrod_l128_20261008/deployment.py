"""Frozen deployment and execution gates, independent of fitted family internals."""
import json
from pathlib import Path
from .invariants import ProtocolError,digest,dump,file_sha


def physics_first_status(state):
    root=state
    while root.get('parent') is not None:
        root=root['parent']
    if root['stage']!='K':
        raise ProtocolError('Canonical chain lacks a K root')
    eligible=root['routing_status']=='ADMITTED'
    return dict(PF_ELIGIBLE=eligible,K_root_route=root['routing_status'],
        physics_first_status='PHYSICS_FIRST_PRISM' if eligible else 'PHYSICS_FIRST_NOT_ELIGIBLE',
        exploratory_chain_remains_available=True)


def require_execution(protocol,operation):
    allowed=protocol['execution_authorization']
    flag={'formal_development':'formal_development_training',
        'source_validation':'source_validation_reporting',
        'target_test':'target_test_inference'}[operation]
    if allowed.get(flag) is not True:
        raise ProtocolError('Execution is stopped by frozen authorization: '+flag)


def seal_deployment(destination,state,registration,release_sha,artifact_paths,*,run_complete):
    destination=Path(destination)
    if destination.exists():
        raise ProtocolError('Deployment checkpoint seals are immutable')
    if not run_complete or state['stage']!='A':
        raise ProtocolError('Seal requires the completed stagewise development chain')
    import copy
    state=copy.deepcopy(state)
    def bind_full(node):
        if node.get('parent'):bind_full(node['parent'])
        node.pop('deployment_model_id',None)
        node['deployment_scope']='FULL_DEVELOPMENT_AFTER_ROUTE_FREEZE'
        node['deployment_model_id']=digest(node)
    bind_full(state)
    expected_policy=digest(registration)
    chain=[]
    current=state
    while current is not None:
        if current.get('execution_mode') in {'solver_smoke','full_registry_smoke'} or current.get('formal_eligible') is False:
            raise ProtocolError('Solver smoke objects cannot become formal deployment checkpoints')
        if current['admission_policy_id']!=expected_policy:
            raise ProtocolError('Deployment chain mixes learning policies')
        if current['routing_status'] not in {'ADMITTED','BYPASSED','NOT_APPLICABLE'}:
            raise ProtocolError('Invalid/numerical-failure route cannot be deployed')
        chain.append(dict(stage=current['stage'],route=current['routing_status'],
            deployment_model_id=current['deployment_model_id'],partition_id=current['partition_id']))
        current=current.get('parent')
    if [record['stage'] for record in chain]!=['A','W','C','K']:
        raise ProtocolError('Deployment chain order changed')
    artifacts={str(Path(path).resolve()):file_sha(path) for path in artifact_paths}
    checkpoint=destination.with_name('deployment_state.json')
    if checkpoint.exists():
        raise ProtocolError('Existing unsealed deployment state cannot be overwritten')
    dump(checkpoint,state)
    artifacts[str(checkpoint.resolve())]=file_sha(checkpoint)
    manifest=dict(schema='CZ_POSTHOC_CANONICAL_DEPLOYMENT_SEAL_V1',status='DEPLOYMENT_FROZEN',
        registration_id=expected_policy,release_sha256=release_sha,
        state_path=str(checkpoint.resolve()),artifact_sha256=artifacts,chain=chain,
        scientific_provenance=registration['scientific_provenance'],
        target_test_accessed_by_this_run=False,fit_during_inference=False,
        **physics_first_status(state))
    dump(destination,manifest)
    return manifest,file_sha(destination)


def verify_deployment(manifest_path,expected_manifest_sha,registration,release_sha):
    manifest_path=Path(manifest_path)
    if not expected_manifest_sha or file_sha(manifest_path)!=expected_manifest_sha:
        raise ProtocolError('Pinned deployment checkpoint SHA256 mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema')!='CZ_POSTHOC_CANONICAL_DEPLOYMENT_SEAL_V1' or manifest['status']!='DEPLOYMENT_FROZEN':
        raise ProtocolError('Deployment checkpoint is not sealed')
    if manifest['registration_id']!=digest(registration) or manifest['release_sha256']!=release_sha:
        raise ProtocolError('Deployment checkpoint belongs to another registration/release')
    for path,expected in manifest['artifact_sha256'].items():
        if file_sha(path)!=expected:
            raise ProtocolError('Deployment checkpoint artifact changed: '+path)
    state=json.loads(Path(manifest['state_path']).read_text(encoding='utf-8'))
    if physics_first_status(state)['PF_ELIGIBLE']!=manifest['PF_ELIGIBLE']:
        raise ProtocolError('PF eligibility changed after route freeze')
    return state,manifest
