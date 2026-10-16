import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.deployment import *
from canonical.cz_h1_posthoc_v22.invariants import digest,dump,file_sha,ProtocolError


def chain(reg,k_route='BYPASSED'):
    current=None
    for stage in ['K','C','W','A']:
        current=dict(stage=stage,parent=current,admission_policy_id=digest(reg),
            partition_id='development_only',deployment_model_id='id_'+stage,
            routing_status=k_route if stage=='K' else 'ADMITTED')
    return current


@pytest.mark.parametrize('root_route',['BYPASSED','NOT_APPLICABLE'])
def test_admitted_downstream_never_restores_pf_eligibility(root_route):
    status=physics_first_status(chain({},root_route))
    assert not status['PF_ELIGIBLE'] and status['physics_first_status']=='PHYSICS_FIRST_NOT_ELIGIBLE'
    assert status['exploratory_chain_remains_available']


def test_pf_eligibility_requires_admitted_k_root():
    status=physics_first_status(chain({},'ADMITTED'))
    assert status['PF_ELIGIBLE'] and status['physics_first_status']=='PHYSICS_FIRST_PRISM'


@pytest.mark.parametrize('operation',['formal_development','source_validation','target_test'])
def test_present_task_stop_blocks_formal_training_validation_and_test(operation):
    protocol=dict(execution_authorization=dict(formal_development_training=False,
        source_validation_reporting=False,target_test_inference=False))
    with pytest.raises(ProtocolError,match='stopped'):require_execution(protocol,operation)


def test_checkpoint_requires_completed_chain_and_detects_modified_artifacts(tmp_path):
    reg=dict(scientific_provenance=dict(revision_status='POST_HOC_PROTOCOL_REVISION',
        historical_target_test_used_for_protocol_design=True))
    state=chain(reg,'ADMITTED')
    artifact=tmp_path/'fitted_contract.json';dump(artifact,{'synthetic':True})
    path=tmp_path/'DEPLOYMENT_SEAL.json'
    with pytest.raises(ProtocolError,match='completed'):
        seal_deployment(path,state,reg,'release',[artifact],run_complete=False)
    manifest,pin=seal_deployment(path,state,reg,'release',[artifact],run_complete=True)
    restored,verified=verify_deployment(path,pin,reg,'release')
    assert restored['deployment_model_id']!=state['deployment_model_id'] and verified['PF_ELIGIBLE']
    assert restored['deployment_scope']=='FULL_DEVELOPMENT_AFTER_ROUTE_FREEZE'
    with pytest.raises(ProtocolError,match='immutable'):
        seal_deployment(path,state,reg,'release',[artifact],run_complete=True)
    artifact.write_bytes(b'changed')
    with pytest.raises(ProtocolError,match='artifact changed'):
        verify_deployment(path,pin,reg,'release')


def test_unfrozen_checkpoint_cannot_open_test(tmp_path):
    path=tmp_path/'bad.json';dump(path,{'status':'TRAINING'})
    with pytest.raises(ProtocolError):verify_deployment(path,file_sha(path),{},'release')


def test_real_solver_smoke_can_never_be_sealed_as_formal_deployment(tmp_path):
    reg=dict(scientific_provenance={})
    state=chain(reg)
    state['execution_mode']='solver_smoke';state['formal_eligible']=False
    with pytest.raises(ProtocolError,match='smoke objects'):
        seal_deployment(tmp_path/'seal.json',state,reg,'release',[],run_complete=True)
