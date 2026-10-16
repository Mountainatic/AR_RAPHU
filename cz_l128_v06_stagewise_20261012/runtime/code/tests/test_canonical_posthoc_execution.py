"""Controller order tests use a synthetic adapter, not numerical integration."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.execution import CanonicalExecution
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError,digest,dump


class Adapter:
    def __init__(self,context,root,*,mode,trace,complete=True,fit_in_inference=False):
        self.mode=mode;self.policy_id=context['policy_id'];self.root=Path(root)
        self.trace=trace;self.complete=complete;self.fit_in_inference=fit_in_inference
        self.fit_calls=0
        self.root.mkdir(parents=True)
        self.trace.append('adapter:'+mode)

    def load_development(self):
        self.trace.append('read:train')
        return pd.DataFrame(dict(split=['train'],y_true=[.3]))

    def learn(self,stage,samples):
        assert self.mode=='formal_development' and stage=='A'
        self.fit_calls+=1
        self.trace.append('fit:development')
        state=None
        for s in ['K','C','W','A']:
            state=dict(stage=s,parent=state,admission_policy_id=self.policy_id,
                partition_id='train',deployment_model_id='fixture-'+s,routing_status='ADMITTED')
        return state

    def complete_development_artifacts(self,state):
        self.trace.append('complete:routes_cf_refit_oof')
        path=self.root/'development.json';dump(path,dict(synthetic_adapter=True))
        return [path]

    def development_complete(self,state):return self.complete

    def bind_deployment(self,state):self.trace.append('bind:sealed_deployment')

    def assert_inference_only(self):
        if self.mode!='inference_only':raise ProtocolError('Adapter cannot fit')

    def infer_and_report(self,role):
        self.trace.append('read:'+role)
        if self.fit_in_inference:self.fit_calls+=1
        path=self.root/'pointwise_fixture.json';dump(path,dict(synthetic_adapter=True,role=role))
        return [path]


def controller(tmp_path,*,authorized=True,complete=True,fit_in_inference=False):
    trace=[]
    reg=dict(scientific_provenance=dict(revision_status='POST_HOC_PROTOCOL_REVISION',
        historical_target_test_used_for_protocol_design=True))
    context=dict(registration=reg,policy_id=digest(reg),release_sha='fixture-release',
        protocol=dict(execution_authorization=dict(formal_development_training=authorized,
            source_validation_reporting=authorized,target_test_inference=authorized)))
    def factory(ctx,root,*,mode):
        return Adapter(ctx,root,mode=mode,trace=trace,complete=complete,
                       fit_in_inference=fit_in_inference)
    runner=CanonicalExecution(module=tmp_path,authority=tmp_path,registration_path=tmp_path/'reg',
        release_seal_path=tmp_path/'release',release_sha='fixture-release',shared=tmp_path,
        case_id='fixture',pipeline_factory=factory)
    def synthetic_boot():
        trace.append('verify:release_c1')
        return context
    runner._boot=synthetic_boot
    return runner,trace


@pytest.mark.parametrize('operation',['formal_development','source_validation','target_test'])
def test_execution_stop_precedes_adapter_and_any_label_access(tmp_path,operation):
    runner,trace=controller(tmp_path,authorized=False)
    with pytest.raises(ProtocolError,match='stopped'):
        if operation=='formal_development':runner.run_development(tmp_path/'run')
        else:runner.run_inference(operation,output_root=tmp_path/'run',
            deployment_path=tmp_path/'absent',deployment_pin='absent')
    assert trace==['verify:release_c1'] and not (tmp_path/'run').exists()


def test_development_validation_and_test_follow_sealed_order(tmp_path):
    runner,trace=controller(tmp_path)
    frozen=runner.run_development(tmp_path/'dev')
    kwargs=dict(deployment_path=frozen['manifest_path'],deployment_pin=frozen['deployment_sha256'])
    with pytest.raises(ProtocolError,match='source-validation receipt'):
        runner.run_inference('target_test',output_root=tmp_path/'premature-test',**kwargs)
    assert 'read:target_test' not in trace and not (tmp_path/'premature-test').exists()
    receipt,pin=runner.run_inference('source_validation',output_root=tmp_path/'validation',**kwargs)
    assert receipt['fit_calls']==0 and not receipt['selection_authority']
    test,_=runner.run_inference('target_test',output_root=tmp_path/'test',
        validation_receipt_path=tmp_path/'validation'/'INFERENCE_RECEIPT.json',
        validation_receipt_pin=pin,**kwargs)
    assert test['fit_calls']==0 and test['status']=='TARGET_TEST_INFERENCE_COMPLETE'
    assert trace.index('fit:development')<trace.index('complete:routes_cf_refit_oof')<trace.index('read:source_validation')<trace.index('read:target_test')
    assert trace.count('fit:development')==1


def test_incomplete_development_cannot_seal_checkpoint(tmp_path):
    runner,trace=controller(tmp_path,complete=False)
    with pytest.raises(ProtocolError,match='incomplete'):runner.run_development(tmp_path/'dev')
    assert not (tmp_path/'dev'/'DEPLOYMENT_SEAL.json').exists()


def test_inference_fit_violation_cannot_produce_receipt(tmp_path):
    runner,trace=controller(tmp_path,fit_in_inference=True)
    frozen=runner.run_development(tmp_path/'dev')
    with pytest.raises(ProtocolError,match='performed fitting'):
        runner.run_inference('source_validation',output_root=tmp_path/'validation',
            deployment_path=frozen['manifest_path'],deployment_pin=frozen['deployment_sha256'])
    assert not (tmp_path/'validation'/'INFERENCE_RECEIPT.json').exists()


def test_validation_report_tamper_stops_test_before_label_reader(tmp_path):
    runner,trace=controller(tmp_path)
    frozen=runner.run_development(tmp_path/'dev')
    kwargs=dict(deployment_path=frozen['manifest_path'],deployment_pin=frozen['deployment_sha256'])
    _,pin=runner.run_inference('source_validation',output_root=tmp_path/'validation',**kwargs)
    (tmp_path/'validation'/'pointwise_fixture.json').write_bytes(b'changed')
    with pytest.raises(ProtocolError,match='report changed'):
        runner.run_inference('target_test',output_root=tmp_path/'test',
            validation_receipt_path=tmp_path/'validation'/'INFERENCE_RECEIPT.json',
            validation_receipt_pin=pin,**kwargs)
    assert 'read:target_test' not in trace and not (tmp_path/'test').exists()


def test_execution_does_not_overwrite_an_existing_namespace(tmp_path):
    runner,_=controller(tmp_path)
    (tmp_path/'existing').mkdir()
    with pytest.raises(ProtocolError,match='must be new'):runner.run_development(tmp_path/'existing')
