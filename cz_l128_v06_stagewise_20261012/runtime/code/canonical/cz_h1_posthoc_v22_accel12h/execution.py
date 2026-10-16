"""Sealed execution order, awaiting the versioned numerical-engine adapter.

The controller contains no model/fold choices. Its pipeline factory is an
explicit adapter interface, not a fallback to the historical engine.
"""
import json
from pathlib import Path
from .invariants import ProtocolError,digest,dump,file_sha
from .runtime_seal import verify_release,verify_c1,_inside
from .deployment import require_execution,seal_deployment,verify_deployment


class CanonicalExecution:
    def __init__(self,*,module,authority,registration_path,release_seal_path,
                 release_sha,shared,case_id,pipeline_factory):
        self.module=Path(module)
        self.authority=Path(authority)
        self.registration_path=Path(registration_path)
        self.release_seal_path=Path(release_seal_path)
        self.release_sha=release_sha
        self.shared=Path(shared)
        self.case_id=case_id
        self.pipeline_factory=pipeline_factory

    def _boot(self):
        registration,audit=verify_release(self.module,self.authority,self.registration_path,
            self.release_seal_path,self.release_sha)
        relative=registration['runner_protocol_relative']
        if relative not in registration['canonical_protocol_sha256']:
            raise ProtocolError('Runner protocol is not bound to the runtime seal')
        protocol=json.loads(_inside(self.module,relative).read_text(encoding='utf-8'))
        if protocol.get('scientific_gate_status')!='FROZEN':
            raise ProtocolError('Scientific protocol choices have not been frozen')
        binding=registration['C1_case_bindings'][self.case_id]
        manifest=verify_c1(self.shared,binding,development_only=True)
        if 'environment_fingerprint' in registration:
            from .environment import verify_environment
            audit['environment']=verify_environment(registration['environment_fingerprint'])
        return dict(registration=registration,protocol=protocol,runtime_audit=audit,
            C1_manifest=manifest,shared=self.shared,authority=self.authority,
            case_id=self.case_id,policy_id=digest(registration),release_sha=self.release_sha,
            release_dir=self.registration_path.parent,runtime_budget_projection=registration.get('runtime_budget_projection',{}))

    def _pipeline(self,context,root,mode):
        root=Path(root)
        if root.exists() and not context.get('resume',False):
            raise ProtocolError('Execution output namespaces must be new')
        pipe=self.pipeline_factory(context,root,mode=mode)
        if pipe.mode!=mode or pipe.policy_id!=context['policy_id']:
            raise ProtocolError('Engine adapter changed the registered execution mode/policy')
        return pipe

    def run_development(self,output_root,*,resume=False):
        context=dict(self._boot(),resume=resume)
        require_execution(context['protocol'],'formal_development')
        pipe=self._pipeline(context,output_root,'formal_development')
        import pandas as pd
        from .data import SAMPLE_RELATIVE
        from .dag_plan import plan
        metadata=pd.read_parquet(self.shared/SAMPLE_RELATIVE/'train.parquet')
        metadata['sample_id']=metadata.view_sample_id.astype(str)
        dag=plan(metadata,context['protocol'],bindings=dict(registries=context['registration']['canonical_protocol_sha256'],
            authority=context['registration']['authority_bindings_sha256_lf'],
            canonical_sources=context['registration']['canonical_source_sha256_lf'],C1=context['C1_manifest']))
        dump(Path(output_root)/'DAG_PLAN.json',dag)
        samples=pipe.load_development()
        if set(samples['split'])!={'train'}:
            raise ProtocolError('Development entry received non-development rows')
        import os,multiprocessing
        from .runtime_budget import watchdog
        supervisor=multiprocessing.get_context('spawn').Process(target=watchdog,
            args=(os.getpid(),output_root,context['protocol']['runtime_budget']),daemon=True)
        supervisor.start()
        try:
            state=pipe.learn('A',samples)
            # The adapter must finish global route freeze, full-development CF
            # refits and canonical OOF exports before the deployment seal exists.
            artifacts=pipe.complete_development_artifacts(state)
            if not pipe.development_complete(state):
                raise ProtocolError('Development fitting/refit/audit sequence is incomplete')
            manifest_path=Path(output_root)/'DEPLOYMENT_SEAL.json'
            manifest,pin=seal_deployment(manifest_path,state,context['registration'],
                context['release_sha'],artifacts,run_complete=True)
            return dict(deployment_manifest=manifest,manifest_path=str(manifest_path.resolve()),
                        deployment_sha256=pin)
        finally:
            pipe.budget.close()
            from .k_acceleration import close_workers
            close_workers(pipe,interrupt=True)
            supervisor.terminate();supervisor.join(timeout=5)

    def _validation_receipt(self,path,pin,context,deployment_pin):
        if path is None or not pin or file_sha(path)!=pin:
            raise ProtocolError('Target test requires a pinned completed source-validation receipt')
        receipt=json.loads(Path(path).read_text(encoding='utf-8'))
        expected=dict(schema='CZ_CANONICAL_INFERENCE_RECEIPT_V1',
            status='SOURCE_VALIDATION_REPORT_COMPLETE',admission_policy_id=context['policy_id'],
            release_sha256=context['release_sha'],deployment_sha256=deployment_pin,
            fit_calls=0)
        if any(receipt.get(key)!=value for key,value in expected.items()):
            raise ProtocolError('Source-validation receipt belongs to an incomplete/different deployment')
        for artifact,sha in receipt['artifact_sha256'].items():
            if file_sha(artifact)!=sha:
                raise ProtocolError('Source-validation report changed before target-test entry')

    def run_inference(self,operation,*,output_root,deployment_path,deployment_pin,
                      validation_receipt_path=None,validation_receipt_pin=None):
        if operation not in {'source_validation','target_test'}:
            raise ProtocolError('Unknown inference partition')
        context=self._boot()
        require_execution(context['protocol'],operation)
        state,checkpoint=verify_deployment(deployment_path,deployment_pin,
            context['registration'],context['release_sha'])
        if operation=='target_test':
            self._validation_receipt(validation_receipt_path,validation_receipt_pin,
                context,deployment_pin)
        context=dict(context,deployment_checkpoint_verified=True,inference_operation=operation)
        # No label reader or inference adapter is reached until every barrier
        # has passed. The adapter itself must also prohibit learn/refit.
        pipe=self._pipeline(context,output_root,'inference_only')
        pipe.bind_deployment(state)
        pipe.assert_inference_only()
        artifacts=pipe.infer_and_report(operation)
        pipe.assert_inference_only()
        if pipe.fit_calls!=0:
            raise ProtocolError('Inference performed fitting after deployment freeze')
        manifest=dict(schema='CZ_CANONICAL_INFERENCE_RECEIPT_V1',
            status='SOURCE_VALIDATION_REPORT_COMPLETE' if operation=='source_validation' else 'TARGET_TEST_INFERENCE_COMPLETE',
            admission_policy_id=context['policy_id'],release_sha256=context['release_sha'],
            deployment_sha256=deployment_pin,fit_calls=0,partition=operation,
            artifact_sha256={str(Path(p).resolve()):file_sha(p) for p in artifacts},
            scientific_provenance=context['registration']['scientific_provenance'],
            PF_ELIGIBLE=checkpoint['PF_ELIGIBLE'],selection_authority=False)
        path=Path(output_root)/'INFERENCE_RECEIPT.json'
        dump(path,manifest)
        return manifest,file_sha(path)
