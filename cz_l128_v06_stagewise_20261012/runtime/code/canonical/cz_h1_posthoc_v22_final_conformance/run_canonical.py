"""Sealed preflight, bounded real solver smoke, and explicitly gated formal actions.

Run as python -m canonical.cz_h1_posthoc_v22_final_conformance.run_canonical.
The current release permits preflight/smoke only. Future formal authorization
requires a new frozen protocol/release; CLI flags cannot bypass that barrier.
"""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from .execution import CanonicalExecution
from .canonical_engine import CanonicalPipeline,CHANNELS
from .release import build_release
from .invariants import ProtocolError,dump,digest
from .fold_geometry import folds,require_top_level_geometry
from .data import SAMPLE_RELATIVE


def open_case(*,authority,shared,case_id,release_dir,release_sha,root,mode,resume=False):
    runner=CanonicalExecution(module=Path(__file__).parent,authority=authority,
        registration_path=Path(release_dir)/'REGISTRATION.json',
        release_seal_path=Path(release_dir)/'RELEASE_SEAL.json',release_sha=release_sha,
        shared=shared,case_id=case_id,pipeline_factory=CanonicalPipeline)
    context=dict(runner._boot(),resume=resume)
    pipe=CanonicalPipeline(context,root,mode=mode)
    return runner,pipe,context


def preflight(pipe,context,root):
    samples=pd.read_parquet(Path(context['shared'])/SAMPLE_RELATIVE/'train.parquet')
    require_top_level_geometry(samples,pipe.protocol['geometry']['outer'])
    records=[]
    for channel in CHANNELS:
        request=pipe.structural_grid_request(samples,channel)
        if len(request['profiles'])!=23 or request['local_comparison_history']!=256:
            raise ProtocolError('Actual runner did not receive the frozen long-history K registry')
        records.append(dict(channel=channel,profiles=request['profiles'],profile_count=23,
            max_history_steps=256,m_tau=request['m_tau_values']))
    proof=dict(status='COMPLETED',role='NO_TRAINING_ACTUAL_RUNNER_PREFLIGHT',
        K_requests=records,anchor_rows=len(samples),folds=[dict(fold=j,fit_n=len(fi),score_n=len(ei),
            fit_hash=pipe.partition_id(samples.iloc[fi]),score_hash=pipe.partition_id(samples.iloc[ei]))
            for j,(fi,ei) in enumerate(folds(samples,pipe.protocol['geometry']['outer']))],
        runtime_audit=context['runtime_audit'],protocol=pipe.protocol,
        target_values_accessed=False,validation_accessed=False,test_accessed=False)
    dump(Path(root)/'ACTUAL_RUNNER_PREFLIGHT.json',proof)
    from .geometry_audit import recursive_geometry_audit
    dump(Path(root)/'FULL_RECURSIVE_METADATA_GEOMETRY_AUDIT.json',recursive_geometry_audit(samples,pipe.protocol))
    return proof


def smoke(pipe):
    started=time.monotonic();samples=pipe.load_development()
    state=pipe.learn('A',samples)
    stages=[];node=state
    while node is not None:
        stages.append(dict(stage=node['stage'],route=node['routing_status'],fit_rows=node['fit_rows'],
            partition_model_id=node['partition_model_id'],
            final_replay_coverage=node.get('final_replay_coverage'),
            channel_status=node.get('channel_status'),A_training_audit=node.get('A_training_prediction_audit')))
        node=node.get('parent')
    import resource
    checkpoints=[p for stage in ('K','C','W','A') for p in (pipe.root/'checkpoints'/(stage+'_CANDIDATE')).glob('*.json')]
    failures={};candidate_counts={};by_stage_family={}
    for p in checkpoints:
        v=json.loads(p.read_text())['value'];status=v['status'];candidate_counts[status]=candidate_counts.get(status,0)+1
        key=v['stage']+'/'+str(v['family']);record=by_stage_family.setdefault(key,{})
        record[status]=record.get(status,0)+1
        if status=='NUMERICALLY_INVALID':
            error=v['error_type'];failures[error]=failures.get(error,0)+1
    report=dict(status='COMPLETED',purpose='FULL_REGISTRY_DEVELOPMENT_ONLY_NUMERICAL_SMOKE' if pipe.mode=='full_registry_smoke' else 'REAL_RECURSIVE_DEVELOPMENT_ONLY_SOLVER_SMOKE',
        mock_learning_used=False,authority_solvers_used=True,development_rows=len(samples),
        numerical_fit_calls=pipe.fit_calls,elapsed_seconds=time.monotonic()-started,
        stages=stages,debug_candidate_scope=pipe.protocol['solver_smoke'] if pipe.mode=='solver_smoke' else None,
        effective_candidate_scope='FULL_FROZEN_K_C_W_A_REGISTRIES' if pipe.mode=='full_registry_smoke' else 'REGISTERED_DEBUG_SUBSET',
        formal_eligible=False,formal_training_performed=False,
        validation_accessed=False,test_accessed=False,policy_id=pipe.policy_id,
        source_code_and_data_seals_verified=True)
    report.update(candidate_count=len(checkpoints),candidate_status_counts=candidate_counts,
        numerical_failure_types=failures,peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        candidate_counts_by_stage_family=by_stage_family,
        cache_hits=pipe.checkpoints.hits,cache_misses=pipe.checkpoints.misses,
        cache_hit_rate=pipe.checkpoints.hits/max(1,pipe.checkpoints.hits+pipe.checkpoints.misses),
        full_candidate_registry=pipe.mode=='full_registry_smoke',environment=pipe.checkpoints.binding['environment'])
    dump(pipe.root/'SOLVER_SMOKE_REPORT.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','preflight','smoke','full-registry-smoke','train','validate','test'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--authority-project',type=Path,required=True)
    parser.add_argument('--c1-root',type=Path,required=True)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--release-sha')
    parser.add_argument('--case-id',choices=['Rod_1_to_Rod_2','Rod_2_to_Rod_1'])
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--deployment-path',type=Path)
    parser.add_argument('--deployment-pin')
    parser.add_argument('--validation-receipt-path',type=Path)
    parser.add_argument('--validation-receipt-pin')
    args=parser.parse_args()
    if args.action=='seal':
        print(build_release(args.authority_project,args.c1_root,args.release_dir));return
    if not args.case_id or not args.output_root or not args.release_sha:
        parser.error('execution requires case-id, output-root and an externally pinned release-sha')
    mode={'preflight':'metadata_only','smoke':'solver_smoke','full-registry-smoke':'full_registry_smoke','train':'formal_development',
          'validate':'inference_only','test':'inference_only'}[args.action]
    runner,pipe,context=open_case(authority=args.authority_project,
        shared=args.c1_root/args.case_id,case_id=args.case_id,release_dir=args.release_dir,
        release_sha=args.release_sha,root=args.output_root,mode=mode,resume=args.resume)
    if args.action=='preflight':preflight(pipe,context,args.output_root)
    elif args.action in {'smoke','full-registry-smoke'}:smoke(pipe)
    elif args.action=='train':runner.run_development(args.output_root,resume=args.resume)
    else:
        runner.run_inference('source_validation' if args.action=='validate' else 'target_test',
            output_root=args.output_root,deployment_path=args.deployment_path,
            deployment_pin=args.deployment_pin,validation_receipt_path=args.validation_receipt_path,
            validation_receipt_pin=args.validation_receipt_pin)


if __name__=='__main__':main()
