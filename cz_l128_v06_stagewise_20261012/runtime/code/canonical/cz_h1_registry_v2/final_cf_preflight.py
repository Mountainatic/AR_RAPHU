"""Metadata-only final replay plan through the actual canonical open_case runner."""
import argparse
import ast
import json
from pathlib import Path
from canonical_core import digest, dump, file_sha
from final_development_cf import frozen_contract, plan_replay
from run_canonical import open_case
from prism_benchmark.v211_support import support_id_hash
from prism_benchmark.cz_h1_temporal_registry import load_frozen_registry


def run(authority, c1_root, original_registration, output):
    authority,c1_root,output=map(Path,(authority,c1_root,output))
    if (output/'FINAL_CF_PREFLIGHT.json').exists():
        raise RuntimeError('Completed preflight is immutable')
    output.mkdir(parents=True,exist_ok=True)
    registration=json.loads(Path(original_registration).read_text(encoding='utf-8'))
    registration.update(pipeline_contract_id='CZ_H1_FINAL_DEVELOPMENT_CF_REFIT_V1',
        final_development_cf=frozen_contract(),
        source_scripts_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')})
    dump(output/'REGISTRATION.json',registration)
    load_frozen_registry(authority)
    results=[]
    records=[]
    for direction in registration['training_support_hashes']['H1']:
        pipe,frame,manifest=open_case(output/'cases'/direction,c1_root/direction,registration,no_training=True)
        assert 'y_true' not in frame and pipe.accessor is None
        grid={ch:pipe.k_grid(ch)[0] for ch in ['main_heater_power','joint_lift','crystal_rotation','crucible_rotation']}
        assert all(len(profiles)==23 and max(h for _,h in profiles)==256 for profiles in grid.values())
        for upstream in ['C','W']:
            blocks,audit=plan_replay(pipe,upstream,frame)
            assert audit['planned_replay_rows']>672 and audit['planned_replay_rows']>4*320
            for index,(fi,ei) in enumerate(blocks):
                fit,ev=frame.iloc[fi],frame.iloc[ei]
                assert fi[-1]<ei[0]
                assert fit.dependency_stop_exclusive.max()<ev.dependency_start.min()
                records.append(dict(direction=direction,upstream_stage=upstream,block=index,
                    fit_rows=len(fi),replay_rows=len(ei),fit_support_hash=support_id_hash(fit),
                    replay_support_hash=support_id_hash(ev),first_origin=int(ev.origin.min()),
                    last_origin=int(ev.origin.max()),entity=str(ev.entity_id.iloc[0]),
                    fit_last_partition_position=int(fi[-1]),replay_first_partition_position=int(ei[0]),
                    fit_dependency_stop_exclusive=int(fit.dependency_stop_exclusive.max()),
                    evaluation_dependency_start=int(ev.dependency_start.min()),
                    plan_only=True,target_values_accessed=False))
            assert blocks[-1][1][-1]==len(frame)-1
            results.append(dict(direction=direction,child_stage='W' if upstream=='C' else 'A',
                policy_id=pipe.policy_id,anchor_manifest_hash=digest(manifest),coverage=audit))
            print('FINAL_CF_METADATA_PLAN',direction,upstream,audit['partition_rows'],
                  audit['planned_replay_rows'],audit['block_count'],audit['block_lengths'][-1],flush=True)
    engine=Path(__file__).with_name('canonical_engine.py')
    def methods(source):
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
    before=methods(engine.with_name('canonical_engine.py.support_integration_reference.txt').read_text(encoding='utf-8'))
    after=methods(engine.read_text(encoding='utf-8'))
    changed=[name for name in before if before[name]!=after[name]]
    assert set(changed)=={'__init__','tune_residual'}
    result=dict(status='COMPLETED',mode='METADATA_ONLY_FULL_DEVELOPMENT_CF_PLAN',
        contract=frozen_contract(),registration_hash=digest(registration),cases=results,
        changed_existing_engine_methods=changed,
        unchanged_existing_engine_methods=[name for name in before if before[name]==after[name]],
        training_performed=False,target_values_accessed=False,validation_accessed=False,test_accessed=False,
        actual_full_cf_generation='NOT_YET_RUN',actual_final_refit='NOT_YET_RUN',
        H4_unchanged=True,K_registry_maximal_anchors_support_families_routing_unchanged=True,
        input_c1_root=str(c1_root),original_registration=str(original_registration))
    dump(output/'PLANNED_BLOCKS.json',records)
    dump(output/'FINAL_CF_PREFLIGHT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--c1-root',type=Path,required=True)
    parser.add_argument('--original-registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.authority,args.c1_root,args.original_registration,args.output)
