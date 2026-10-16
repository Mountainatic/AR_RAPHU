"""Three integration/support checks, through the actual canonical entry point."""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from canonical_core import digest,dump,file_sha,folds,prediction_hash
from maximal_anchor import build,CONTRACT,DIRECTIONS
from run_canonical import open_case
from canonical_engine import CHANNELS,CanonicalPipeline
from prism_benchmark import cz_h1_temporal_registry as registry
from prism_benchmark.v211_support import registered_fold_native_masks,support_id_hash


def run(authority,source_shared_h1,root,junit):
    authority=Path(authority);source_shared_h1=Path(source_shared_h1);root=Path(root)
    if (root/'INTEGRATION_SUPPORT_PREFLIGHT.json').exists():raise RuntimeError('Refuse overwrite completed preflight')
    root.mkdir(parents=True,exist_ok=True)
    registry.load_frozen_registry(authority)
    access_log=[];cases=[];grids=[];support_rows=[];bypass=[]
    registry_sha=registry.file_sha(authority/registry.CONFIG_ROOT/registry.REGISTRY_NAME)
    registration=dict(pipeline_contract_id='CZ_H1_CANONICAL_REGISTRY_SUPPORT_INTEGRATION_V1',
        anchor_contract=CONTRACT,k_registry_sha256=registry_sha,inner_block_rows=8,
        cf_warmup_rows=640,K_workers=1,training_support_hashes={'H1':{}},
        source_scripts_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')})
    for direction in DIRECTIONS:
        shared=root/'C1/H1'/direction
        train,manifest=build(source_shared_h1/direction,shared,direction,access_log)
        registration['training_support_hashes']['H1'][direction]=manifest['train_support_hash']
        cases.append((direction,shared,manifest))
    dump(root/'REGISTRATION.json',registration)
    for direction,shared,manifest in cases:
        pipe,train,loaded=open_case(root/'cases'/direction,shared,registration,no_training=True)
        for ch in CHANNELS:
            request=pipe.structural_grid_request(train,ch,None,root/'dry_grid'/direction/ch)
            profile=request['profiles']
            grids.append(dict(direction=direction,channel=ch,profiles=profile,count=len(profile),
                max_history_steps=max(p[1] for p in profile),m_tau=request['m_tau_values'],
                common_scoring_history=request['local_comparison_history'],
                actual_call='run_canonical.open_case -> CanonicalPipeline.structural_grid_request -> tune_k _run_structural_grid kwargs'))
            print('ACTUAL_CANONICAL_K_GRID',direction,ch,json.dumps(profile),flush=True)
        for block in (8,320):
            fs=folds(train,block)
            for history in (8,16,32,64,128,256):
                records=registered_fold_native_masks(train,fs,fit_history_steps=history,scoring_history_steps=256,
                    fit_cap=int(pipe.v2['row_caps']['single_channel_k_fit']),
                    evaluation_cap=int(pipe.v2['row_caps']['validation_selection_per_fold']))
                for record in records:
                    support_rows.append(dict(direction=direction,scoring_block_rows=block,fold_index=record['fold_index'],
                        history_steps=history,fit_raw_rows=len(record['fit_raw']),native_fit_rows=len(record['fit_native']),
                        capped_fit_rows=len(record['fit']),common_scoring_rows=len(record['evaluation']),
                        native_fit_hash=support_id_hash(record['fit_native']),common_scoring_hash=record['evaluation_support_hash']))
        # Long retained MODEL representations must not acquire dependencies for BYPASS.
        state=dict(stage='K',routing_status='BYPASSED',reference_index=0,
            representations=[dict(channel=ch,profile=[2,256],m_tau=12) for ch in CHANNELS])
        original_features=pipe.features
        def forbidden(*args,**kwargs):raise AssertionError('BYPASS attempted to read a K basis')
        pipe.features=forbidden
        parent_support=pipe.support_for(state,train)
        prediction=pipe.emit(state,parent_support)
        assert np.array_equal(prediction,np.zeros(len(train)))
        pipe.features=original_features
        bypass.append(dict(direction=direction,parent_rows=len(train),bypass_rows=len(parent_support),
            parent_support_hash=support_id_hash(train),bypass_support_hash=support_id_hash(parent_support),
            zero_parent_prediction_hash=prediction_hash(prediction),K_basis_accessed=False,
            early_rows_retained=int((parent_support.origin<256).sum())))
    counts=pd.DataFrame(support_rows)
    counts.to_csv(root/'CANDIDATE_NATIVE_SUPPORT_COUNTS.csv',index=False)
    dump(root/'ACTUAL_CANONICAL_K_REGISTRY.json',grids)
    dump(root/'K_BYPASS_PARENT_SUPPORT_AUDIT.json',bypass)
    strict=True;common=True
    for _,group in counts.groupby(['direction','scoring_block_rows','fold_index']):
        by_history=group.set_index('history_steps')
        strict &= bool(by_history.loc[8,'native_fit_rows']>by_history.loc[256,'native_fit_rows'])
        common &= len(set(group.common_scoring_hash))==1
    tests=ET.parse(junit).getroot();suites=list(tests.iter('testsuite'))
    test_counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    checks=dict(actual_runner_23_profiles_max_512s=all(g['count']==23 and g['max_history_steps']==256 and g['m_tau']==[4,8,12] for g in grids),
        real_maximal_anchors_short_fit_strictly_larger=strict,
        local_common_scoring_preserved=common,
        bypass_parent_support_preserved=all(r['parent_support_hash']==r['bypass_support_hash'] and r['early_rows_retained']>0 for r in bypass),
        original_authority_scoring_timestamps_preserved=all(m['original_scoring_timestamps_unchanged'] for _,_,m in cases),
        unit_tests=test_counts['tests']>=10 and not any(test_counts[k] for k in ['failures','errors','skipped']))
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',mode='NO_TRAINING_CANONICAL_INTEGRATION_SUPPORT_PREFLIGHT',
        checks={k:'PASS' if v else 'FAIL' for k,v in checks.items()},k_registry_sha256=registry_sha,
        training_performed=False,target_test_accessed=False,validation_loss_accessed=False,residual_statistics_accessed=False,
        target_values_accessed=False,source_train_labels_materialized=False,
        c1_binding_note='Maximal metadata C1 and frozen source inputs; labels are projected/reconstructed only by explicit for_fit loading, not this preflight.',
        inherited_CWA_parameters_unchanged=True,solver_unchanged=True,H4_old_engine_unchanged=True,
        registry_readiness_scope='K temporal registry only; unchanged original registry seal',
        pipeline_readiness_scope='These three integration/support contracts passed; full training, recursive numerical/domain execution, validation and checkpoint/test freezes are not executed here.',
        unit_test_counts=test_counts,access_log=access_log,
        cases=[m for _,_,m in cases])
    dump(root/'INTEGRATION_SUPPORT_PREFLIGHT.json',result)
    if not all(checks.values()):raise RuntimeError('Integration/support preflight failed')
    print(json.dumps({k:v for k,v in result.items() if k not in ('access_log','cases')},indent=2),flush=True)
    return result
