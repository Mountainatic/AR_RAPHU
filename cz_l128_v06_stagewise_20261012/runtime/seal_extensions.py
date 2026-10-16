import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/src')]
from canonical.cz_crossrod_l128_20261008.invariants import dump,file_sha
def main():
    if (ROOT/'EXPERIMENTS_RELEASE.json').exists():raise RuntimeError('Extensions release is immutable')
    from research_extension.l128_adapter import configure
    ext=configure();audit=[]
    for case in ext.DIRECTIONS:
        for information in ('input_only','dynamic'):
            data=ext._registered_cz_arrays(ROOT,case,information)
            import numpy as np
            assert len(data['y'])==2048 and data['groups'].shape==(2048,)
            for indices in data['raw_process_indices']:
                assert np.all(indices<data['origins'][:,None]) and np.all(indices>=data['origins'][:,None]-128)
            assert np.array_equal(data['latest_target'],data['origins']-1)
            audit.append(dict(case=case,information=information,rows=len(data['y']),strict_past_L128=True,head_H1=True,source_train_only=True))
    files=list((ROOT/'research_extension').glob('*.py'))+[ROOT/n for n in ['run_synthetic_extensions.py','run_dependent_extensions.py','supervise_extensions.py']]
    files += [ROOT/'code/src/prism_benchmark'/n for n in ['e1e6_revalidation.py','e1e6_phase_b.py','e1e6_robustness.py','strict_oof_selection.py']]
    release=dict(status='FROZEN_BEFORE_EXTENSION_RUNS',main_release_sha256=json.loads((ROOT/'release/RELEASE_PIN.json').read_text())['release_sha256'],
        github_reference_commit='6ba6f1b510cc3476f9dbd1c64e86b07710b239da',file_sha256={str(p):file_sha(p) for p in files},
        experiments={
            'E2':dict(scope='DATASET_INDEPENDENT_IDENTIFIABLE_E2_V2_ORIGINAL_PROTOCOL',workers=6,dependency=[]),
            'E3':dict(scope='ORIGINAL_EQUAL_BUDGET_LINEAR_DIAGNOSTIC_ON_FROZEN_LEGAL_K_CHANNELS; NOT_FULL_PRISM_K_RETRAIN',histories=[32,64,128],seeds=list(range(10)),fold_PCA='FIT_PARTITION_ONLY',dependency=['BOTH_E1_FREEZES']),
            'E4':dict(scope='DATASET_INDEPENDENT_CANDIDATE_UNIVERSE_AND_FAMILY_SENSITIVITY',workers=6,dependency=[]),
            'E5':dict(scope='E3_E4_STRUCTURAL_STABILITY_AGGREGATE_NO_NEW_FIT',dependency=['E3','E4']),
            'E6':dict(scope='ORIGINAL_COMPACT_SOURCE_MEASUREMENT_NOISE_SURROGATE; NOT_FULL_E1_DEPLOYMENT_ROBUSTNESS',histories=[32,64,128],blocks=4,block_rows=512,embargo=132,N1_seeds=30,N2_seeds=10,workers=6,raw_control_coordinates=True,dependency=['BOTH_E1_FREEZES'])},
        source_support_audit=audit,baseline_models_retrained=False,private_data_publish=False,target_test_labels_used=False)
    dump(ROOT/'EXPERIMENTS_RELEASE.json',release)
    log=(ROOT/'supervise_extensions.log').open('ab');p=subprocess.Popen(['/root/miniconda3/bin/python',str(ROOT/'supervise_extensions.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    dump(ROOT/'EXTENSIONS_START_RECEIPT.json',dict(pid=p.pid,unix=time.time(),release_sha256=file_sha(ROOT/'EXPERIMENTS_RELEASE.json')))
    print('E2/E4 started; E3/E5/E6 frozen in dependency queue',p.pid,flush=True)
if __name__=='__main__':main()
