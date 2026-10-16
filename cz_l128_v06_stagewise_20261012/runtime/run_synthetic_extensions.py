import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/src')]
from canonical.cz_crossrod_l128_20261008.invariants import dump,file_sha
def verify():
    release=json.loads((ROOT/'EXPERIMENTS_RELEASE.json').read_text())
    for p,h in release['file_sha256'].items():
        if file_sha(p)!=h:raise RuntimeError('Experiment source binding changed: '+p)
    return release
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['E2','E4']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    release=verify();out=ROOT/'experiments';out.mkdir(exist_ok=True);status=ROOT/(a.stage+'_STATUS.json')
    if status.exists():raise RuntimeError('Synthetic experiment already started')
    dump(status,dict(status='RUNNING',experiment=a.stage,main_release_sha256=release['main_release_sha256'],private_data_accessed=False))
    started=time.time()
    try:
        if a.stage=='E2':
            from prism_benchmark.e1e6_revalidation import run_e2,audit_before_gate,phase_a_gate,SYNTHETIC_ID
            formal,null=run_e2(out,workers=a.workers);audit=audit_before_gate(out,formal,null);verdict=phase_a_gate(out,formal,null,audit)
            extra=dict(generator=SYNTHETIC_ID,phase_a_verdict=verdict,formal_rows=len(formal),null_rows=len(null))
        else:
            from prism_benchmark.e1e6_phase_b import run_e4
            run_e4(out,workers=a.workers);extra={}
        dump(status,dict(status='COMPLETED',experiment=a.stage,main_release_sha256=release['main_release_sha256'],seconds=time.time()-started,
            scope='DATASET_INDEPENDENT_ORIGINAL_SYNTHETIC_PROTOCOL',private_data_accessed=False,target_test_accessed=False,**extra))
    except Exception as e:
        dump(status,dict(status='FAILED_RETAINED',experiment=a.stage,error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
