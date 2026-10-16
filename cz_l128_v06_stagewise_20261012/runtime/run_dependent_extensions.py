import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/src')]
from canonical.cz_crossrod_l128_20261008.invariants import dump
from run_synthetic_extensions import verify
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['E3','E5','E6']);parser.add_argument('--workers',type=int,default=6);a=parser.parse_args()
    release=verify();status=ROOT/(a.stage+'_STATUS.json');out=ROOT/'experiments';out.mkdir(exist_ok=True)
    if status.exists():raise RuntimeError('Extension output requires a new namespace')
    if not all((ROOT/'formal_actual'/c/'DEPLOYMENT_SEAL.json').exists() for c in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1')):raise RuntimeError('Both main deployments required')
    dump(status,dict(status='RUNNING',experiment=a.stage,main_release_sha256=release['main_release_sha256'],target_test_accessed=False))
    started=time.time()
    try:
        from research_extension.l128_adapter import configure
        ext=configure()
        if a.stage=='E3':ext.run_cz_e3(out,ROOT,workers=2)
        elif a.stage=='E6':ext.run_cz_e6(out,ROOT,workers=a.workers)
        else:
            import pandas as pd
            ext.run_cz_e5(out,pd.read_csv(out/'E3_MULTISCALE/equal_budget_multiscale.csv'))
        dump(status,dict(status='COMPLETED',experiment=a.stage,main_release_sha256=release['main_release_sha256'],seconds=time.time()-started,
            task='CZ_L128_H1',target_test_accessed=False,
            scope=release['experiments'][a.stage]['scope'],full_E1_model_substituted=False))
    except Exception as e:
        dump(status,dict(status='FAILED_RETAINED',experiment=a.stage,error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
