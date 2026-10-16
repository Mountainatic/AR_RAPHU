import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/src')]
from canonical.cz_crossrod_l128_20261008.execution import CanonicalExecution
from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
def runner(case):
    pin=json.loads((ROOT/'release/RELEASE_PIN.json').read_text())['release_sha256']
    return CanonicalExecution(module=ROOT/'code/canonical/cz_crossrod_l128_20261008',authority=ROOT/'code',registration_path=ROOT/'release/REGISTRATION.json',release_seal_path=ROOT/'release/RELEASE_SEAL.json',release_sha=pin,shared=ROOT/'C1'/case,case_id=case,pipeline_factory=StagewisePipeline)
def main():
    p=argparse.ArgumentParser();p.add_argument('--case-id',required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    report=runner(a.case_id).run_development(a.output_root,resume=a.resume);print(report['deployment_sha256'],flush=True)
if __name__=='__main__':main()
