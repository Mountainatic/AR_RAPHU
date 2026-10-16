import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,subprocess,time,shutil,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CASES=('Rod_1_to_Rod_2','Rod_2_to_Rod_1')
def dump(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
    assert json.loads((ROOT/'STAGEWISE_ACCEPTANCE.json').read_text())['status']=='PASS'
    pin=json.loads((ROOT/'release/RELEASE_PIN.json').read_text())['release_sha256']
    records=[];handles=[]
    for case in CASES:
        log=ROOT/(case+'_stagewise.log');f=log.open('ab');handles.append(f)
        out=ROOT/'formal_actual'/case;cmd=['/root/miniconda3/bin/python',str(ROOT/'stagewise_entry.py'),'--case-id',case,'--output-root',str(out)]
        if out.exists():cmd.append('--resume')
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(case=case,pid=p.pid,exit_code=None,process=p,command=cmd,log=str(log)))
    while True:
        free=shutil.disk_usage(ROOT).free
        for r in records:r['exit_code']=r['process'].poll()
        status='RUNNING' if any(r['exit_code'] is None for r in records) else 'FINISHED_CHECK_FREEZE_MANIFESTS'
        if any(r['exit_code'] not in (None,0) for r in records):status='FAILED_RETAINED'
        if free<3*1024**3:
            for r in records:
                if r['exit_code'] is None:os.killpg(r['pid'],signal.SIGSTOP)
            status='PAUSED_LOW_DISK'
        dump(ROOT/'MAIN_RUN_STATUS.json',dict(status=status,updated_unix=time.time(),free_bytes=free,
            release_sha256=pin,method='FINITE_STAGEWISE_OUTER_HOLDOUT',fully_nested_hyperparameter_CF=False,
            baseline='EXACT_FROZEN_REUSE',runs=[{k:v for k,v in r.items() if k!='process'} for r in records],
            E2_E6='REGISTERED_DEPENDENCY_QUEUE',old_recursive_run='PAUSED_RETAINED'))
        if status=='PAUSED_LOW_DISK' or all(r['exit_code'] is not None for r in records):break
        time.sleep(10)
    for f in handles:f.close()
if __name__=='__main__':main()
