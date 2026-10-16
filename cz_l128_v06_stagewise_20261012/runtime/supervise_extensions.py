import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from run_synthetic_extensions import verify
def dump(p,o):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2)+'\n');q.replace(p)
def main():
    verify();active={};done={};queue={e:'WAITING' for e in ['E2','E3','E4','E5','E6']}
    def launch(e):
        script='run_synthetic_extensions.py' if e in {'E2','E4'} else 'run_dependent_extensions.py'
        f=(ROOT/(e+'.log')).open('ab');p=subprocess.Popen(['/root/miniconda3/bin/python',str(ROOT/script),e,'--workers','6'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
        active[e]=p;queue[e]='RUNNING'
    launch('E2');launch('E4')
    while True:
        for e,p in list(active.items()):
            code=p.poll()
            if code is not None:
                done[e]=code;queue[e]='COMPLETED' if code==0 else 'FAILED_RETAINED';active.pop(e)
        frozen=all((ROOT/'formal_actual'/c/'DEPLOYMENT_SEAL.json').exists() for c in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1'))
        low=shutil.disk_usage(ROOT).free<3*1024**3
        if frozen and not low:
            # Keep total numerical processes within the 16-core quota.
            if not active and queue['E3']=='WAITING':launch('E3')
            elif not active and queue['E6']=='WAITING' and queue['E3']=='COMPLETED':launch('E6')
            if queue['E5']=='WAITING' and all(queue[e]=='COMPLETED' for e in ['E3','E4']):launch('E5')
        for e in ['E3','E5','E6']:
            if queue[e]=='WAITING' and (done.get('E3',0)!=0 or (e=='E5' and done.get('E4',0)!=0)):queue[e]='DEPENDENCY_FAILED'
        main=ROOT/'MAIN_RUN_STATUS.json'
        main_failed=main.exists() and json.loads(main.read_text()).get('status')=='FAILED_RETAINED'
        if main_failed:
            for e in ['E3','E5','E6']:
                if queue[e]=='WAITING':queue[e]='MAIN_DEPENDENCY_FAILED'
        dump(ROOT/'E2_E6_PROCESS_STATUS.json',dict(updated_unix=time.time(),experiments=queue,
            processes={e:p.pid for e,p in active.items()},both_main_frozen=frozen,low_disk=low,
            E3='ORIGINAL_EQUAL_BUDGET_LINEAR_DIAGNOSTIC',E6='ORIGINAL_COMPACT_SOURCE_NOISE_SURROGATE',
            private_data_published=False,all_experiments_complete=all(s=='COMPLETED' for s in queue.values())))
        if all(s not in {'WAITING','RUNNING'} for s in queue.values()):break
        time.sleep(15)
if __name__=='__main__':main()
