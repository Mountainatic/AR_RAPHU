"""Two GPU workers; all baseline fits freeze before reference replay and scoring."""
from pathlib import Path
import sys,subprocess,concurrent.futures,json,time,hashlib
ROOT=Path(__file__).resolve().parent
LEGACY='/root/autodl-tmp/PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020/.venv/bin/python'
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def call(job):
    mode,name,delay=job;(ROOT/'logs').mkdir(exist_ok=True);start=time.time()
    args=[sys.executable,'-u',str(ROOT/'experiment.py'),'linear',str(delay)] if mode=='linear' else [sys.executable,'-u',str(ROOT/'experiment.py'),'train',name,str(delay)]
    with (ROOT/'logs'/f'{mode}_{name}_D{delay}.log').open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    result=dict(mode=mode,name=name,delay=delay,exit_code=p.returncode,seconds=time.time()-start)
    print('JOB_COMPLETE',json.dumps(result),flush=True);return result
if __name__=='__main__':
    assert json.loads((ROOT/'MIGRATION_CHECK.json').read_text())['status']=='COMPLETED'
    assert json.loads((ROOT/'SMOKE_TESTS.json').read_text())['status']=='COMPLETED'
    assert not (ROOT/'QUEUE_STARTED.json').exists();dump(ROOT/'QUEUE_STARTED.json',dict(time=time.time(),gpu_workers=2,cpu_threads=3))
    records=[];jobs=[('linear','linear',d) for d in [4,8,16]]+[('train',n,d) for d in [4,8,16] for n in ['LSTM','Transformer','iTransformer','TimeXer']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(call,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):records.append(f.result());dump(ROOT/'QUEUE_PROGRESS.json',records)
    assert all(r['exit_code']==0 for r in records),records
    receipts={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['runs','prism_reference'] for p in (ROOT/directory).rglob('FROZEN.json')}
    assert len(receipts)==18
    dump(ROOT/'ALL_FITS_FROZEN.json',dict(all_fits_frozen_before_evaluation=True,jobs=records,receipts=receipts))
    for d in [4,8,16]:
        with (ROOT/'logs'/f'PRISM_REPLAY_D{d}.log').open('w') as f:subprocess.run([LEGACY,'-u',str(ROOT/'prism_replay.py'),str(d)],stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'logs/EVALUATE.log').open('w') as f:p=subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'evaluate'],stdout=f,stderr=subprocess.STDOUT)
    dump(ROOT/'QUEUE_COMPLETED.json',dict(jobs=records,evaluate_exit_code=p.returncode));assert p.returncode==0
    with (ROOT/'logs/REPORT.log').open('w') as f:p=subprocess.run([sys.executable,'-u',str(ROOT/'report.py')],stdout=f,stderr=subprocess.STDOUT)
    dump(ROOT/'FINISH_COMPLETED.json',dict(status='COMPLETED' if p.returncode==0 else 'FAILED',report_exit_code=p.returncode));sys.exit(p.returncode)
