"""Three CPU workers, all candidate freezes before selection and development replay."""
from pathlib import Path
import os,sys,subprocess,json,time,concurrent.futures
ROOT=Path(__file__).resolve().parent
def job(pair):
    d,w=pair;log=ROOT/'logs'/f'D{d}_L{w}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'prism_search.py'),'train',str(d),str(w)],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('JOB_COMPLETED',d,w,flush=True)
if __name__=='__main__':
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,[(d,w) for d in [4,8,16] for w in [8,16,32,64,128]]))
    subprocess.run([sys.executable,str(ROOT/'prism_search.py'),'select'],check=True)
    for d in [4,8,16]:subprocess.run([sys.executable,str(ROOT/'prism_search.py'),'evaluate',str(d)],check=True)
    (ROOT/'QUEUE_COMPLETED.json').write_text(json.dumps(dict(status='COMPLETED',seconds=time.time()-start)),encoding='utf-8')
