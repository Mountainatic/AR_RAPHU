from pathlib import Path
import json,subprocess,concurrent.futures,sys,time
ROOT=Path(__file__).resolve().parent;assert not (ROOT/'QUEUE_STARTED.json').exists();r=json.loads((ROOT/'REGISTRATION_BEFORE_FITS.json').read_text());(ROOT/'QUEUE_STARTED.json').write_text(json.dumps(dict(time=time.time(),cases=r['cases'])));(ROOT/'logs').mkdir()
def call(case,mode):
    with (ROOT/'logs'/f'{mode}_{case}.log').open('w') as f:p=subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),mode,case],stdout=f,stderr=subprocess.STDOUT)
    return dict(case=case,mode=mode,exit_code=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:fits=list(pool.map(lambda c:call(c,'train'),r['cases']))
(ROOT/'QUEUE_ALL_FITS_FROZEN.json').write_text(json.dumps(dict(fits=fits,all_models_frozen_before_any_development=all(x['exit_code']==0 for x in fits)),indent=2))
if any(x['exit_code'] for x in fits):sys.exit(1)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:ev=list(pool.map(lambda c:call(c,'evaluate'),r['cases']))
(ROOT/'QUEUE_COMPLETED.json').write_text(json.dumps(dict(fits=fits,evaluations=ev),indent=2))
if any(x['exit_code'] for x in ev):sys.exit(1)
