import os,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert json.loads((ROOT/'STAGEWISE_ACCEPTANCE.json').read_text())['status']=='PASS'
    if (ROOT/'START_RECEIPT.json').exists():raise RuntimeError('Already launched; explicit resume required')
    items=[]
    for name in ('supervise_stagewise.py','finalize_crossrod.py'):
        f=(ROOT/(name+'.log')).open('ab');p=subprocess.Popen(['/root/miniconda3/bin/python',str(ROOT/name)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
        items.append(dict(controller=name,pid=p.pid,source_sha256=sha(ROOT/name)))
    (ROOT/'START_RECEIPT.json').write_text(json.dumps(dict(status='STARTED',unix=time.time(),controllers=items,
        main_release_sha256=json.loads((ROOT/'release/RELEASE_PIN.json').read_text())['release_sha256'],
        acceptance_sha256=sha(ROOT/'STAGEWISE_ACCEPTANCE.json'),baseline_import_receipt_sha256=sha(ROOT/'BASELINE_IMPORT_RECEIPT.json'),old_run_resumed=False),indent=2))
    print(json.dumps(items))
if __name__=='__main__':main()
