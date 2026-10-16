"""Complete an authorized source run and start sealing/scoring after acceptance."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,o):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2)+'\n');q.replace(p)
def main():
    if (ROOT/'AUTO_START_STATUS.json').exists():raise RuntimeError('Auto-start controller already registered')
    while not (ROOT/'STAGEWISE_ACCEPTANCE.json').exists():
        logs=[ROOT/('acceptance_'+c+'.log') for c in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1')]
        if any(p.exists() and 'Traceback (most recent call last)' in p.read_text()[-6000:] for p in logs):
            dump(ROOT/'AUTO_START_STATUS.json',dict(status='ACCEPTANCE_FAILED_RETAINED_NO_RESTART'))
            dump(ROOT/'MAIN_RUN_STATUS.json',dict(status='FAILED_RETAINED',phase='FULL_SOURCE_ACCEPTANCE'));return
        processes=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if str(ROOT/'accept_stagewise.py') in cmd:
                case='Rod_1_to_Rod_2' if 'Rod_1_to_Rod_2' in cmd else 'Rod_2_to_Rod_1'
                processes.append(dict(case=case,pid=int(p.name),exit_code=None,log=str(ROOT/('acceptance_'+case+'.log'))))
        dump(ROOT/'MAIN_RUN_STATUS.json',dict(status='RUNNING',phase='FULL_SOURCE_DEVELOPMENT_AND_ACCEPTANCE',
            runs=processes,updated_unix=time.time(),baseline='EXACT_FROZEN_REUSE',K_candidate_search_skipped=True))
        time.sleep(10)
    assert json.loads((ROOT/'STAGEWISE_ACCEPTANCE.json').read_text())['status']=='PASS'
    subprocess.run(['/root/miniconda3/bin/python',str(ROOT/'launch_stagewise.py')],check=True)
    subprocess.run(['/root/miniconda3/bin/python',str(ROOT/'seal_extensions.py')],check=True)
    dump(ROOT/'AUTO_START_STATUS.json',dict(status='ACCEPTED_SOURCE_TRAINING_REUSED_SEALING_AND_EXTENSIONS_STARTED',unix=time.time()))
if __name__=='__main__':main()
