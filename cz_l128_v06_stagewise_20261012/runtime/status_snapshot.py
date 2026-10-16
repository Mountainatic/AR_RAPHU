"""Read-only current-version status, never executes a model or opens labels."""
import json,pathlib,os,shutil,time
ROOT=pathlib.Path(__file__).resolve().parent
def main():
    files=['MAIN_RUN_STATUS.json','AUTO_START_STATUS.json','STAGEWISE_ACCEPTANCE.json','FINALIZATION_STATUS.json','E2_E6_PROCESS_STATUS.json','FROZEN_DIAGNOSTICS_STATUS.json']+[e+'_STATUS.json' for e in ['E2','E3','E4','E5','E6']]
    result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root=str(ROOT),free_GiB=shutil.disk_usage(ROOT).free/1024**3,states={})
    for name in files:
        p=ROOT/name;result['states'][name]=json.loads(p.read_text()) if p.exists() else None
    result['directions']={}
    for case in ['Rod_1_to_Rod_2','Rod_2_to_Rod_1']:
        p=ROOT/'formal_actual'/case
        result['directions'][case]=dict(MODEL_counts={s:len(list((p/'tuning'/s).glob('*/model.json'))) for s in ['K','C','W','A']},
            deployed=(p/'DEPLOYMENT_SEAL.json').exists(),A_candidate_count=len(list((p/'checkpoints/A_CANDIDATE').glob('*.json'))))
    result['actual_processes']=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(ROOT) in cmd:result['actual_processes'].append(dict(pid=int(p.name),command=cmd))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
