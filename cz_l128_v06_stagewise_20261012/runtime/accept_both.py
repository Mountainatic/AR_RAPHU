import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    jobs=[]
    for c in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1'):
        f=(ROOT/('acceptance_'+c+'.log')).open('wb')
        p=subprocess.Popen(['/root/miniconda3/bin/python',str(ROOT/'accept_stagewise.py'),'--case-id',c],stdout=f,stderr=subprocess.STDOUT)
        jobs.append((c,p,f))
    codes=[]
    for c,p,f in jobs:codes.append(p.wait());f.close()
    if any(codes):raise RuntimeError('Full source acceptance failed; inspect per-direction logs')
    items=[json.loads((ROOT/('ACCEPTANCE_'+c+'.json')).read_text()) for c,_,_ in jobs]
    result=dict(status='PASS',reports=[r for item in items for r in item['reports']],
        full_real_source_chain_passed=True,target_test_accessed=False)
    (ROOT/'STAGEWISE_ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
