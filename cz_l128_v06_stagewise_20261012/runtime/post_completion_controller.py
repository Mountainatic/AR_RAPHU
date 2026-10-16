import json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    while True:
        p=ROOT/'FINALIZATION_STATUS.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('status')=='FAILED_RETAINED':return
        if state.get('status')=='E1_AND_BASELINE_TARGET_INFERENCE_COMPLETED':break
        if (ROOT/'AUTO_START_STATUS.json').exists() and json.loads((ROOT/'AUTO_START_STATUS.json').read_text()).get('status')=='ACCEPTANCE_FAILED_RETAINED_NO_RESTART':return
        time.sleep(15)
    try:subprocess.run(['/root/miniconda3/bin/python',str(ROOT/'run_frozen_diagnostics.py')],check=True)
    except Exception as e:(ROOT/'FROZEN_DIAGNOSTICS_STATUS.json').write_text(json.dumps(dict(status='FAILED_RETAINED',error=str(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
