#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--exit-code',type=int,required=True); a=p.parse_args()
text=Path(a.input).read_text(encoding='utf-8',errors='replace')
match=re.search(r'(\d+) passed',text); failed=re.search(r'(\d+) failed',text); skipped=re.search(r'(\d+) skipped',text)
payload={'collected_or_passed':int(match.group(1)) if match else 0,'failed':int(failed.group(1)) if failed else 0,'skipped':int(skipped.group(1)) if skipped else 0,'exit_code':a.exit_code,'passed':a.exit_code==0 and bool(match)}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
