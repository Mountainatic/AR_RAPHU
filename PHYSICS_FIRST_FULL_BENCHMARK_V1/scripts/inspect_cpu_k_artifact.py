#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-results', required=True)
    args=parser.parse_args()
    root=Path(args.cpu_results).expanduser().resolve()
    reports=[]
    for path in sorted(root.rglob('K-only.npz')):
        with np.load(path, allow_pickle=False) as stored:
            arrays={name:{'shape':list(stored[name].shape),'dtype':str(stored[name].dtype)} for name in stored.files}
        keys=set(arrays)
        has_oof=bool(keys & {'train_oof_prediction','oof_prediction','train_y_pred_oof','y_pred_train_oof'})
        reports.append({'path':str(path),'arrays':arrays,'has_train_oof_prediction':has_oof})
    result={'status':'PASS' if reports else 'NO_K_ARTIFACT','artifacts':reports}
    print('K_ARTIFACT_INSPECTION='+json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if reports else 2
if __name__=='__main__':
    raise SystemExit(main())
