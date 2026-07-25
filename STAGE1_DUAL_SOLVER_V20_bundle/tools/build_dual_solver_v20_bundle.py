#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def validate_tree(root: Path) -> dict:
    counts={'json':0,'csv':0,'npz':0,'pt':0}
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        suffix=p.suffix.lower()
        if suffix=='.json':
            json.loads(p.read_text(encoding='utf-8-sig')); counts['json']+=1
        elif suffix=='.csv':
            with p.open('r',encoding='utf-8-sig',newline='') as f:
                rows=list(csv.reader(f))
            if len(rows)<2:
                raise ValueError(f'empty/header-only CSV: {p}')
            counts['csv']+=1
        elif suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for key in z.files:
                    arr=z[key]
                    if np.issubdtype(arr.dtype,np.number) and not np.isfinite(arr).all():
                        raise ValueError(f'nonfinite {p}:{key}')
            counts['npz']+=1
        elif suffix=='.pt':
            torch.load(p,map_location='cpu',weights_only=False); counts['pt']+=1
    return counts


def copy_item(root: Path, stage: Path, rel: str):
    src=root/rel; dst=stage/rel
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    else:
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',default='.')
    p.add_argument('--output',default='STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip')
    p.add_argument('--bundle-name',default='STAGE1_DUAL_SOLVER_V20_RESULTS_bundle')
    a=p.parse_args()
    root=Path(a.project_root).resolve(); stage=root/a.bundle_name; output=(root/a.output).resolve()
    if stage.exists(): shutil.rmtree(stage)
    if output.exists(): output.unlink()
    stage.mkdir(parents=True)
    includes=[
        'stage1','tests','tools','layers.py',
        'run_kan_fast_s0_v20.py','run_variational_stage1_s0_v20.py',
        'run_stage1_dual_solver_v20.py','RUN_V20_FULL.ps1','PACKAGE_V20_RESULTS.ps1',
        'README_V20.md',
        'results_stage1/STAGE1_DUAL_SOLVER_V20',
        'results_stage1/KAN_O1_v12/data_snapshot.npz',
        'results_stage1/KAN_O1_v12/data_manifest.json',
        'results_stage1/KAN_O1_v12/split_manifest.json',
        'results_stage1/M2_JOINT_RECOVERY_v17_REFERENCE',
    ]
    for rel in includes: copy_item(root,stage,rel)
    required=[
        'results_stage1/STAGE1_DUAL_SOLVER_V20/kan_fast/selected/summary.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/kan_fast/selected/selected_config.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/variational/selected/clean/summary.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/variational/selected/clean/selected_config.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/runtime_scientific_comparison.csv',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/pass_fail_summary.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/artifact_consistency_audit.json',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/STAGE1_DUAL_SOLVER_V20_report.md',
        'results_stage1/STAGE1_DUAL_SOLVER_V20/pytest_summary.json',
    ]
    missing=[rel for rel in required if not (stage/rel).exists()]
    if missing: raise RuntimeError(f'missing final artifacts: {missing}')
    counts=validate_tree(stage)
    files=sorted(p for p in stage.rglob('*') if p.is_file())
    manifest={'bundle_name':a.bundle_name,'file_count':len(files),'files':[{
        'path':str(p.relative_to(stage)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha256(p)
    } for p in files]}
    (stage/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (stage/'SHA256SUMS.txt').write_text('\n'.join(f"{r['sha256']}  {r['path']}" for r in manifest['files']),encoding='utf-8')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for pth in stage.rglob('*'):
            if pth.is_file(): z.write(pth,pth.relative_to(stage.parent))
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad is not None: raise RuntimeError(f'corrupt member {bad}')
        count=len(z.namelist())
    result={'ZIP_ABSOLUTE_PATH':str(output),'ZIP_SIZE_BYTES':output.stat().st_size,
            'ZIP_SHA256':sha256(output),'BUNDLE_FILE_COUNT':count,
            'JSON_VALIDATION_PASS':True,'CSV_VALIDATION_PASS':True,
            'NPZ_VALIDATION_PASS':True,'CHECKPOINT_VALIDATION_PASS':True,
            'ZIP_VALIDATION_PASS':True,'VALIDATED_COUNTS':counts}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
