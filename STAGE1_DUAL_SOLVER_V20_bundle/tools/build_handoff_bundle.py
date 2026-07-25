#!/usr/bin/env python
"""Create a self-validating handoff zip while preserving include paths."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def validate(root):
    json_ok = npz_ok = csv_ok = checkpoint_ok = rowcount_ok = True
    for p in root.rglob('*'):
        if p.suffix == '.json':
            try: json.loads(p.read_text(encoding='utf-8'))
            except Exception: json_ok = False; raise
        elif p.suffix == '.npz':
            try:
                with np.load(p, allow_pickle=False) as z:
                    for key in z.files:
                        a = z[key]
                        if not np.isfinite(a).all(): raise ValueError(f'nonfinite {p}:{key}')
            except Exception: npz_ok = False; raise
        elif p.suffix == '.csv':
            try:
                frame = pd.read_csv(p)
                if frame.empty: raise ValueError(f'empty CSV {p}')
                numeric = frame.select_dtypes(include=[np.number]).to_numpy()
                if numeric.size and not np.isfinite(numeric).all(): raise ValueError(f'nonfinite CSV {p}')
            except Exception: csv_ok = False; raise
        elif p.suffix == '.pt':
            try: torch.load(p, map_location='cpu', weights_only=False)
            except Exception: checkpoint_ok = False; raise
    v16 = root / 'results_stage1/O2_O3_AUDIT_CLOSURE_v16'
    v17 = root / 'results_stage1/M2_JOINT_RECOVERY_v17'
    if v17.exists():
        required_rows = ['O2_matched_kernel_control/seed_metrics.csv','M2_clean/strategy_screen.csv','M2_clean/seed0_sparsity_path.csv','M2_clean/seed_metrics.csv','M2_clean/support_metrics.csv','M2_clean/delay_metrics.csv','M2_clean/centered_function_metrics.csv','M2_clean/contribution_metrics.csv','baselines/comparison.csv']
        rowcount_ok = all(len(pd.read_csv(v17/p)) >= (5 if p.endswith(('seed_metrics.csv','support_metrics.csv')) else 1) for p in required_rows)
        if not rowcount_ok: raise ValueError('v17 metric row-count validation failed')
    if v16.exists() and not v17.exists():
        required_rows = ['O1_centering/uncentered_metrics.csv','O1_centering/centered_metrics.csv','O3/original/centered_function_metrics.csv','O3/original/response_conditioned_contribution.csv','O3/original/full_contribution_recovery.csv','O3/refit/centered_function_metrics.csv','O3/refit/response_conditioned_contribution.csv','O3/refit/full_contribution_recovery.csv']
        rowcount_ok = all(len(pd.read_csv(v16/p)) == 45 for p in required_rows)
        if not rowcount_ok: raise ValueError('v16 metric row-count validation failed')
    return json_ok, npz_ok, csv_ok, checkpoint_ok, rowcount_ok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project-root', default='.')
    ap.add_argument('--bundle-name', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--include', nargs='+', required=True); a = ap.parse_args()
    root = Path(a.project_root).resolve(); stage = root / a.bundle_name; out = (root / a.output).resolve()
    if stage.exists(): shutil.rmtree(stage)
    if out.exists(): out.unlink()
    stage.mkdir()
    for item in a.include:
        src = root / item
        if not src.exists(): raise FileNotFoundError(src)
        dst = stage / item
        if src.is_dir(): shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
        else: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    v16 = stage / 'results_stage1/O2_O3_AUDIT_CLOSURE_v16'
    v17 = stage / 'results_stage1/M2_JOINT_RECOVERY_v17'
    v15 = stage / 'results_stage1/O2_O3_PROTOCOL_FIX_v15'
    v13 = stage / 'results_stage1/O1_CENTERED_O2_O3_v13'
    required = (['results_stage1/M2_JOINT_RECOVERY_v17/pass_fail_summary.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/M2_clean/summary.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/M2_noisy/summary.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/O2_matched_kernel_control/summary.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/baselines/comparison.csv',
                 'results_stage1/M2_JOINT_RECOVERY_v17/pytest_summary.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/artifact_consistency_audit.json',
                 'results_stage1/M2_JOINT_RECOVERY_v17/M2_JOINT_RECOVERY_v17_report.md'] if v17.exists() else
                ['results_stage1/O2_O3_AUDIT_CLOSURE_v16/pass_fail_summary.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/O1_centering/summary.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/O2/summary.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/O3/summary.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/pytest_summary.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/artifact_consistency_audit.json',
                 'results_stage1/O2_O3_AUDIT_CLOSURE_v16/O2_O3_AUDIT_CLOSURE_v16_report.md'] if v16.exists() else
                ['results_stage1/O2_O3_PROTOCOL_FIX_v15/pass_fail_summary.json',
                 'results_stage1/O2_O3_PROTOCOL_FIX_v15/O1_centering_fix/summary.json',
                 'results_stage1/O2_O3_PROTOCOL_FIX_v15/O2_branch_prox/summary.json',
                 'results_stage1/O2_O3_PROTOCOL_FIX_v15/O3/summary.json',
                 'results_stage1/O2_O3_PROTOCOL_FIX_v15/pytest_summary.json',
                 'results_stage1/O2_O3_PROTOCOL_FIX_v15/O2_O3_PROTOCOL_FIX_v15_report.md'] if v15.exists() else
                ['results_stage1/O1_CENTERED_O2_O3_v13/pass_fail_summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/O1_centering/summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/O2_variable_selection/summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/D0_delay_only/summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/O3_active_oracle/summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/pytest_summary.json',
                 'results_stage1/O1_CENTERED_O2_O3_v13/O1_CENTERED_O2_O3_v13_report.md'] if v13.exists() else
                ['results_stage1/KAN_O1_v12/pass_fail_summary.json',
                 'results_stage1/KAN_O1_v12/pytest_summary.json',
                 'results_stage1/KAN_O1_v12/KAN_O1_v12_report.md',
                 'results_stage1/KAN_O1_v12/truth_response_oracle/metrics.json'])
    missing = [p for p in required if not (stage / p).exists()]
    if missing: raise RuntimeError(f'missing required artifacts: {missing}')
    json_ok, npz_ok, csv_ok, checkpoint_ok, rowcount_ok = validate(stage)
    source_ok = runtime_ok = True
    if v17.exists():
        audit = json.loads((v17/'artifact_consistency_audit.json').read_text(encoding='utf8'))
        before = json.loads((v17/'source_hashes_before.json').read_text(encoding='utf8'))
        after = json.loads((v17/'source_hashes_after.json').read_text(encoding='utf8'))
        source_ok = bool(audit.get('source_hash_validation_pass')) and before == after and all(
            (stage/path).exists() and sha256(stage/path) == digest for path,digest in after.items())
        runtime_required = ['results_stage1/KAN_O1_v12/data_snapshot.npz','results_stage1/KAN_O1_v12/data_manifest.json','results_stage1/KAN_O1_v12/split_manifest.json','results_stage1/KAN_O1_v12/O1_convergence/selected_config.json','results_stage1/O2_O3_AUDIT_CLOSURE_v16/O1_centering','results_stage1/O2_O3_AUDIT_CLOSURE_v16/O2','results_stage1/O2_O3_AUDIT_CLOSURE_v16/O3','results_stage1/O2_D0_O3_DIAG_v14/D0']
        runtime_ok = all((stage/p).exists() for p in runtime_required)
        checkpoints=list(v17.rglob('*.pt'));payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in checkpoints]
        checkpoint_ok = checkpoint_ok and bool(checkpoints) and all('selection_mask' in x and 'learned_q' in x and x.get('metadata',{}).get('run_id')=='M2_JOINT_RECOVERY_v17' for x in payloads)
        run_id_ok=json.loads((v17/'run_manifest.json').read_text(encoding='utf8')).get('run_id')=='M2_JOINT_RECOVERY_v17'
        if not source_ok or not runtime_ok or not checkpoint_ok or not run_id_ok: raise RuntimeError('v17 source/runtime/checkpoint/run-id validation failed')
    elif v16.exists():
        audit = json.loads((v16/'artifact_consistency_audit.json').read_text(encoding='utf8'))
        source_ok = bool(audit.get('source_hash_validation_pass'))
        runtime_required = ['results_stage1/KAN_O1_v12/data_snapshot.npz','results_stage1/KAN_O1_v12/data_manifest.json','results_stage1/KAN_O1_v12/split_manifest.json','results_stage1/KAN_O1_v12/O1_convergence/selected_config.json','results_stage1/KAN_O1_v12/O1_convergence/checkpoints','results_stage1/O2_D0_O3_DIAG_v14/D0']
        runtime_ok = all((stage/p).exists() for p in runtime_required)
        if not source_ok or not runtime_ok: raise RuntimeError('source/runtime dependency validation failed')
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    manifest = {'files': [{'path': str(p.relative_to(stage)).replace('\\','/'),
                           'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]}
    (stage / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (stage / 'SHA256SUMS.txt').write_text('\n'.join(f"{x['sha256']}  {x['path']}" for x in manifest['files']), encoding='utf-8')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in stage.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(stage.parent))
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist()); expected = {str(p.relative_to(stage.parent)).replace('\\','/') for p in stage.rglob('*') if p.is_file()}
        if not expected <= names or z.testzip() is not None: raise RuntimeError('zip validation failed')
    print(json.dumps({'ZIP_ABSOLUTE_PATH':str(out), 'ZIP_SIZE_BYTES':out.stat().st_size,
        'ZIP_SHA256':sha256(out), 'BUNDLE_FILE_COUNT':len(expected),
        'JSON_VALIDATION_PASS':json_ok, 'NPZ_VALIDATION_PASS':npz_ok,
        'CSV_VALIDATION_PASS':csv_ok, 'CSV_ROWCOUNT_VALIDATION_PASS':rowcount_ok,
        'CHECKPOINT_VALIDATION_PASS':checkpoint_ok,
        'SOURCE_HASH_VALIDATION_PASS':source_ok,'RUNTIME_DEPENDENCY_PASS':runtime_ok,
        'ZIP_VALIDATION_PASS':True}, indent=2))
if __name__ == '__main__': main()
