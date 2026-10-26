"""Verify the public aggregate-only snapshot without data or bytecode writes."""
from pathlib import Path
import ast,csv,hashlib,json,math
R=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
manifest=read(R/'FILE_MANIFEST.json')
for name,item in manifest.items():
    p=R/name
    assert p.is_file(),name
    assert p.stat().st_size==item['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
    assert p.suffix.lower() not in ('.xlsx','.xls','.npz','.npy','.parquet','.pkl','.pt','.pth','.pem','.key'),name
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=name)
actual={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file() and p.name not in ('FILE_MANIFEST.json','LOCAL_VERIFICATION.json')}
assert actual==set(manifest),(actual-set(manifest),set(manifest)-actual)
metrics=read(R/'results/MODEL_METRICS.json')['metrics']
assert all(r['entity']!='CZ_Rod2' or r['variant']=='full' for r in metrics)
main=[r for r in metrics if r['experiment']=='PRISM_CZ_E128_20261026' and r['runtime']=='E128' and r['variant']=='full']
assert len(main)==6
assert {(r['entity'],r['delay']) for r in main}=={(e,d) for e in ('CZ_Rod1','CZ_Rod2') for d in (4,8,16)}
d=read(R/'results/INTERPRETABILITY_AND_ROD1_ABLATION.json')
assert len(d['rod1_ablations'])==12 and all(r['entity']=='CZ_Rod1' for r in d['rod1_ablations'])
for r in d['attribution']:
    assert math.isclose(sum(r['output_path_R2_points'].values())+r['mean_bias_penalty_R2_points'],100*r['R2'],rel_tol=1e-10,abs_tol=1e-8)
    assert r['additive_closure_max_abs']<1e-9
assert len(d['checks'])==7 and all(r['frozen_full_prediction_match'] and r['fits_during_development_replay']==0 and not r['reserved_tail_read'] for r in d['checks'])
effects=read(R/'results/SRU_ABLATION_EFFECTS.json')['effects']
assert effects and all(r['entity']!='CZ_Rod2' for r in effects)
with (R/'results/MODEL_METRICS.csv').open(encoding='utf-8',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==len(metrics)
for a,b in zip(rows,metrics):assert float(a['RMSE'])==b['RMSE']
print(json.dumps(dict(status='COMPLETED',verified_files=len(manifest),published_metric_rows=len(metrics),CZ_E128_full_rows=len(main),Rod1_ablation_rows=12,attribution_heads=7,raw_data_or_weights_included=False,Rod2_module_ablation_included=False)))
