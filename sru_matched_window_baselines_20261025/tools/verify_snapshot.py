from pathlib import Path
import json,csv,hashlib,math,sys
ROOT=Path(__file__).resolve().parents[1];WINDOWS={4:8,8:16,16:16}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
manifest=read(ROOT/'PUBLICATION_MANIFEST.json');assert all(sha(ROOT/k)==v['sha256'] and (ROOT/k).stat().st_size==v['bytes'] for k,v in manifest.items())
reg=read(ROOT/'REGISTRATION.json');gate=read(ROOT/'ALL_FITS_FROZEN.json');assert gate['all_fits_frozen_before_evaluation']
assert {int(k):v for k,v in reg['windows_by_label_delay'].items()}==WINDOWS
assert not reg['neural_precision']['TF32'] and reg['neural_precision']['SDPA']=='math only'
for group in [reg['code_sha256'],gate['receipts']]:assert all(sha(ROOT/k)==v for k,v in group.items())
for name,digest in reg['reference_sha256'].items():
    p=ROOT/name
    if p.exists():assert sha(p)==digest
    else:assert p.suffix=='.pkl' or p.name.endswith('_TRAINING.json')
old=read(ROOT/'previous/ALL_SELECTIONS_FROZEN.json');assert {int(k):v['full']['window'] for k,v in old['choices'].items()}==WINDOWS
for d,L in WINDOWS.items():
    for name in ['linear','LSTM','Transformer','iTransformer','TimeXer']:
        r=read(ROOT/'runs'/f'D{d}'/name/'FROZEN.json')
        assert not r['development_read_during_fit'] and r['latest_fit_target']<=6048-d
        if name=='linear':
            assert all(v['config']['Ly']==L for v in r['selected'].values())
            assert r['selected']['ARX']['config']['Xlags']==list(range(L))
        else:
            assert r['selected']['config']['L']==L and len(r['files'])==3
            best=min(r['ledger'],key=lambda v:(v['MSE'],v['candidate']));assert best['candidate']==r['selected']['candidate']
            for row in best['folds']:assert row['latest_fit_target']<row['first_validation_origin']-128-d
    assert read(ROOT/'audits'/f'PRISM_D{d}_CAUSAL_REPLAY.json')['status']=='COMPLETED'
assert read(ROOT/'MIGRATION_CHECK.json')['strict_FP32_to_FP64_passed']
assert read(ROOT/'results/INDEPENDENT_VERIFICATION.json')['full_PRISM_predictions_preserved']
assert len(read(ROOT/'results/CAUSAL_AUDIT.json')['checks'])==36
m=rows(ROOT/'results/metrics.csv');assert len(m)==27;expected={}
names=['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer','PRISM_no_A','PRISM']
for d,L in WINDOWS.items():
    f=rows(ROOT/'results'/f'point_predictions_D{d}.csv');assert len(f)==2016
    for i,row in enumerate(f):assert int(row['origin'])==6048+i and int(row['latest_available_y_index'])==6048+i-d
    for name in names:
        y=[float(r['y_true']) for r in f];p=[float(r[name]) for r in f];assert all(math.isfinite(v) for v in y+p)
        e=[a-b for a,b in zip(y,p)];sse=math.fsum(v*v for v in e);mean=math.fsum(y)/len(y);valid=[i for i,v in enumerate(y) if abs(v)>1e-8]
        values=dict(R2=1-sse/math.fsum((v-mean)**2 for v in y),RMSE=math.sqrt(sse/len(y)),MAE=math.fsum(abs(v) for v in e)/len(y),
            MAPE_pct=100*math.fsum(abs(e[i])/abs(y[i]) for i in valid)/len(valid),
            sMAPE_pct=200*math.fsum(abs(v)/max(abs(a)+abs(b),1e-8) for a,b,v in zip(y,p,e))/len(y),
            WAPE_pct=100*math.fsum(abs(v) for v in e)/math.fsum(abs(v) for v in y),bias=-math.fsum(e)/len(e))
        selected=[r for r in m if int(r['label_delay'])==d and r['model']==name];assert len(selected)==1
        assert int(selected[0]['direct_window'])==(1 if name=='Persistence' else L)
        for k,v in values.items():assert close(v,selected[0][k]),(d,name,k)
        for i in range(len(y)):expected[(d,name,6048+i)]=(y[i],p[i],e[i])
long=rows(ROOT/'results/point_predictions_long.csv');assert len(long)==54432;seen=set()
for r in long:
    key=(int(r['label_delay']),r['model'],int(r['origin']));assert key not in seen;seen.add(key);y,p,e=expected[key]
    for k,v in [('y_true',y),('y_pred',p),('residual',e),('absolute_error',abs(e)),('squared_error',e*e)]:assert close(r[k],v)
assert seen==set(expected)
result=dict(status='COMPLETED',verified_publication_files=len(manifest),metric_rows=len(m),point_rows=len(long),
    matched_windows_verified=True,training_only_selection_verified=True,all_metrics_recomputed=True,all_source_hashes_verified=True,reserved_tail_read=False)
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
