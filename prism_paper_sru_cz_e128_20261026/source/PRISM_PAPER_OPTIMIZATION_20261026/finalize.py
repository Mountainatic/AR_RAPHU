"""Audit completed runs and export two explicitly exploratory deployment heads."""
from pathlib import Path
import sys,json,pickle,hashlib,shutil,math
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp')
sys.path[:0]=[str(ROOT),str(BASE/'PRISM_CZ_E128_20261026/E128')]
from streaming import compact_model,run_stream
from suite import dump,effect,geometry
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
roots=[ROOT,BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026',BASE/'PRISM_CZ_E128_20261026',BASE/'PRISM_CZ_E128_TRANSPORT_20261026']
fits=0;model_files=0;metric_checks=0;source_checks=0
for root in roots:
    assert (root/'QUEUE_COMPLETED.json').exists()
    reg=json.loads((root/'REGISTRATION.json').read_text())
    for k,v in reg['code_hashes'].items():assert sha(root/k)==v,(root,k)
    for k,v in reg['source_hashes'].items():assert sha(k)==v,k;source_checks+=1
    gate=json.loads((root/'ALL_NEW_FITS_FROZEN.json').read_text());fits+=gate['job_count']
    for k,v in gate['receipts'].items():
        p=root/k;assert sha(p)==v
        f=json.loads(p.read_text());assert not f['development_read']
        assert sha(p.parent/'model.pkl')==f['model_sha256']
        assert sha(p.parent/'TRAINING_LEDGER.json')==f['ledger_sha256'];model_files+=1
    for p in (root/'results').rglob('RESULTS.json'):
        j=json.loads(p.read_text());assert j['fit_calls_during_evaluation']==0 and not j['reserved_tail_read']
        for row in j['metrics']:
            t=pd.read_parquet(p.parent/f"D{row['delay']}_predictions.parquet")
            error=math.sqrt(math.fsum((float(y)-float(z))**2 for y,z in zip(t.y_true,t[row['variant']]))/len(t))
            assert math.isclose(error,row['RMSE'],rel_tol=1e-12,abs_tol=1e-14);metric_checks+=1
assert fits==model_files==87

dest=ROOT/'deployment';dest.mkdir(exist_ok=True)
shutil.copy2(ROOT/'streaming.py',dest/'streaming.py')
shutil.copy2(BASE/'PRISM_CZ_E128_20261026/E128/stable_spline.py',dest/'stable_spline.py')
specs=[dict(name='SRU_H2S_D8',entity='SRU_H2S',delay=8,L=16,bins=None,
    source=BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026/runs/repaired/SRU_H2S/D8/full/model.pkl',
    predictions=BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026/results/repaired/SRU_H2S/D8_predictions.parquet'),
    dict(name='CZ_Rod1_D4_E128',entity='CZ_Rod1',delay=4,L=128,bins=list(zip([0,1,2,4,8,16,32,64],[1,2,4,8,16,32,64,128])),
    source=BASE/'PRISM_CZ_E128_20261026/runs/E128/CZ_Rod1/D4/full/model.pkl',
    predictions=BASE/'PRISM_CZ_E128_20261026/results/E128/CZ_Rod1/D4_predictions.parquet')]
exports=[]
for s in specs:
    original=pickle.loads(s['source'].read_bytes());compact=compact_model(original)
    bundle=dict(schema='PRISM_EXPLORATORY_DEPLOYMENT_20261026',name=s['name'],L=s['L'],delay=s['delay'],temporal_bins=s['bins'],model=compact,
        task='Concurrent y(t) with X through t and y through t-delay. Delay is an experimental availability assumption.',
        encoding='RAW_BLOCK_MEANS_THEN_AMPLITUDE_BASIS' if s['bins'] else 'POINTWISE_AMPLITUDE_CURRENT_PLUS_PAST_MEAN',
        source_model_sha256=sha(s['source']),status='ADAPTIVELY_SELECTED_DEVELOPMENT_EXAMPLE_NOT_FINAL_TEST')
    path=dest/f"{s['name']}.pkl";path.write_bytes(pickle.dumps(bundle,protocol=5))
    loaded=pickle.loads(path.read_bytes());g=geometry(s['entity']);a=np.vstack([np.load(Path(g['root'])/'data/train.npz')['values'],np.load(Path(g['root'])/'data/development.npz')['values']])
    pred,state=run_stream(loaded['model'],s['L'],s['delay'],a,g['train_end'],g['dev_end'],s['bins'])
    wide=pd.read_parquet(s['predictions']);np.testing.assert_allclose(pred,wide.full,rtol=1e-12,atol=1e-12)
    metric=dict(n=len(pred),RMSE=float(np.sqrt(np.mean((wide.y_true.to_numpy()-pred)**2))),MAE=float(np.mean(np.abs(wide.y_true.to_numpy()-pred))))
    if s['entity']=='SRU_H2S':
        baseline=pd.read_csv(BASE/'PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/results/point_predictions_D8.csv')
    else:baseline=pd.read_csv(BASE/'cz_sru_matched_20261024/CZ_Rod1/results/point_predictions_D4.csv')
    np.testing.assert_array_equal(wide.origin,baseline.origin)
    bp=baseline.Transformer.to_numpy();y=wide.y_true.to_numpy()
    ci=effect(y,pred,bp,0)
    parent=original['upper']['parent'];A=original['A'];w=original['upper']['W']
    count=parent['ridge']['parameter_count']+(w['ridge']['parameter_count'] if w else 0)+(len(A['model']['theta']) if A and A['admitted'] else 0)
    exports.append(dict(name=s['name'],file=path.name,sha256=sha(path),bytes=path.stat().st_size,
        source_model=str(s['source']),source_model_sha256=sha(s['source']),source_prediction_sha256=sha(s['predictions']),
        L=s['L'],delay=s['delay'],regression_coefficients=count,metrics=metric,KC=parent['config'],W=original['upper']['selected']['mode'],
        A=None if A is None else dict(selected=A['selected'],admitted=A['admitted']),
        Transformer_RMSE=float(np.sqrt(np.mean((y-bp)**2))),relative_RMSE_reduction_pct=float(100*(1-metric['RMSE']/np.sqrt(np.mean((y-bp)**2)))),
        paired_comparator_interval=ci,stream_max_absolute_error=float(np.max(np.abs(pred-wide.full.to_numpy()))),
        warm_checkpoint_bytes=len(pickle.dumps(state,protocol=5))))
dump(dest/'SELECTION_MANIFEST.json',dict(status='COMPLETED',exports=exports,source_selection='Examples selected after repeated development; formal SOTA/generalization not established',
    raw_workbook_read=False,reserved_tail_read=False,no_current_target_input=True,no_future_X_input=True))
dump(ROOT/'FINAL_VERIFICATION.json',dict(status='COMPLETED',new_variant_task_fits=fits,frozen_model_files_verified=model_files,
    saved_metric_RMSE_recomputed=metric_checks,original_source_bindings_verified=source_checks,export_count=len(exports),
    no_new_fitting_during_evaluation=True,raw_workbook_read=False,reserved_tail_read=False,
    all_negative_results_retained=True))
manifest={}
for root in roots:
    for p in root.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.gz','.zip','.log'):
            if p.name=='FINAL_FILE_MANIFEST.json':continue
            manifest[str(p.relative_to(BASE))]=dict(bytes=p.stat().st_size,sha256=sha(p))
dump(ROOT/'FINAL_FILE_MANIFEST.json',manifest)
print(json.dumps(dict(fits=fits,RMSE_rows=metric_checks,exports=exports),indent=2))
