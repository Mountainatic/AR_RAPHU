"""Residual-only A0/A1/A2 comparison; frozen base, no raw workbook/control access."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import time, json, hashlib, sys
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import welch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'cz_h1_light_whitening_20261006'
ATTACH=Path('C:/Users/27437/.codex/attachments/f09a01b0-85dc-4981-9182-3b5e117d02e6/已粘贴的文本.txt')
ALPHAS=[1e-4,1e-2,1.,100.]
HISTORIES=[32,64,96]
SHORT=np.asarray([1,2,3,4,5,7,8],int)
F0=.14;DT=2.;BAND=(.125,.155)
START=time.perf_counter()
SELECTION=[];DIAGNOSTICS=[];ACF_ROWS=[];PSD_ROWS=[];PROBE_ROWS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def log(*s):
    print(f'[{time.perf_counter()-START:.1f}s]',*s,flush=True)
    if time.time()-ATTACH.stat().st_ctime>1800:raise RuntimeError('STOP 30-minute task budget')

def transform(x,kind):
    """x[:,k-1] = mature base residual t-k; all differences stay inside H."""
    if kind=='A0':return x.copy()
    at=x[:,SHORT-1];older=x[:,SHORT]
    out=[x,np.sign(at),np.abs(at),at-older,(np.sign(at)!=np.sign(older)).astype(float)]
    if kind=='A2':
        k=np.arange(1,x.shape[1]+1)
        out.extend([(x@np.cos(2*np.pi*F0*k*DT))[:,None],(x@np.sin(2*np.pi*F0*k*DT))[:,None]])
    return np.concatenate(out,axis=1)

def history(frame,H,column='base_residual'):
    x=np.full((len(frame),H),np.nan);buffer=deque(maxlen=H);entity=None;previous=None
    for i,(e,t,v) in enumerate(zip(frame.entity,frame.origin,frame[column])):
        if e!=entity or previous is None or int(t)!=previous+1:buffer.clear()
        if len(buffer)==H:x[i]=np.asarray(buffer)[::-1]
        # Read/mature current base residual only after current features have been constructed.
        buffer.append(float(v));entity=e;previous=int(t)
    return x,np.isfinite(x).all(axis=1)

class RidgeGrid:
    def __init__(self,x,y):
        self.mean=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-12]=1.
        z=(x-self.mean)/self.scale;self.intercept=float(y.mean())
        ev,vec=eigh(z.T@z,check_finite=False,driver='evd');self.eig=np.maximum(ev,0.)
        self.vec=vec;self.rhs=vec.T@(z.T@(y-self.intercept))
    def state(self,alpha):return dict(mean=self.mean,scale=self.scale,coef=self.vec@(self.rhs/(self.eig+alpha)),intercept=self.intercept,alpha=float(alpha))

def predict(model,x):return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept']

def infer(frame,kind,H,model):
    # Offline vectorization is equivalent to the explicit causal deque loop above.
    x,available=history(frame,H)
    correction=np.zeros(len(frame));correction[available]=predict(model,transform(x[available],kind))
    out=frame.copy();out['A_model']=kind;out['H']=H;out['alpha_A']=float(model['alpha'])
    out['A_prediction']=correction;out['final_prediction']=out.base_prediction+correction
    out['final_residual']=out.y_true-out.final_prediction;out['A_state_available']=available
    return out

def acf(frame,column='final_residual',maxlag=900):
    z=frame[column].to_numpy(float);mean=z.mean();num=np.zeros(maxlag+1);pairs=np.zeros(maxlag+1,int)
    groups=((frame.entity!=frame.entity.shift())|(frame.origin.diff()!=1)).cumsum()
    for _,g in frame.groupby(groups,sort=False):
        a=g[column].to_numpy(float)-mean;n=len(a);size=1<<(2*n-1).bit_length()
        f=np.fft.rfft(a,size);raw=np.fft.irfft(f*f.conj(),size)[:min(n,maxlag+1)]
        num[:len(raw)]+=raw;pairs[:len(raw)]+=np.arange(n,n-len(raw),-1)
    return num/num[0],pairs

def describe(frame):
    y=frame.final_residual.to_numpy(float);a,pairs=acf(frame)
    assert frame.entity.nunique()==1 and (frame.origin.diff().dropna()==1).all()
    f,psd=welch(y,fs=.5,nperseg=1024,noverlap=512,detrend='constant',scaling='density')
    band=(f>=BAND[0])&(f<=BAND[1]);dom=int(np.argmax(psd[1:])+1);near=int(np.argmin(abs(f-F0)));bmax=np.flatnonzero(band)[np.argmax(psd[band])]
    row=dict(n=len(y),RMSE=float(np.sqrt(np.mean(y*y))),MAE=float(np.mean(abs(y))),bias=float(y.mean()),E160=float(np.sum(a[1:161]**2)),E900=float(np.sum(a[1:901]**2)),band_power=float(psd[band].sum()*(f[1]-f[0])),nearest_0140_frequency_Hz=float(f[near]),nearest_0140_PSD=float(psd[near]),dominant_frequency_Hz=float(f[dom]),dominant_period_seconds=float(1/f[dom]),dominant_PSD=float(psd[dom]),band_peak_frequency_Hz=float(f[bmax]),band_peak_PSD=float(psd[bmax]),A_cold_start_rows=int((~frame.A_state_available).sum()))
    for k in [3,4,7,14]:row[f'rho_lag{k}']=float(a[k])
    return row,a,pairs,f,psd

def improvement(child,reference):
    changes={key:100*(1-child[key]/reference[key]) for key in ['RMSE','MAE','E160','E900','band_power','nearest_0140_PSD','dominant_PSD']}
    success=changes['band_power']>=10. and changes['E160']>=-1e-9 and changes['E900']>=-1e-9 and changes['RMSE']>=-.5-1e-9
    return bool(success),changes

def training_folds(frame):
    edges=np.linspace(0,len(frame),6,dtype=int)
    for j in range(4):yield j,np.arange(edges[j+1]),np.arange(edges[j+1],edges[j+2])

def fit_probe(train,kind,H,alpha,model_A0_probe=None):
    if kind=='A0' and model_A0_probe is not None:return model_A0_probe,None
    x,complete=history(train,H);blocks=[]
    for j,fi,ei in training_folds(train):
        valid=fi[complete[fi]]
        assert train.global_origin.iloc[valid].max()<train.global_origin.iloc[ei].min()
        model=RidgeGrid(transform(x[valid],kind),train.base_residual.to_numpy()[valid]).state(alpha)
        b=infer(train.iloc[ei].reset_index(drop=True),kind,H,model)
        b['probe_train_fold']=j;b['probe_A_fit_end']=int(train.global_origin.iloc[valid].max())
        blocks.append(b)
    residual_stream=pd.concat(blocks,ignore_index=True)
    x,ok=history(residual_stream,32,'final_residual')
    probe=RidgeGrid(x[ok],residual_stream.final_residual.to_numpy()[ok]).state(1.)
    return probe,residual_stream

def probe_eval(frame,model):
    x,available=history(frame,32,'final_residual');p=np.zeros(len(frame));p[available]=predict(model,x[available])
    rows=[]
    for support,mask in [('all_cold_start',np.ones(len(frame),bool)),('mature_history',available)]:
        y=frame.final_residual.to_numpy()[mask];err=y-p[mask];rmse=float(np.sqrt(np.mean(err*err)));zero=float(np.sqrt(np.mean(y*y)));mean=float(np.sqrt(np.mean((y-model['intercept'])**2)))
        rows.append(dict(support=support,n=len(y),probe_H=32,probe_alpha=1.,R2=float(1-np.sum(err*err)/np.sum((y-y.mean())**2)),probe_RMSE=rmse,zero_baseline_RMSE=zero,train_mean_baseline_RMSE=mean,RMSE_improvement_vs_zero_percent=100*(1-rmse/zero),RMSE_improvement_vs_train_mean_percent=100*(1-rmse/mean)))
    return rows

def develop(arm,train,val,old_recipe):
    H0=old_recipe['H'];models={};outputs={};metrics={};probes={};probe_streams=[]
    histories=sorted(set(HISTORIES+[H0]))
    for H in histories:
        x,ok=history(train,H)
        for kind in ['A0','A1','A2']:
            grid=RidgeGrid(transform(x[ok],kind),train.base_residual.to_numpy()[ok])
            candidates=[]
            for alpha in ALPHAS:
                model=grid.state(alpha);out=infer(val,kind,H,model);m,*_=describe(out)
                rec=dict(arm=arm,A_model=kind,H=H,alpha_A=alpha,features=len(model['coef']),fit_rows=int(ok.sum()),validation_RMSE=m['RMSE'],validation_MAE=m['MAE'],validation_bias=m['bias'],selected_alpha_for_history=False,primary_frozen_H=H==H0,selected_primary=False,sensitivity_best_H=False,status='VALIDATION_ONLY_SELECTION')
                SELECTION.append(rec);candidates.append((m['RMSE'],-alpha,model,out,m,rec))
            best=min(candidates,key=lambda c:(c[0],c[1]));best[-1]['selected_alpha_for_history']=True
            if H==H0:
                if kind=='A0':
                    model=dict(np.load(PREV/f'{arm}_A.npz'))
                    out=infer(val,kind,H,model);m,*_=describe(out)
                    assert np.max(abs(out.final_residual-val.final_residual))<1e-12
                    # Primary A0 is exact frozen baseline, regardless of sensitivity-grid ranking.
                    for c in candidates:
                        if c[2]['alpha']==model['alpha']:c[-1]['selected_primary']=True;c[-1]['status']='EXACT_FROZEN_A0_REUSED'
                else:_,_,model,out,m,rec=best;rec['selected_primary']=True
                models[kind]=model;outputs[kind]=out;metrics[kind]=m
                np.savez_compressed(ROOT/f'{arm}_{kind}.npz',**model)
            log(arm,kind,'H',H,'best alpha',float(best[2]['alpha']),'validation RMSE',round(best[0],8))
    for kind in ['A0','A1','A2']:
        selected=[row for row in SELECTION if row['arm']==arm and row['A_model']==kind and row['selected_alpha_for_history']]
        best=min(selected,key=lambda r:(r['validation_RMSE'],r['H'],-r['alpha_A']));best['sensitivity_best_H']=True
        old_probe=dict(np.load(PREV/f'{arm}_PROBE_final.npz')) if kind=='A0' else None
        probes[kind],stream=fit_probe(train,kind,H0,float(models[kind]['alpha']),old_probe)
        np.savez_compressed(ROOT/f'{arm}_{kind}_PROBE.npz',**probes[kind])
        if stream is not None:stream['arm']=arm;probe_streams.append(stream)
    a1_success,a1_change=improvement(metrics['A1'],metrics['A0'])
    a2_success,a2_change=improvement(metrics['A2'],metrics['A0'])
    gate=a1_success or a2_success
    log('Rod development gate',arm,gate,'A1 vs A0',a1_change,'A2 vs A0',a2_change)
    return dict(arm=arm,H=H0,models=models,probes=probes,validation=outputs,validation_metrics=metrics,probe_streams=probe_streams,validation_gate=gate,A1_validation_success=a1_success,A2_validation_success=a2_success)

def record(arm,split,kind,frame,probe):
    row,a,pairs,f,p=describe(frame);row.update(arm=arm,split=split,A_model=kind,H=int(frame.H.iloc[0]),alpha_A=float(frame.alpha_A.iloc[0]))
    DIAGNOSTICS.append(row)
    for lag,v in enumerate(a):ACF_ROWS.append(dict(arm=arm,split=split,A_model=kind,lag_samples=lag,lag_seconds=lag*2,ACF=v,pairs=int(pairs[lag])))
    for freq,psd in zip(f,p):PSD_ROWS.append(dict(arm=arm,split=split,A_model=kind,frequency_Hz=freq,period_seconds=1/freq if freq>0 else np.nan,PSD=psd))
    for r in probe_eval(frame,probe):PROBE_ROWS.append(dict(arm=arm,split=split,A_model=kind,**r))

def plot(arm,split):
    df=pd.DataFrame(ACF_ROWS);ps=pd.DataFrame(PSD_ROWS)
    label='Rod2' if arm=='Rod2_Common4' else arm
    for typ in ['ACF','PSD']:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for kind,color in [('A0','#2676b8'),('A1','#d66b33'),('A2','#6f9a55')]:
            for ax in axes:
                if typ=='ACF':
                    g=df[(df.arm==arm)&(df.split==split)&(df.A_model==kind)&(df.lag_samples>0)];ax.plot(g.lag_seconds,g.ACF,label=kind,color=color,lw=1.)
                else:
                    g=ps[(ps.arm==arm)&(ps.split==split)&(ps.A_model==kind)];ax.semilogy(g.frequency_Hz,g.PSD,label=kind,color=color,lw=1.)
        if typ=='ACF':
            axes[0].set_xlim(2,320);axes[1].set_xlim(2,1800)
            for ax in axes:ax.set_xlabel('Lag (seconds)');ax.set_ylabel('ACF')
        else:
            axes[0].set_xlim(0,.25);axes[1].set_xlim(.10,.18)
            for ax in axes:ax.axvspan(*BAND,color='gray',alpha=.1);ax.axvline(.14,color='gray',ls='--',lw=.8);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD')
        for ax in axes:ax.grid(alpha=.15);ax.legend()
        fig.suptitle(f'{arm} | {split} | frozen H | A2: narrowband diagnostic')
        fig.tight_layout();suffix='' if split=='descriptive_holdout' else '_validation'
        fig.savefig(ROOT/f'{label}_A0_A1_A2_{typ}{suffix}.png',dpi=160);plt.close(fig)

def main():
    if (ROOT/'GLOBAL_SELECTION_FREEZE.json').exists():raise RuntimeError('Refuse frozen experiment overwrite')
    old=json.loads((PREV/'GLOBAL_SELECTION_FREEZE.json').read_text(encoding='utf-8'))
    source_files=['FROZEN_MEMBERSHIP.parquet','BASE_OOF_POINTWISE.parquet','POINTWISE_VALIDATION.parquet','POINTWISE_TEST.parquet','GLOBAL_SELECTION_FREEZE.json']
    source_files+=list(old['artifacts'])
    hashes={name:sha(PREV/name) for name in source_files}
    write('EXPERIMENT_PREREGISTRATION.json',dict(task='SHORT_CYCLE_RESIDUAL_A_ONLY',classification='POST_HOC_DESCRIPTIVE_DIAGNOSTIC',prior_files_sha256=hashes,base_predictor='EXACT_FROZEN_NO_REFIT',primary_history='previous arm-specific H fixed for A0/A1/A2',sensitivity_histories=HISTORIES,alphas=ALPHAS,A0='exact old A.npz loaded with old scaler/coefficients; no re-selection',A1='full raw history plus sign/abs/difference/zero-crossing at fixed short lags only',short_lags=SHORT.tolist(),feature_dimension='A0 H; A1 H+28; A2 H+30',difference='r(t-k)-r(t-k-1), short max k=8 needs <=9, all within H>=32',no_raw_lag_duplication=True,exact_zero_event_indicator=False,A2=dict(f0_Hz=F0,dt=DT,features=['sum past residual cos','sum past residual sin'],role='DIAGNOSTIC_ONLY',note='harmonics are linear combinations of raw lags already present; extra columns change Ridge penalty, not the information set'),alpha_selection='minimum online cold-start validation RMSE; exact ties prefer stronger alpha',history_sensitivity='validation-only best H reported, does not replace frozen primary H and is not scored on holdout',selection_whiteness_used=False,replication_gate='Rod2 validation only: A1 or A2 compared to A0 meets prespecified success; then replicate unchanged protocol on both Rod1 arms before any current holdout scoring',success=dict(band_power_reduction_min_percent=10.,E160_reduction_min_percent=0.,E900_reduction_min_percent=0.,RMSE_worsening_max_percent=.5,stable='criteria met on both validation and descriptive holdout'),A2_extra_diagnostic_success='A2 meets same criteria vs A1 on validation and holdout while A1 has no stable success vs A0',cold_start='empty each split/entity/continuous segment, correction=0 first H rows, append current residual after feature construction',train='only saved prequential base OOF residuals, never base in-sample residuals',probe=dict(H=32,alpha=1.,train='A0 exact old saved probe; A1/A2 probe fit only another fixed-recipe prequential corrected residual stream',scope='prediction metrics only, no tuning'),PSD=dict(fs=.5,nperseg=1024,noverlap=512,band=BAND,nearest_frequency=F0),ACF='same centered biased segmented FFT as prior experiment; maxlag900',budget_minutes=30))
    oof=pd.read_parquet(PREV/'BASE_OOF_POINTWISE.parquet').rename(columns={'r_base':'base_residual','y_base_pred':'base_prediction'})
    val=pd.read_parquet(PREV/'POINTWISE_VALIDATION.parquet')
    assert (oof.global_origin>oof.fit_end).all()
    fitted={};first='Rod2_Common4'
    fitted[first]=develop(first,oof[oof.arm==first].reset_index(drop=True),val[val.arm==first].reset_index(drop=True),old['arms'][first])
    replication=fitted[first]['validation_gate']
    if replication:
        for arm in ['Rod1_Common4','Rod1_AllLegal']:fitted[arm]=develop(arm,oof[oof.arm==arm].reset_index(drop=True),val[val.arm==arm].reset_index(drop=True),old['arms'][arm])
    else:log('Rod2 validation gate not passed: no Rod1 replication; protocol stops after Rod2 reporting')
    pd.DataFrame(SELECTION).to_csv(ROOT/'SHORT_CYCLE_MODEL_SELECTION.csv',index=False)
    probe_streams=[x for recipe in fitted.values() for x in recipe['probe_streams']]
    pd.concat(probe_streams,ignore_index=True).to_parquet(ROOT/'PROBE_TRAIN_PREQUENTIAL.parquet',index=False)
    write('GLOBAL_SELECTION_FREEZE.json',dict(status='ALL_INCLUDED_RECIPES_AND_PROBES_FROZEN_BEFORE_CURRENT_HOLDOUT_ACCESS',included_arms=list(fitted),Rod1_replication_gate_validation_only=replication,recipes={arm:dict(H=recipe['H'],alphas={kind:float(model['alpha']) for kind,model in recipe['models'].items()},feature_dims={kind:len(model['coef']) for kind,model in recipe['models'].items()},A1_validation_success=recipe['A1_validation_success'],A2_validation_success=recipe['A2_validation_success']) for arm,recipe in fitted.items()},model_hashes={p.name:sha(p) for p in ROOT.glob('*.npz')},source_hashes=hashes))
    log('GLOBAL FREEZE COMPLETE; opening saved descriptive holdout once')
    holdout=pd.read_parquet(PREV/'POINTWISE_TEST.parquet')
    pointwise=[];decision=[]
    for arm,recipe in fitted.items():
        test=holdout[holdout.arm==arm].reset_index(drop=True)
        hold_metrics={}
        for kind in ['A0','A1','A2']:
            m=dict(np.load(ROOT/f'{arm}_{kind}.npz'))
            output=infer(test,kind,recipe['H'],m)
            if kind=='A0':assert np.max(abs(output.final_residual-test.final_residual))<1e-12
            hold_metrics[kind]=describe(output)[0]
            for split,f in [('validation',recipe['validation'][kind]),('descriptive_holdout',output)]:
                f=f.copy();f['split']=split;pointwise.append(f)
                record(arm,split,kind,f,recipe['probes'][kind])
        a1_hold,a1changes=improvement(hold_metrics['A1'],hold_metrics['A0'])
        a1_stable=recipe['A1_validation_success'] and a1_hold
        a2_extra_val,a2valchanges=improvement(recipe['validation_metrics']['A2'],recipe['validation_metrics']['A1'])
        a2_extra_hold,a2holdchanges=improvement(hold_metrics['A2'],hold_metrics['A1'])
        a2_extra=a2_extra_val and a2_extra_hold
        verdict='SHORT_CYCLE_A_IMPROVEMENT_SUPPORTED' if a1_stable else 'NARROWBAND_PRIOR_DIAGNOSTIC_ONLY' if a2_extra else 'NO_ADDITIONAL_SHORT_CYCLE_GAIN'
        decision.append(dict(arm=arm,verdict=verdict,recommendation='A1' if a1_stable else 'A0; A2 diagnostic only',A1_validation_success=recipe['A1_validation_success'],A1_holdout_success=a1_hold,A1_stable_success=a1_stable,A2_extra_validation_success=a2_extra_val,A2_extra_holdout_success=a2_extra_hold,A1_holdout_change=a1changes,A2_vs_A1_validation_change=a2valchanges,A2_vs_A1_holdout_change=a2holdchanges))
        log(arm,'verdict',verdict,'A1 holdout changes',a1changes)
    pd.concat(pointwise,ignore_index=True).to_parquet(ROOT/'SHORT_CYCLE_POINTWISE.parquet',index=False)
    pd.DataFrame(DIAGNOSTICS).to_csv(ROOT/'SHORT_CYCLE_WHITENING_METRICS.csv',index=False)
    pd.DataFrame(ACF_ROWS).to_csv(ROOT/'SHORT_CYCLE_ACF.csv',index=False)
    pd.DataFrame(PSD_ROWS).to_csv(ROOT/'SHORT_CYCLE_PSD.csv',index=False)
    pd.DataFrame(PROBE_ROWS).to_csv(ROOT/'SHORT_CYCLE_RESIDUAL_PREDICTABILITY.csv',index=False)
    write('DECISIONS.json',decision)
    for arm in fitted:
        for split in ['validation','descriptive_holdout']:plot(arm,split)
    assert all(sha(PREV/name)==h for name,h in hashes.items())
    write('RUN_SUMMARY.json',dict(status='COMPLETE',runner_wall_seconds=time.perf_counter()-START,request_to_runner_completion_seconds=time.time()-ATTACH.stat().st_ctime,included_arms=list(fitted),source_files_unchanged=True,raw_workbook_opened=False,raw_controls_read=False,base_models_refitted=False,base_search_performed=False,holdout_read_events=1,primary_comparison_histories_fixed=True,Rod1_replication_validation_only=replication))
    log('COMPLETE')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
