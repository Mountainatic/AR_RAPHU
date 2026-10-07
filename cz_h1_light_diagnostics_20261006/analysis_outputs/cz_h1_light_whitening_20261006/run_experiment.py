"""Independent lightweight H1 Ridge + causal residual-history experiment."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import time, json, hashlib, sys
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import welch, find_peaks
from statsmodels.tsa.stattools import pacf
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import audit_and_data as d

ROOT = d.ROOT
LS = [8,16,32,64,96,128]
ALPHAS = [1e-6,1e-4,1e-2,1.,100.]
ARMS = {'Rod1_Common4':('Rod_1',d.COMMON),'Rod1_AllLegal':('Rod_1',d.ALL),'Rod2_Common4':('Rod_2',d.COMMON)}
START = time.perf_counter()
ROWS = []

def log(*args):
    print(f'[{time.perf_counter()-START:.1f}s]',*args,flush=True)
    if time.perf_counter()-START > 3*3600: raise RuntimeError('STOP: 3h runtime budget')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metric(r):
    r=np.asarray(r,float)
    return dict(n=len(r),RMSE=float(np.sqrt(np.mean(r*r))),MAE=float(np.mean(abs(r))),bias=float(r.mean()))

def split_freeze():
    mem=pd.read_parquet(ROOT/'ORIGINAL_SOURCE_MEMBERSHIP.parquet')
    mem['split']='train'
    for rod in mem.rod.unique():
        ix=mem.index[(mem.rod==rod)&(mem.original_split=='source_validation')].to_numpy()
        half=len(ix)//2
        mem.loc[ix[:half],'split']='validation'
        mem.loc[ix[half:half+257],'split']='purged'
        mem.loc[ix[half+257:],'split']='test'
    mem.to_parquet(ROOT/'FROZEN_MEMBERSHIP.parquet',index=False)
    d.write_json('EXPERIMENT_PREREGISTRATION.json',dict(classification='POST_HOC_RESIDUAL_WHITENING_DIAGNOSTIC',user_authorized_split_amendment=True,split_rule='retain original source train exactly; source validation first floor(n/2) selection; next257 excluded; remaining tail descriptive test',membership_sha256=digest(ROOT/'FROZEN_MEMBERSHIP.parquet'),histories=LS,alphas=ALPHAS,A_histories=LS,A_alphas=ALPHAS,alpha_selection='4 expanding prequential blocks, first 1/5 initial fit',history_selection='source-validation RMSE; within 0.1% of minimum prefer shorter L',A_selection='trainOOF-only fit, minimum online cold-start validation RMSE; within 0.1% prefer shorter H, then stronger alpha',probe=dict(history=32,alpha=1.,fit='base OOF / second-stage prequential final residuals'),base_test_fit='source train only, same model as validation; no validation refit',A_test_fit='train base OOF residuals only; same model as validation',acf='segmented globally centered biased FFT estimator, never cross segments or gaps; E sums lags1..k',PSD=dict(fs=.5,nperseg=1024,noverlap=512,low_frequency_band=[0,.02],seven_second_band=[.125,.155]),test_use='only after all three recipes + probes frozen',rod1_variable_sha256=digest(ROOT/'ROD1_ALL_LEGAL_VARIABLES.json'),initial_time_unix=ROOT.stat().st_ctime))
    log('Frozen membership',mem.groupby(['rod','split']).size().to_dict())
    return mem

def points(mem,segs,rod,split,with_y=True):
    out=mem[(mem.rod==rod)&(mem.split==split)][['entity','origin']].copy().reset_index(drop=True)
    by={s.segment_id:s for s in segs[rod]}
    out['global_origin']=[by[e].raw_start+int(o) for e,o in zip(out.entity,out.origin)]
    out=out.sort_values('global_origin').reset_index(drop=True)
    if with_y:
        out['y_true']=[float(by[e].frame.crystal_diameter.iloc[o]-by[e].frame.crystal_diameter.iloc[o-1]) for e,o in zip(out.entity,out.origin)]
    return out

def folds(frame):
    edges=np.linspace(0,len(frame),6,dtype=int)
    for j in range(4):
        fit=np.arange(edges[j+1]); score=np.arange(edges[j+1],edges[j+2])
        assert frame.global_origin.iloc[fit].max() < frame.global_origin.iloc[score].min()
        yield j,fit,score

def controls(segs,fit,score,L,variables):
    d.HISTORY_STEPS=L
    origins={s.segment_id:set(fit.loc[fit.entity==s.segment_id,'origin'].astype(int)) for s in segs}
    pca=d._fit_joint_lift(segs,origins)
    transformed=d._apply_joint_lift(segs,pca)
    by={s.segment_id:s.frame[variables].to_numpy(float) for s in transformed}
    def materialize(frame):
        x=np.empty((len(frame),len(variables)*L))
        for i,(e,t) in enumerate(zip(frame.entity,frame.origin)):
            assert int(t)>=L
            x[i]=by[e][int(t)-L:int(t)][::-1].T.reshape(-1)
        assert np.isfinite(x).all()
        return x
    return materialize(fit),materialize(score),pca

class RidgeGrid:
    """Standardized Ridge, intercept unpenalized; single Gram eigensolve per grid."""
    def __init__(self,x,y):
        self.mean=x.mean(0); self.scale=x.std(0)
        self.scale[self.scale<1e-12]=1.
        z=(x-self.mean)/self.scale
        self.ym=float(np.mean(y))
        ev,vec=eigh(z.T@z,check_finite=False,driver='evd')
        self.eig=np.maximum(ev,0.)
        self.vec=vec
        self.rhs=vec.T@(z.T@(y-self.ym))
    def state(self,alpha):
        coef=self.vec@(self.rhs/(self.eig+alpha))
        return dict(mean=self.mean,scale=self.scale,coef=coef,intercept=self.ym,alpha=float(alpha))
    def predictions(self,x):
        z=(x-self.mean)/self.scale
        projection=z@self.vec
        return {a:projection@(self.rhs/(self.eig+a))+self.ym for a in ALPHAS}

def predict(model,x): return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept']

def save_model(name,state):
    np.savez_compressed(ROOT/f'{name}.npz',**state)

def select_base(arm,segs,train,val,variables):
    candidates=[]
    for L in LS:
        errors={a:[] for a in ALPHAS}
        for fold,fi,ei in folds(train):
            x,z,_=controls(segs,train.iloc[fi],train.iloc[ei],L,variables)
            grid=RidgeGrid(x,train.y_true.iloc[fi].to_numpy())
            for a,p in grid.predictions(z).items(): errors[a].extend((train.y_true.iloc[ei].to_numpy()-p)**2)
        cv={a:float(np.sqrt(np.mean(e))) for a,e in errors.items()}
        alpha=min(ALPHAS,key=lambda a:(cv[a],-a))
        x,z,pca=controls(segs,train,val,L,variables)
        grid=RidgeGrid(x,train.y_true.to_numpy())
        state=grid.state(alpha)
        pred=predict(state,z)
        vrmse=metric(val.y_true-pred)['RMSE']
        for a in ALPHAS:
            ROWS.append(dict(arm=arm,history_samples=L,history_seconds=L*2,alpha=a,train_CV_RMSE=cv[a],alpha_selected_for_history=a==alpha,validation_RMSE=vrmse if a==alpha else np.nan,selected=False,validation_used_only_after_train_alpha_selection=True))
        candidates.append(dict(L=L,alpha=alpha,model=state,pca=pca,val_pred=pred,validation_RMSE=vrmse))
        log(arm,'base L',L,'alpha',alpha,'val RMSE',round(vrmse,6))
    best_rmse=min(c['validation_RMSE'] for c in candidates)
    best=min((c for c in candidates if c['validation_RMSE']<=best_rmse*1.001),key=lambda c:c['L'])
    for r in ROWS:
        if r['arm']==arm and r['history_samples']==best['L'] and r['alpha']==best['alpha']:r['selected']=True
    save_model(arm+'_BASE',best['model'])
    d.write_json(arm+'_PCA.json',best['pca'])
    return best

def base_oof(arm,segs,train,variables,best):
    blocks=[]
    for fold,fi,ei in folds(train):
        fit=train.iloc[fi]; score=train.iloc[ei].copy()
        x,z,_=controls(segs,fit,score,best['L'],variables)
        state=RidgeGrid(x,fit.y_true.to_numpy()).state(best['alpha'])
        score['y_base_pred']=predict(state,z)
        score['r_base']=score.y_true-score.y_base_pred
        score['arm']=arm;score['fold']=fold
        score['fit_end']=int(fit.global_origin.max())
        score['score_block']=f'prequential_{fold}'
        assert (score.global_origin>score.fit_end).all()
        blocks.append(score)
    return pd.concat(blocks,ignore_index=True)

def residual_x(frame,column,H):
    x=np.full((len(frame),H),np.nan)
    buffer=deque(maxlen=H); previous=None; entity=None
    for i,(e,t,r) in enumerate(zip(frame.entity,frame.origin,frame[column])):
        if e!=entity or previous is None or int(t)!=previous+1:buffer.clear()
        if len(buffer)==H:x[i]=np.asarray(buffer)[::-1]
        buffer.append(float(r))
        previous=int(t);entity=e
    return x,np.isfinite(x).all(1)

def online(frame,model,H,column='base_residual'):
    # Predict first; only then mature current residual and append for future origins.
    correction=np.zeros(len(frame));available=np.zeros(len(frame),bool)
    buffer=deque(maxlen=H);previous=None;entity=None
    for i,(e,t,r) in enumerate(zip(frame.entity,frame.origin,frame[column])):
        if e!=entity or previous is None or int(t)!=previous+1:buffer.clear()
        if len(buffer)==H:
            correction[i]=predict(model,np.asarray(buffer)[::-1][None,:])[0]
            available[i]=True
        buffer.append(float(r));previous=int(t);entity=e
    return correction,available

def base_frame(arm,frame,pred):
    result=frame.copy();result['arm']=arm
    result['base_prediction']=pred;result['base_residual']=result.y_true-pred
    return result

def select_A(arm,oof,val):
    records=[];candidates=[]
    for H in LS:
        x,mask=residual_x(oof,'r_base',H)
        grid=RidgeGrid(x[mask],oof.r_base.to_numpy()[mask])
        for alpha in ALPHAS:
            model=grid.state(alpha)
            c,available=online(val,model,H)
            loss=metric(val.base_residual-c)['RMSE']
            records.append(dict(arm=arm,history_samples=H,history_seconds=2*H,alpha_A=alpha,fit_rows=int(mask.sum()),validation_rows=len(val),validation_cold_start_rows=int((~available).sum()),validation_RMSE=loss,selected=False))
            candidates.append(dict(H=H,alpha=alpha,model=model,validation_RMSE=loss))
    loss=min(c['validation_RMSE'] for c in candidates)
    best=min((c for c in candidates if c['validation_RMSE']<=loss*1.001),key=lambda c:(c['H'],-c['alpha']))
    for row in records:
        if row['history_samples']==best['H'] and row['alpha_A']==best['alpha']:row['selected']=True
    save_model(arm+'_A',best['model'])
    log(arm,'A selected',best['H'],best['alpha'],'val RMSE',best['validation_RMSE'])
    return best,records

def final_frame(frame,a):
    out=frame.copy();c,available=online(out,a['model'],a['H'])
    out['final_prediction']=out.base_prediction+c
    out['final_residual']=out.y_true-out.final_prediction
    out['A_state_available']=available
    return out

def probe_models(arm,oof,a):
    # No final-residual probe fit on A's in-sample errors: another ordinary prequential pass.
    x,mask=residual_x(oof,'r_base',a['H'])
    final_blocks=[]
    for fold,fi,ei in folds(oof):
        legal=fi[mask[fi]]
        state=RidgeGrid(x[legal],oof.r_base.to_numpy()[legal]).state(a['alpha'])
        block=oof.iloc[ei].copy().rename(columns={'r_base':'base_residual'})
        c,_=online(block,state,a['H'])
        block['final_residual']=block.base_residual-c
        final_blocks.append(block)
    final_oof=pd.concat(final_blocks,ignore_index=True)
    final_oof['arm']=arm
    models={}
    for stage,frame,col in [('base',oof,'r_base'),('final',final_oof,'final_residual')]:
        xx,ok=residual_x(frame,col,32)
        yy=frame[col].to_numpy()[ok]
        models[stage]=RidgeGrid(xx[ok],yy).state(1.)
        save_model(arm+'_PROBE_'+stage,models[stage])
    return models,final_oof

def acf_segmented(frame,column,maxlag=900):
    values=frame[column].to_numpy(float);mean=values.mean()
    numerator=np.zeros(maxlag+1);pairs=np.zeros(maxlag+1,int)
    for _,g in continuous_groups(frame):
        z=g[column].to_numpy(float)-mean;n=len(z)
        size=1<<(2*n-1).bit_length()
        f=np.fft.rfft(z,size);raw=np.fft.irfft(f*f.conj(),size)[:min(n,maxlag+1)]
        numerator[:len(raw)]+=raw
        pairs[:len(raw)]+=np.arange(n,n-len(raw),-1)
    return numerator/numerator[0],pairs

def continuous_groups(frame):
    boundaries=(frame.entity!=frame.entity.shift())|(frame.origin.diff()!=1)
    return frame.groupby(boundaries.cumsum(),sort=False)

def diagnose(allframes,probes):
    metrics=[];white=[];acfs=[];pacfs=[];psds=[];spectra=[];peaks=[];predictability=[]
    for (arm,split),frame in allframes.items():
        corr={};power={}
        for stage,column in [('raw','y_true'),('base','base_residual'),('final','final_residual')]:
            metrics.append(dict(arm=arm,split=split,stage=stage,**metric(frame[column])))
            acf,pairs=acf_segmented(frame,column)
            corr[stage]=acf
            for lag in range(901):acfs.append(dict(arm=arm,split=split,stage=stage,lag_samples=lag,lag_seconds=lag*2,ACF=acf[lag],pairs=int(pairs[lag])))
            # Single contiguous validation/test tail per rod; never concatenate disjoint segments for PACF.
            assert len(list(continuous_groups(frame)))==1
            pa=pacf(frame[column].to_numpy(),nlags=160,method='ywm')
            for lag,v in enumerate(pa):pacfs.append(dict(arm=arm,split=split,stage=stage,lag_samples=lag,lag_seconds=lag*2,PACF=float(v)))
            freq,psd=welch(frame[column].to_numpy(),fs=.5,nperseg=1024,noverlap=512,detrend='constant',scaling='density')
            power[stage]=(freq,psd)
            for f,p in zip(freq,psd):psds.append(dict(arm=arm,split=split,stage=stage,frequency_Hz=f,period_seconds=1/f if f>0 else np.nan,PSD=p))
            peak,_=find_peaks(psd);peak=peak[freq[peak]>0]
            strongest=sorted(peak,key=lambda i:psd[i],reverse=True)[:8]
            i=int(np.argmax(psd[1:])+1);df=freq[1]-freq[0]
            spectra.append(dict(arm=arm,split=split,stage=stage,dominant_frequency_Hz=freq[i],dominant_period_seconds=1/freq[i],dominant_PSD=psd[i],low_frequency_energy=float(psd[(freq>0)&(freq<=.02)].sum()*df),seven_second_band_energy=float(psd[(freq>=.125)&(freq<=.155)].sum()*df),total_energy=float(psd.sum()*df),welch_segments=1+(len(frame)-1024)//512))
            for rank,j in enumerate(strongest,1):peaks.append(dict(arm=arm,split=split,stage=stage,peak_type='PSD',rank=rank,frequency_Hz=freq[j],period_seconds=1/freq[j],value=psd[j]))
            inds,_=find_peaks(abs(acf[1:]));inds+=1
            for rank,j in enumerate(sorted(inds,key=lambda j:abs(acf[j]),reverse=True)[:8],1):peaks.append(dict(arm=arm,split=split,stage=stage,peak_type='ACF',rank=rank,lag_samples=j,lag_seconds=2*j,value=acf[j]))
        w=dict(arm=arm,split=split,n=len(frame))
        for k in [160,900]:
            for stage in corr:w[f'E{k}_{stage}']=float(np.sum(corr[stage][1:k+1]**2))
            w[f'E{k}_reduction_percent']=100*(1-w[f'E{k}_final']/w[f'E{k}_base'])
        for lag in [1,7,14]:
            for stage in corr:w[f'lag{lag}_{stage}']=float(corr[stage][lag])
        white.append(w)
        for stage,column in [('base','base_residual'),('final','final_residual')]:
            p,available=online(frame,probes[arm][stage],32,column)
            y=frame[column].to_numpy()
            for support,mask in [('all_cold_start',np.ones(len(frame),bool)),('mature_history',available)]:
                yy=y[mask];err=yy-p[mask];mean=probes[arm][stage]['intercept']
                rmse=metric(err)['RMSE'];zero=metric(yy)['RMSE'];mb=metric(yy-mean)['RMSE']
                predictability.append(dict(arm=arm,split=split,stage=stage,support=support,n=len(yy),probe_H=32,probe_alpha=1.,R2=1-float(np.sum(err**2)/np.sum((yy-yy.mean())**2)),probe_RMSE=rmse,zero_baseline_RMSE=zero,train_mean_baseline_RMSE=mb,RMSE_improvement_vs_zero_percent=100*(1-rmse/zero),RMSE_improvement_vs_train_mean_percent=100*(1-rmse/mb)))
        for typ in ['ACF','PSD']:
            fig,axes=plt.subplots(1,2,figsize=(13,4.5))
            for stage,color in [('raw','#8b98a6'),('base','#2676b8'),('final','#dc6e35')]:
                for ax in axes:
                    if typ=='ACF':ax.plot(np.arange(901)*2,corr[stage],label=stage,color=color,lw=1.)
                    else:ax.semilogy(power[stage][0],power[stage][1],label=stage,color=color,lw=1.)
            if typ=='ACF':
                axes[0].set_xlim(0,320);axes[1].set_xlim(0,1800)
                for ax in axes:
                    ax.axhline(1.96/np.sqrt(len(frame)),color='gray',ls='--',lw=.7)
                    ax.axhline(-1.96/np.sqrt(len(frame)),color='gray',ls='--',lw=.7)
                    ax.set_xlabel('Lag (seconds; lag 7 = 14 s)');ax.set_ylabel('ACF')
            else:
                axes[0].set_xlim(0,.25);axes[1].set_xlim(.10,.18)
                for ax in axes:ax.axvspan(.125,.155,color='gray',alpha=.12);ax.set_xlabel('Frequency (Hz; 0.14 Hz ~ 7 s)');ax.set_ylabel('PSD (raw units squared / Hz)')
            for ax in axes:ax.legend();ax.grid(alpha=.15)
            fig.suptitle(f'{arm} | {split} | {typ} | n={len(frame)}')
            fig.tight_layout();fig.savefig(ROOT/f'{arm}_{split}_{typ}.png',dpi=160);plt.close(fig)
        log('diagnosed',arm,split)
    for name,rows in [('METRICS_ALL_ARMS',metrics),('WHITENING_METRICS',white),('ACF_ALL_ARMS',acfs),('PACF_ALL_ARMS',pacfs),('PSD_ALL_ARMS',psds),('SPECTRAL_SUMMARY',spectra),('TOP_PEAKS',peaks),('RESIDUAL_PREDICTABILITY',predictability)]:pd.DataFrame(rows).to_csv(ROOT/f'{name}.csv',index=False)

def main():
    if (ROOT/'GLOBAL_SELECTION_FREEZE.json').exists():raise RuntimeError('refuse overwrite frozen experiment')
    mem=split_freeze();_,segs=d.load()
    fitted={};allframes={};oof_all=[];a_rows=[];probes={};final_oof_all=[]
    for arm,(rod,variables) in ARMS.items():
        train=points(mem,segs,rod,'train');val=points(mem,segs,rod,'validation')
        best=select_base(arm,segs[rod],train,val,variables)
        oof=base_oof(arm,segs[rod],train,variables,best);oof_all.append(oof)
        valframe=base_frame(arm,val,best['val_pred'])
        a,records=select_A(arm,oof,valframe);a_rows.extend(records)
        probes[arm],final_oof=probe_models(arm,oof,a);final_oof_all.append(final_oof)
        allframes[arm,'validation']=final_frame(valframe,a)
        fitted[arm]=dict(base=best,A=a,train=train,rod=rod,variables=variables)
        log('development done',arm)
    pd.DataFrame(ROWS).to_csv(ROOT/'BASE_MODEL_SELECTION.csv',index=False)
    pd.DataFrame(a_rows).to_csv(ROOT/'A_MODEL_SELECTION.csv',index=False)
    pd.concat(oof_all,ignore_index=True).to_parquet(ROOT/'BASE_OOF_POINTWISE.parquet',index=False)
    pd.concat(final_oof_all,ignore_index=True).to_parquet(ROOT/'FINAL_PREQUENTIAL_POINTWISE.parquet',index=False)
    freeze={arm:dict(rod=c['rod'],variables=c['variables'],L=c['base']['L'],alpha_base=c['base']['alpha'],H=c['A']['H'],alpha_A=c['A']['alpha'],base_fit_rows=len(c['train'])) for arm,c in fitted.items()}
    d.write_json('GLOBAL_SELECTION_FREEZE.json',dict(status='FROZEN_BEFORE_TEST_TARGET_CONSTRUCTION_OR_SCORING',arms=freeze,probe_history=32,probe_alpha=1.,artifacts={p.name:digest(p) for p in ROOT.glob('*.npz')},variable_list_sha256=digest(ROOT/'ROD1_ALL_LEGAL_VARIABLES.json'),membership_sha256=digest(ROOT/'FROZEN_MEMBERSHIP.parquet')))
    log('ALL SELECTIONS FROZEN; opening test targets once per rod')
    test_targets={rod:points(mem,segs,rod,'test') for rod in segs}
    for arm,c in fitted.items():
        # Apply exact saved source-training PCA; no test fit of any preprocessing.
        d.HISTORY_STEPS=c['base']['L']
        transformed=d._apply_joint_lift(segs[c['rod']],c['base']['pca'])
        by={s.segment_id:s.frame[c['variables']].to_numpy(float) for s in transformed}
        test=test_targets[c['rod']]
        L=c['base']['L']
        x=np.vstack([by[e][int(t)-L:int(t)][::-1].T.reshape(-1) for e,t in zip(test.entity,test.origin)])
        model=dict(np.load(ROOT/f'{arm}_BASE.npz'))
        a_model=dict(np.load(ROOT/f'{arm}_A.npz'))
        afit=dict(c['A']);afit['model']=a_model
        allframes[arm,'test']=final_frame(base_frame(arm,test,predict(model,x)),afit)
    for split in ['validation','test']:
        pd.concat([f for (a,s),f in allframes.items() if s==split],ignore_index=True).to_parquet(ROOT/f'POINTWISE_{split.upper()}.parquet',index=False)
    diagnose(allframes,probes)
    d.write_json('RUN_SUMMARY.json',dict(status='COMPLETE',runner_wall_seconds=time.perf_counter()-START,audit_to_completion_wall_seconds=time.time()-ROOT.stat().st_ctime,model_grid_base=90,model_grid_A=90,cpu_thread_limit=2,classification='POST_HOC_RESIDUAL_WHITENING_DIAGNOSTIC',recipes=freeze,python=sys.version,versions={m:__import__(m).__version__ for m in ['numpy','pandas','scipy','sklearn','statsmodels']},test_target_open_events=2,train_test_origin_disjoint=True,raw_control_history_max_index='t-1',A_history_max_residual_origin='t-1',canonical_runner_imported=False))
    log('COMPLETE')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
