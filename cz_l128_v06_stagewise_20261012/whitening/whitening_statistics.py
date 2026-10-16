"""Fixed reporting statistics and AR probe; no PRISM model/route selection."""
from collections import deque
import numpy as np
import pandas as pd
from scipy.signal import welch
from statsmodels.tsa.stattools import pacf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def runs(frame, *, fold_sensitive=False):
    boundaries=(frame.entity_id!=frame.entity_id.shift())|(frame.origin.diff()!=1)
    if fold_sensitive and 'fold' in frame:
        boundaries |= frame.fold!=frame.fold.shift()
    return frame.groupby(boundaries.cumsum(),sort=False)

def acf_segmented(frame, column, maxlag=900):
    mean=float(frame[column].mean());numerator=np.zeros(maxlag+1);pairs=np.zeros(maxlag+1,dtype=int)
    for _,g in runs(frame):
        z=g[column].to_numpy(float)-mean;n=len(z);size=1<<(2*n-1).bit_length()
        f=np.fft.rfft(z,size);raw=np.fft.irfft(f*f.conj(),size)[:min(n,maxlag+1)]
        numerator[:len(raw)]+=raw;pairs[:len(raw)]+=np.arange(n,n-len(raw),-1)
    if numerator[0]<=0:raise ValueError('Undefined zero-variance whiteness statistics')
    return numerator/numerator[0],pairs

def diagnose(frame, column, **labels):
    y=frame[column].to_numpy(float);acf,pairs=acf_segmented(frame,column)
    metric=dict(labels,n=len(y),RMSE=float(np.sqrt(np.mean(y*y))),MAE=float(np.mean(abs(y))),
        bias=float(np.mean(y)),E160=float(np.sum(acf[1:161]**2)),E900=float(np.sum(acf[1:901]**2)),
        lag1=float(acf[1]),lag7=float(acf[7]),lag14=float(acf[14]),lag900_pairs=int(pairs[900]))
    acf_table=pd.DataFrame(dict(lag_samples=np.arange(901),lag_seconds=2*np.arange(901),ACF=acf,pairs=pairs))
    spectra=[];pacfs=[];energy=0.;band=0.;low=0.
    for segment,(_,g) in enumerate(runs(frame)):
        nperseg=min(1024,len(g));noverlap=min(512,nperseg//2)
        frequency,density=welch(g[column].to_numpy(float),fs=.5,nperseg=nperseg,noverlap=noverlap,detrend='constant',scaling='density')
        df=frequency[1]-frequency[0];weight=len(g)/len(frame)
        spectra.append(pd.DataFrame(dict(segment=segment,frequency_Hz=frequency,PSD=density)))
        pa=pacf(g[column].to_numpy(float),nlags=min(160,len(g)//2-1),method='ywm')
        pacfs.append(pd.DataFrame(dict(segment=segment,lag_samples=np.arange(len(pa)),lag_seconds=2*np.arange(len(pa)),PACF=pa)))
        energy+=weight*float(density.sum()*df)
        band+=weight*float(density[(frequency>=.125)&(frequency<=.155)].sum()*df)
        low+=weight*float(density[(frequency>0)&(frequency<=.02)].sum()*df)
        peak=int(np.argmax(density[1:])+1)
        if segment==0:metric.update(dominant_frequency_Hz=float(frequency[peak]),dominant_period_seconds=float(1/frequency[peak]),dominant_PSD=float(density[peak]))
    metric.update(total_spectral_energy=energy,seven_second_band_energy=band,low_frequency_energy=low,
        seven_second_band_fraction=band/energy,continuous_runs=len(spectra))
    return metric,acf_table,pd.concat(pacfs,ignore_index=True),pd.concat(spectra,ignore_index=True)

def residual_features(frame,column,H=32,*,fold_sensitive=False):
    x=np.full((len(frame),H),np.nan);cursor=0
    for _,g in runs(frame,fold_sensitive=fold_sensitive):
        buffer=deque(maxlen=H)
        for v in g[column].to_numpy(float):
            if len(buffer)==H:x[cursor]=np.asarray(buffer)[::-1]
            buffer.append(float(v));cursor+=1
    return x,np.isfinite(x).all(axis=1)

def fit_probe(frame,column,H=32,alpha=1.):
    x,mask=residual_features(frame,column,H,fold_sensitive=True);y=frame[column].to_numpy(float)[mask]
    if mask.sum()<H+2:raise ValueError('Insufficient independent CF rows for fixed probe')
    scaler=StandardScaler().fit(x[mask]);model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(x[mask]),y)
    return dict(mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=float(model.intercept_),
                fit_rows=int(mask.sum()),H=H,alpha=alpha)

def probe_predict(frame,column,model):
    x,mask=residual_features(frame,column,int(model['H']));prediction=np.zeros(len(frame))
    prediction[mask]=((x[mask]-model['mean'])/model['scale'])@model['coef']+model['intercept']
    return prediction,mask

def probe_metrics(frame,column,model,**labels):
    p,mask=probe_predict(frame,column,model);result=[]
    for support,keep in [('all_cold_start',np.ones(len(frame),bool)),('mature_history',mask)]:
        y=frame[column].to_numpy(float)[keep];error=y-p[keep]
        mse=float(np.mean(error*error));zero=float(np.mean(y*y));centered=float(np.mean((y-y.mean())**2))
        result.append(dict(labels,support=support,n=len(y),R2=1-mse/centered,probe_RMSE=float(np.sqrt(mse)),
            RMSE_improvement_vs_zero_percent=100*(1-np.sqrt(mse/zero)),H=int(model['H']),alpha=float(model['alpha'])))
    return result
