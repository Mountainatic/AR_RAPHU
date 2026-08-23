#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRISM-CT exploratory prototype on the provided silicon-pulling data.

Input is an NPZ extracted from the uploaded XLSX using artifact_tool.
The numerical core only needs NumPy.

Models:
  Persistence
  Current-Ridge
  Lag-Ridge (lags matched to CT time constants)
  CT-Absolute-Ridge
  CT-Multires-Ridge

All Ridge models use the same normalized regularization rule:
    alpha = lambda * n_train, lambda=1 by default.
This avoids horizon-by-horizon test-driven tuning in the exploratory run.
"""
from __future__ import annotations
import argparse, csv, math
import numpy as np

DT = 2.0
TAUS = np.array([10,30,60,120,300,600,1200,2400,4800,7200.], dtype=float)
HORIZONS = [1,5,15,30,60,120,300,600]
LAGS = np.unique(np.maximum(1, np.rint(TAUS/DT).astype(int)))
START = int(LAGS.max())


def lag_features(B):
    n,p=B.shape; blocks=[B]
    for lag in LAGS:
        z=np.full((n,p), np.nan); z[lag:]=B[:-lag]; blocks.append(z)
    return np.concatenate(blocks, axis=1)


def ct_states(B):
    n,p=B.shape; out=[]
    for tau in TAUS:
        a=float(np.exp(-DT/tau)); z=np.empty((n,p)); z[0]=B[0]
        for t in range(1,n):
            z[t]=a*z[t-1]+(1-a)*B[t]
        out.append(z)
    return out


def ct_absolute(B):
    z=ct_states(B)
    return np.concatenate([B]+z, axis=1)


def ct_multires(B):
    z=ct_states(B)
    blocks=[B, B-z[0]]
    blocks += [z[i]-z[i+1] for i in range(len(z)-1)]
    return np.concatenate(blocks, axis=1)


def fit_ridge(X, d, lam=1.0):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1.0
    Xs=(X-mu)/sd
    dmu=d.mean(); dc=d-dmu
    alpha=lam*len(Xs)
    A=Xs.T@Xs + alpha*np.eye(Xs.shape[1])
    w=np.linalg.solve(A, Xs.T@dc)
    return mu,sd,w,dmu,alpha


def predict_ridge(model, X):
    mu,sd,w,dmu,_=model
    return ((X-mu)/sd)@w+dmu


def metrics(yt, yp, pp):
    mse=float(np.mean((yt-yp)**2)); pmse=float(np.mean((yt-pp)**2))
    rmse=math.sqrt(mse)
    denom=float(np.sum((yt-yt.mean())**2))
    r2=1-float(np.sum((yt-yp)**2))/denom if denom>0 else float('nan')
    skill=1-mse/pmse if pmse>0 else float('nan')
    return rmse,r2,skill


def within(F,y,h,lam):
    n=len(y); tr_end=int(.6*n); te_start=int(.8*n)
    tr=np.arange(START,tr_end-h); te=np.arange(te_start,n-h)
    d=y[tr+h]-y[tr]
    model=fit_ridge(F[tr],d,lam)
    yp=y[te]+predict_ridge(model,F[te])
    return model[-1], *metrics(y[te+h],yp,y[te])


def cross(Fs,ys,Fd,yd,h,lam):
    tr=np.arange(START,len(ys)-h); te=np.arange(START,len(yd)-h)
    d=ys[tr+h]-ys[tr]
    model=fit_ridge(Fs[tr],d,lam)
    yp=yd[te]+predict_ridge(model,Fd[te])
    return model[-1], *metrics(yd[te+h],yp,yd[te])


def persistence_within(y,h):
    n=len(y); te=np.arange(int(.8*n),n-h)
    return '', *metrics(y[te+h],y[te],y[te])


def persistence_cross(y,h):
    te=np.arange(START,len(y)-h)
    return '', *metrics(y[te+h],y[te],y[te])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--out',default='prism_ct_results.csv')
    ap.add_argument('--lambda-ridge',type=float,default=1.0)
    args=ap.parse_args()
    d=np.load(args.npz)
    B1=d['sheet1']; B2=d['sheet2']; y1=B1[:,-1]; y2=B2[:,-1]

    print('building features...')
    models={
      'Current-Ridge': (B1,B2),
      'Lag-Ridge': (lag_features(B1),lag_features(B2)),
      'CT-Absolute-Ridge': (ct_absolute(B1),ct_absolute(B2)),
      'CT-Multires-Ridge': (ct_multires(B1),ct_multires(B2)),
    }
    rows=[]
    for dom,y,idx in [('Sheet1',y1,0),('Sheet2',y2,1)]:
        for h in HORIZONS:
            al,rm,r2,sk=persistence_within(y,h)
            rows.append(['within_60_20_20',dom,'Persistence',h,h*DT,al,rm,r2,sk])
        for name,(F1,F2) in models.items():
            F=F1 if idx==0 else F2
            for h in HORIZONS:
                al,rm,r2,sk=within(F,y,h,args.lambda_ridge)
                rows.append(['within_60_20_20',dom,name,h,h*DT,al,rm,r2,sk])

    for dom,ys,yd,src in [('S1->S2',y1,y2,0),('S2->S1',y2,y1,1)]:
        for h in HORIZONS:
            al,rm,r2,sk=persistence_cross(yd,h)
            rows.append(['cross_sheet',dom,'Persistence',h,h*DT,al,rm,r2,sk])
        for name,(F1,F2) in models.items():
            Fs,Fd=(F1,F2) if src==0 else (F2,F1)
            for h in HORIZONS:
                al,rm,r2,sk=cross(Fs,ys,Fd,yd,h,args.lambda_ridge)
                rows.append(['cross_sheet',dom,name,h,h*DT,al,rm,r2,sk])

    with open(args.out,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['evaluation','domain','model','horizon_steps','horizon_seconds','alpha','rmse','r2','persistence_skill_mse'])
        w.writerows(rows)
    print('saved',args.out)

if __name__=='__main__': main()
