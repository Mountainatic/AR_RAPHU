"""Bounded-memory sequential deployment of the frozen finite-response model.

One call receives only current X and the single newly mature target. Training
ledgers and candidate models are excluded. Linear splines are compiled into
piecewise affine functions; A keeps stored-issued feature vectors until labels
mature. No fitting or parameter selection is performed here.
"""
from collections import deque
import copy
import numpy as np

def compact_model(obj):
    u=obj['upper'];p=u['parent'];r=p['ridge']
    parent=dict(config=copy.deepcopy(p['config']),recipe=copy.deepcopy(p['recipe']),
                ridge={k:copy.deepcopy(r[k]) for k in ('mean','scale','ym','ys','beta')},
                coordinate=copy.deepcopy(p['coordinate']))
    w=None
    if u['W'] is not None:
        wr=u['W']['ridge'];w=dict(recipe=copy.deepcopy(u['W']['recipe']),ridge={k:copy.deepcopy(wr[k]) for k in ('mean','scale','ym','ys','beta')})
    a=None
    if obj['A'] is not None and obj['A']['admitted']:
        src=obj['A'];am=src['model']
        keys=('fit_last_target_end',) if src['selected']['family']=='TRANSPORT' else ('mean','scale','ym','ys','theta','P','fit_last_target_end')
        a=dict(selected=copy.deepcopy(src['selected']),admitted=True,
               model={k:copy.deepcopy(am[k]) for k in keys})
    return dict(upper=dict(parent=parent,W=w),A=a)

class LinearSplines:
    def __init__(self,transformer):
        assert transformer.degree==1 and not transformer.include_bias
        self.pieces=[]
        breaks=[np.unique(bs.t[bs.k:-bs.k]) for bs in transformer.bsplines_]
        assert len({len(k) for k in breaks})==1
        # Compile the transform actually used by the frozen environment. In
        # sklearn1.7.1 its multi-feature linear-extrapolation implementation
        # mutates a local degree across columns. Direct BSpline evaluation would
        # silently alter those frozen predictions, so use transform probes.
        probes=np.column_stack(breaks);actual=transformer.transform(probes)
        offset=0
        for bs,knots in zip(transformer.bsplines_,breaks):
            width=bs.c.shape[1]-1;values=actual[:,offset:offset+width];offset+=width
            assert len(knots)>1 and np.all(np.diff(knots)>0)
            slopes=np.diff(values,axis=0)/np.diff(knots)[:,None]
            self.pieces.append((knots,values,slopes))
    def __call__(self,x):
        out=[]
        for v,(knots,values,slopes) in zip(x,self.pieces):
            i=min(max(int(np.searchsorted(knots,v,side='right'))-1,0),len(knots)-2)
            out.extend(values[i]+slopes[i]*(v-knots[i]))
        return np.asarray(out,dtype=np.float64)

def ridge_predict(model,x):
    return model['ym']+model['ys']*((x-model['mean'])/model['scale']@model['beta'])

def inverse(x,coordinate):
    kind=coordinate['kind']
    if kind=='identity':return x
    if kind=='signed_sqrt':return x*abs(x)
    if kind=='signed_log1p':return np.sign(x)*np.expm1(abs(x))
    return coordinate['scale']*np.sinh(x)

class StreamingPRISM:
    def __init__(self,obj,L,delay,x_past,y_mature_past,first_origin,temporal_bins=None):
        self.obj=compact_model(obj);self.L=L;self.delay=delay;self.t=0
        assert len(x_past)==L-1 and len(y_mature_past)==L-1
        p=self.obj['upper']['parent'];self.config=p['config'];self.kind=self.config['kind']
        self.spline=LinearSplines(p['recipe']['spline']) if self.kind=='URYSOHN' else None
        self.temporal_bins=temporal_bins
        self.history=deque((self.spline(v) if self.spline and temporal_bins is None else np.asarray(v,dtype=np.float64) for v in x_past),maxlen=L)
        self.own=deque(map(float,y_mature_past),maxlen=L)
        w=self.obj['upper']['W'];self.wspline=LinearSplines(w['recipe']['spline']) if w else None
        self.q={};self.residual={};self.pending=deque();self.updates=0
        self.A=self.obj['A']
        if self.A:
            am=self.A['model'];self.theta=np.array(am.get('theta',[]),dtype=np.float64);self.P=np.array(am.get('P',[]),dtype=np.float64)
            self.mu=np.array(am.get('mean',[]));self.sd=np.array(am.get('scale',[]));self.bias=float(am.get('ym',0.))
            assert am['fit_last_target_end']<=first_origin-delay

    def upper(self,x,mature_y):
        self.own.append(float(mature_y));self.history.append(self.spline(x) if self.spline and self.temporal_bins is None else np.asarray(x,dtype=np.float64))
        own=np.asarray(self.own)[::-1];hist=np.asarray(self.history)[::-1];cfg=self.config
        if self.kind=='OWN':design=own
        elif self.kind=='DENSE':design=np.r_[own,hist.reshape(-1)]
        elif self.kind=='ARX':design=np.r_[own,hist[[0,1,4,7]].reshape(-1)]
        else:
            states=own[[0,self.L-1]] if cfg.get('own',True) else np.empty(0)
            if self.temporal_bins is not None:
                blocks=np.stack([hist[lo:hi].mean(0) for lo,hi in self.temporal_bins])
                transformed=np.stack([self.spline(v) for v in blocks]) if self.spline else blocks
                design=np.r_[states,transformed.reshape(-1)]
            else:design=np.r_[states,hist[0],hist[1:].mean(0)] if cfg.get('memory',False) else np.r_[states,hist[0]]
        p=self.obj['upper']['parent'];q=inverse(float(ridge_predict(p['ridge'],design)),p['coordinate'])
        w=self.obj['upper']['W']
        if w:
            xx=np.r_[q,self.wspline([q])];q+=float(ridge_predict(w['ridge'],xx))
        return float(q)

    def a_features(self,q):
        s=self.A['selected'];family=s['family'];p=s['p'];t=self.t
        rs=[t-self.delay-j for j in range(p)]
        if any(i not in self.residual for i in rs):return None
        r=np.array([self.residual[i] for i in rs])
        if family=='RLS':return r
        if family=='Q_RLS':return np.r_[q,r]
        lags=[1,4,8] if family in ('QD8_RLS','QC8_RLS') else [1,4,8,16,32]
        if t<max(lags):return None
        delta=q-np.array([self.q[t-j] for j in lags])
        cross=q*delta[[-1] if len(lags)==3 else [-2,-1]]
        x=np.r_[q,delta,cross,r]
        if family in ('QC8_RLS','QC32_RLS'):
            cols=[0,2,3 if family=='QC8_RLS' else 5];x=np.r_[x,x[-1]*x[cols]]
        return x

    def step(self,x_current,y_newly_mature):
        q=self.upper(np.asarray(x_current,dtype=np.float64),y_newly_mature);t=self.t;self.q[t]=q
        mature=t-self.delay
        if mature>=0:self.residual[mature]=float(y_newly_mature)-self.q[mature]
        correction=0.
        if self.A:
            spec=self.A['selected'];am=self.A['model']
            if spec['family']=='BIAS':
                if mature>=0:self.bias+=(self.residual[mature]-self.bias)/spec['tau'];self.updates+=int(mature>=0)
                correction=spec['gain']*self.bias
            elif spec['family']=='TRANSPORT':
                indices=[mature-j for j in range(spec['p'])]
                if all(j in self.residual for j in indices):correction=spec['gain']*float(np.mean([self.residual[j] for j in indices]))
            else:
                while self.pending and self.pending[0][0]<=mature:
                    j,phi=self.pending.popleft();assert j==mature
                    target=(self.residual[j]-am['ym'])/am['ys'];pa=self.P@phi;den=float(spec['forget']+phi@pa)
                    if not np.isfinite(den) or den<=0:raise FloatingPointError('Nonpositive RLS denominator')
                    gain=pa/den;innovation=target-phi@self.theta
                    if spec.get('innovation_clip') is not None:innovation=float(np.clip(innovation,-spec['innovation_clip'],spec['innovation_clip']))
                    self.theta+=gain*innovation;self.P=(self.P-np.outer(gain,pa))/spec['forget'];self.P=(self.P+self.P.T)*.5
                    if not np.isfinite(self.theta).all() or not np.isfinite(self.P).all() or np.linalg.norm(self.theta)>1e6:raise FloatingPointError('RLS state invalid')
                    self.updates+=1
                    if self.updates%256==0:np.linalg.cholesky(self.P)
                x=self.a_features(q)
                if x is not None:
                    phi=np.r_[1.,(x-self.mu)/self.sd];correction=spec['gain']*(am['ym']+am['ys']*(phi@self.theta));self.pending.append((t,phi.copy()))
        # Exactly bounded history: residual p<=8, q lag<=32 and maturity queue.
        keep=t-max(32,self.delay+8)-2
        for store in (self.q,self.residual):
            for key in list(store):
                if key<keep:del store[key]
        self.t+=1;prediction=q+correction
        if not np.isfinite(prediction):raise FloatingPointError('Nonfinite prediction')
        return prediction

def run_stream(obj,L,delay,values,start,stop,temporal_bins=None):
    state=StreamingPRISM(obj,L,delay,values[start-L+1:start,1:],values[start-delay-L+1:start-delay,0],start,temporal_bins)
    result=np.array([state.step(values[t,1:],values[t-delay,0]) for t in range(start,stop)])
    return result,state
