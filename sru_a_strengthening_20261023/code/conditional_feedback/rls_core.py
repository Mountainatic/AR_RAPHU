"""Report-defined scaled ridge/RLS, adapted to target-window maturity.

Initialization and RLS equations match the sealed CZ implementation. Added
features, parameter-label queues, and generator resets support direct H/W.
No harmonic features or online upstream refits exist in this module.
"""
from collections import deque
import numpy as np

class SupportLimited(Exception):pass
class NumericalFailure(Exception):pass

def runs(frame):
    origin=frame.origin.to_numpy(np.int64)
    change=np.ones(len(frame),bool)
    if not len(frame):return np.zeros(0,int)
    change[1:]=np.diff(origin)!=1
    for name in ['entity_id','segment_id','upstream_realization_id','split_id']:
        if name in frame:
            values=frame[name].astype(str).to_numpy()
            change[1:]|=values[1:]!=values[:-1]
    return np.cumsum(change)-1

def features(frame,p):
    if p<1:raise ValueError('Positive memory required')
    if not set(frame.residual_source_role).issubset({'CROSS_FITTED_PREQUENTIAL','DEPLOYED_W_ONLINE'}):
        raise ValueError('Unverified residual source')
    o=frame.origin.to_numpy(np.int64);end=frame.target_stop_exclusive.to_numpy(np.int64)-1
    latest=frame.latest_available_target_index.to_numpy(np.int64)
    r=frame.residual.to_numpy(np.float64)
    if np.any(latest>=o) or np.any(end<=latest) or np.any(end<o):raise ValueError('Forecast/target maturity mismatch')
    if not np.isfinite(r).all():raise ValueError('Nonfinite residual source')
    group=runs(frame);x=np.full((len(frame),p),np.nan,np.float64)
    native=np.zeros(len(frame),bool);max_read_end=np.full(len(frame),-1,np.int64)
    for g in np.unique(group):
        ix=np.flatnonzero(group==g)
        if np.any(np.diff(end[ix])<0) or np.any(np.diff(latest[ix])<0):raise ValueError('Nonmonotone maturity within stream')
        newest=np.searchsorted(end[ix],latest[ix],side='right')-1
        loc=newest[:,None]-np.arange(p)[None,:]
        safe=np.clip(loc,0,len(ix)-1);take=ix[safe]
        valid=(loc>=0)&(end[take]<=latest[ix,None])&(o[take]<o[ix,None])
        keep=valid.all(1);dest=ix[keep]
        native[dest]=True;x[dest]=r[take[keep]]
        if len(dest):max_read_end[dest]=end[take[keep]].max(1)
    return x,native,dict(groups=group,latest_feature_target_end=max_read_end,
        no_cross_generator=True,only_mature_target_windows=True)

def initialize(x,y,*,fit_last_target_end,alpha=100.):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    required=max(200,20*(x.shape[1]+1))
    if len(x)<required:raise SupportLimited((len(x),required))
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite initializer')
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    ym=float(y.mean());ys=max(float(y.std()),1e-8)
    a=np.column_stack([np.ones(len(x)),(x-mean)/scale]);yn=(y-ym)/ys
    penalty=alpha*np.eye(a.shape[1]);penalty[0,0]=1e-6
    precision=a.T@a+penalty;rhs=a.T@yn
    P=np.linalg.inv(precision);theta=P@rhs
    kkt=float(np.linalg.norm(precision@theta-rhs)/max(1.,np.linalg.norm(rhs),np.linalg.norm(precision)*np.linalg.norm(theta)))
    cond=float(np.linalg.cond(precision))
    if not np.isfinite(P).all() or not np.isfinite(theta).all() or kkt>1e-8 or cond>1e14 or np.linalg.norm(theta)>1e6:
        raise NumericalFailure(dict(kkt=kkt,condition=cond))
    np.linalg.cholesky((P+P.T)*.5)
    return dict(mean=mean.tolist(),scale=scale.tolist(),ym=ym,ys=ys,theta=theta.tolist(),P=P.tolist(),alpha=alpha,
        fit_rows=len(y),fit_last_target_end=int(fit_last_target_end),
        certificate=dict(status='PASS',relative_kkt=kkt,condition_number=cond,positive_definite_P=True))

def replay(frame,x,native,model,forget=.999,*,update=True,update_delay=0,return_states=False):
    """Issue each forecast once; later learn from its stored issued feature.

    update_delay delays coefficient labels only. Residual feature delivery
    still follows the declared latest_available_target_index for all controls.
    """
    if not (0<forget<=1) or update_delay<0:raise ValueError('Invalid RLS policy')
    o=frame.origin.to_numpy(np.int64);end=frame.target_stop_exclusive.to_numpy(np.int64)-1
    latest=frame.latest_available_target_index.to_numpy(np.int64);r=frame.residual.to_numpy(np.float64)
    if len(x)!=len(frame) or len(native)!=len(frame):raise ValueError('Unpaired feature frame')
    if np.any(end<=latest):raise ValueError('Current target is already mature')
    group=runs(frame);out=np.zeros(len(frame));trained=np.full(len(frame),int(model['fit_last_target_end']),np.int64)
    counts=np.zeros(len(frame),int);states=[];events=[];total_updates=0;reset_count=0
    mu=np.asarray(model['mean']);sd=np.asarray(model['scale'])
    for t in range(len(frame)):
        if t==0 or group[t]!=group[t-1]:
            theta=np.array(model['theta']);P=np.array(model['P']);pending=deque()
            last=int(model['fit_last_target_end']);count=0;reset_count+=1
            if last>latest[t]:raise ValueError('Initializer contains a label unavailable at first forecast')
        while pending and pending[0][0]+update_delay<=latest[t]:
            target_end,u,a=pending.popleft()
            if target_end>latest[t]:raise AssertionError('Unmatured parameter update')
            if not update:continue
            # r[u] is first accessed for parameter learning only now, after maturity.
            y=(r[u]-model['ym'])/model['ys'];pa=P@a;den=float(forget+a@pa)
            if not np.isfinite(den) or den<=0:raise NumericalFailure('Nonpositive RLS denominator')
            gain=pa/den;theta+=gain*(y-a@theta)
            P=(P-np.outer(gain,pa))/forget;P=(P+P.T)*.5
            if not np.isfinite(theta).all() or not np.isfinite(P).all() or np.linalg.norm(theta)>1e6:
                raise NumericalFailure('RLS state nonfinite or coefficient bound exceeded')
            count+=1;total_updates+=1;last=max(last,int(target_end))
            if count%256==0:
                try:np.linalg.cholesky(P)
                except np.linalg.LinAlgError as error:raise NumericalFailure('RLS covariance lost positive definiteness') from error
            if return_states:events.append(dict(at=t,u=int(u),end=int(target_end),phi=a.copy(),target=float(y)))
        if native[t]:
            a=np.r_[1.,(x[t]-mu)/sd]
            out[t]=model['ym']+model['ys']*(a@theta)
            pending.append((int(end[t]),t,a.copy()))
        trained[t]=last;counts[t]=count
        if last>latest[t]:raise AssertionError('Parameter training label exceeds issue information')
        if return_states:states.append(dict(theta=theta.copy(),P=P.copy(),group=int(group[t])))
    if not np.isfinite(out).all():raise NumericalFailure('Nonfinite prediction')
    return out,dict(latest_parameter_target_end=trained,update_count=counts,total_updates=total_updates,
        resets=reset_count,current_target_used=False,stored_issued_features=True,states=states,events=events)
