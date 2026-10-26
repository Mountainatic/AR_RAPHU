"""Current-X soft sensing: finite Urysohn KC, scalar W, mature report RLS A.

Experimental redesign, not unchanged PRISM v0.6. Output label at t is withheld.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import inspect
import numpy as np,pandas as pd
from stable_spline import StableLinearSpline as SplineTransformer
import legacy as old
import mature_a
from legacy import rls,source,split,mse,SupportLimited
L=128;PMAX=32;DELAY=4;SUPPORT=131;FITS=0;LINKS=['identity','signed_sqrt','asinh','signed_log1p']
old.L=L;old.PMAX=PMAX
# Adapt only registered outer partition geometry to short Debutanizer record.
# Restore original long-record outer layout for this SRU-only comparison.

def configure(delay):
    global DELAY,SUPPORT
    DELAY=delay;SUPPORT=L+delay-1;old.L=SUPPORT;mature_a.configure(delay,SUPPORT)

def frame(D,start,stop):
    f=old.frame(D,0,1,start,stop)
    for e,ix in f.groupby('entity_id',sort=False).indices.items():f.loc[ix,'anchor']=D[e][f.iloc[ix].origin.to_numpy()-DELAY,0]
    return f

def source(T,p,split_id):
    f=old.source(T,p,split_id);f['latest_available_target_index']=f.origin.to_numpy()-DELAY;f['issued_q']=np.asarray(p);return f
def earlier(T,ev,lag=None,ends='target_end'):return old.earlier(T,ev,SUPPORT if lag is None else lag,ends)
def folds(T):return [(earlier(T,ix),ix) for ix in [split(T,a,b) for a,b in [(.7,.85),(.85,1.)]]]

def raw(D,V):
    own=[];u=[]
    assert V.entity_id.nunique()==1
    a=D[V.entity_id.iloc[0]];o=V.origin.to_numpy()
    own=a[o[:,None]-np.arange(DELAY,DELAY+L),0]
    u=a[o[:,None]-np.arange(L)[None,:],1:]
    # u begins with x(t); own begins with y(t-1). These offsets are deliberate.
    return own,u

def design(D,V,config,recipe=None):
    own,u=raw(D,V);kind=config['kind'];n=len(V);current=u[:,0,:]
    if kind=='OWN':return own,None
    if kind=='DENSE':return np.c_[own,u.reshape(n,-1)],None
    if kind=='ARX':return np.c_[own,u[:,[0,1,4,7],:].reshape(n,-1)],None
    memory=config.get('memory',False);with_own=config.get('own',True)
    states=own[:,[0,L-1]] if with_own else np.empty((n,0))
    if kind=='LINEAR':return np.c_[states,current,u[:,1:].mean(1)] if memory else np.c_[states,current],None
    if recipe is None:recipe=dict(spline=SplineTransformer(n_knots=3,degree=1,include_bias=False,extrapolation='linear').fit(current))
    basis=recipe['spline'].transform(u.reshape(-1,u.shape[-1])).reshape(n,L,-1)
    x=np.c_[states,basis[:,0],basis[:,1:].mean(1)] if memory else np.c_[states,basis[:,0]]
    return x,recipe

def fit_parent(D,T,config,link):
    global FITS
    x,recipe=design(D,T,config);w=old.link_fit(T.y,link);target=old.forward(T.y.to_numpy(),w)
    obj=old.ridge(x,target,np.full(x.shape[1],config['alpha']));FITS+=1
    return dict(config=config,recipe=recipe,ridge=obj,coordinate=w,certificate=dict(rows=obj['fit_rows'],parameters=obj['parameter_count'],**obj['certificate']))

def predict_parent(D,V,obj):
    x,_=design(D,V,obj['config'],obj['recipe']);return old.inverse(old.rp(obj['ridge'],x),obj['coordinate'])

def w_features(q,recipe=None):
    q=np.asarray(q).reshape(-1,1)
    if recipe is None:recipe=dict(spline=SplineTransformer(n_knots=3,degree=1,include_bias=False,extrapolation='linear').fit(q))
    return np.c_[q,recipe['spline'].transform(q)],recipe

def w_fit(q,y):
    global FITS
    x,recipe=w_features(q);obj=old.ridge(x,y-q,np.full(x.shape[1],100.));FITS+=1
    return dict(recipe=recipe,ridge=obj,certificate=dict(rows=obj['fit_rows'],parameters=obj['parameter_count'],**obj['certificate']))

def w_predict(q,obj):
    if obj is None:return q
    x,_=w_features(q,obj['recipe']);return q+old.rp(obj['ridge'],x)

def calibration(D,T,config):
    q=[];y=[];proof=[]
    for a,b in [(.6,.8),(.8,1.)]:
        ev=split(T,a,b);fi=earlier(T,ev);fit=T.iloc[fi];score=T.iloc[ev]
        obj=fit_parent(D,fit,config,'identity');q.extend(predict_parent(D,score,obj));y.extend(score.y.to_numpy())
        assert fit.target_end.max()<score.origin.min()-SUPPORT
        proof.append(dict(last_fit_label=int(fit.target_end.max()),first_score_dependency=int(score.origin.min())-SUPPORT,certificate=obj['certificate']))
    return np.asarray(q),np.asarray(y),proof

def configs(variant,family):
    if family in ['OWN','ARX']:return [dict(kind=family,alpha=a) for a in [.01,1.,100.,1000.]]
    if variant=='no_KC':return configs(variant,'OWN')
    return configs(variant,'OWN')+[dict(kind=k,alpha=a,own=own,memory=mem) for k in ['LINEAR','URYSOHN'] for own in [False,True] for mem in [False,True] for a in [1.,100.,1000.]]

def fit_kcw(D,T,variant='full',family=None,allow_calibration=True):
    ledger=[];links=['identity'] if variant=='no_W' else LINKS
    for config in configs(variant,family):
        modes=links+(['CF_spline'] if variant!='no_W' and allow_calibration else [])
        for mode in modes:
            err=[];proof=[]
            try:
                for fi,ei in folds(T):
                    fit=T.iloc[fi];ev=T.iloc[ei];obj=fit_parent(D,fit,config,'identity' if mode=='CF_spline' else mode);q=predict_parent(D,ev,obj);w=None;wp=[]
                    if mode=='CF_spline':
                        cq,cy,wp=calibration(D,fit,config);w=w_fit(cq,cy)
                    p=w_predict(q,w);err.extend((ev.y.to_numpy()-p)**2)
                    assert fit.target_end.max()<ev.origin.min()-SUPPORT
                    proof.append(dict(last_fit_label=int(fit.target_end.max()),first_score_dependency=int(ev.origin.min())-SUPPORT,parent=obj['certificate'],W=None if w is None else w['certificate'],W_CF=wp))
                ledger.append(dict(config=config,mode=mode,status='AVAILABLE',MSE=float(np.mean(err)),proof=proof))
            except (SupportLimited,rls.SupportLimited) as e:ledger.append(dict(config=config,mode=mode,status='SUPPORT_LIMITED',reason=str(e)))
            except FloatingPointError as e:ledger.append(dict(config=config,mode=mode,status='NUMERICALLY_REJECTED',reason=str(e)))
    legal=[x for x in ledger if x['status']=='AVAILABLE'];assert legal
    best=min(legal,key=lambda x:(x['MSE'],x['mode']!='identity',str(x['config'])));obj=fit_parent(D,T,best['config'],'identity' if best['mode']=='CF_spline' else best['mode']);w=None;wp=[]
    if best['mode']=='CF_spline':
        q,y,wp=calibration(D,T,best['config']);w=w_fit(q,y)
    return dict(parent=obj,W=w,selected=best,ledger=ledger,W_final_CF=wp)

def predict_kcw(D,V,obj):return w_predict(predict_parent(D,V,obj['parent']),obj['W'])
def a_fit(f,report=False):return old.a_fit(f,DELAY,report=True) if report else mature_a.fit(f)
def a_predict(f,obj):return mature_a.predict(f,obj) if obj.get('family')=='MATURE_ISSUED_Q_BIAS_RLS' else old.a_predict(f,obj)


def fixed_kcw(D,T,row):
    config=row['config'];mode=row['mode'];p=fit_parent(D,T,config,'identity' if mode=='CF_spline' else mode);w=None;wp=[]
    if mode=='CF_spline':
        q,y,wp=calibration(D,T,config);w=w_fit(q,y)
    return dict(parent=p,W=w,selected=row,ledger=[],W_final_CF=wp)

def joint_train(D,T,cal,variant,family=None):
    """One train-only end-to-end choice; dev never used. One parent per A stream."""
    seed=fit_kcw(D,T,variant,family=family);legal=[r for r in seed['ledger'] if r['status']=='AVAILABLE'];rows=[]
    # Per response family retain identity and one train-selected nonidentity W.
    for kind in ([family] if family else ['OWN','LINEAR','URYSOHN']):
        for nonlinear in [False,True]:
            candidates=[r for r in legal if r['config']['kind']==kind and (r['mode']!='identity')==nonlinear]
            if candidates:rows.append(min(candidates,key=lambda r:(r['MSE'],str(r['config']))))
    candidates=[];objects=[]
    # Equal fixed mature score rows for all candidates; cold resets per outer block.
    score=np.concatenate([split(cal,a,b)[PMAX+DELAY-1:] for a,b in [(.6,.75),(.75,.9),(.9,1.)]])
    for row in rows:
        u=fixed_kcw(D,T,row);p=predict_kcw(D,cal,u);f=source(cal,p,'joint_Acal_one_parent');a=None if variant=='no_A' else a_fit(f,report=variant=='report_A')
        parent=float(np.mean((cal.y.to_numpy()[score]-p[score])**2))
        # A outer risk includes cold resets and legal nested policy selection.
        if a is not None:assert abs(parent-a['parent_risk'])<=1e-10*max(1.,parent)
        risk=a['child_risk'] if a is not None and a['admitted'] else parent
        candidates.append(dict(kind=row['config']['kind'],KCW_selected=row,A=a,selection_MSE=risk,no_A=variant=='no_A'))
        objects.append(dict(upper=u,A=None if variant=='no_A' else a))
    best=min(range(len(candidates)),key=lambda i:(candidates[i]['selection_MSE'],candidates[i]['kind']!='OWN'))
    obj=objects[best];obj['joint_candidates']=candidates;obj['KCW_ledger']=seed['ledger'];obj['joint_selected_index']=best
    return obj
