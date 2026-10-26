"""Entity-safe pooled conditional KC, latent Wiener link, report mature RLS."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np,pandas as pd
from sklearn.preprocessing import SplineTransformer
import rls_core as rls
L=16;PMAX=8;LAGS=np.array([1,2,4,8,16]);LINKS=['identity','signed_sqrt','asinh','signed_log1p'];FITS=0
class SupportLimited(Exception):pass
def mse(y,p):return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def ridge(x,y,penalty):
    global FITS
    x=np.asarray(x,float);y=np.asarray(y,float);required=max(200,20*(x.shape[1]+1))
    if len(y)<required:raise SupportLimited((len(y),required))
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);ym=float(y.mean());ys=max(float(y.std()),1e-8);a=(x-mu)/sd
    G=a.T@a+np.diag(penalty);rhs=a.T@((y-ym)/ys);beta=np.linalg.solve(G,rhs)
    kkt=float(np.linalg.norm(G@beta-rhs)/max(1.,np.linalg.norm(rhs),np.linalg.norm(G)*np.linalg.norm(beta)));cond=float(np.linalg.cond(G))
    if not np.isfinite(beta).all() or kkt>1e-8 or cond>1e14 or np.linalg.norm(beta)>1e6:raise FloatingPointError('Ridge certificate failed')
    FITS+=1
    return dict(mean=mu,scale=sd,ym=ym,ys=ys,beta=beta,fit_rows=len(y),parameter_count=x.shape[1]+1,certificate=dict(kkt=kkt,condition=cond))
def rp(m,x):return m['ym']+m['ys']*((x-m['mean'])/m['scale']@m['beta'])
def link_fit(y,kind):return dict(kind=kind,scale=max(float(np.median(np.abs(y))),1e-8) if kind=='asinh' else 1.)
def forward(x,w):
    x=np.asarray(x,float);kind=w['kind']
    if kind=='identity':return x
    if kind=='signed_sqrt':return np.sign(x)*np.sqrt(np.abs(x))
    if kind=='signed_log1p':return np.sign(x)*np.log1p(np.abs(x))
    return np.arcsinh(x/w['scale'])
def inverse(x,w):
    kind=w['kind']
    if kind=='identity':return x
    if kind=='signed_sqrt':out=x*np.abs(x)
    elif kind=='signed_log1p':out=np.sign(x)*np.expm1(np.abs(x))
    else:out=w['scale']*np.sinh(x)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite inverse output link')
    return out
def frame(data,head,H,start,stop):
    records=[]
    for entity in sorted(data):
        x=data[entity];o=np.arange(max(start,L),stop-H+1);assert o.max()+H-1<len(x)
        records.append(pd.DataFrame(dict(entity_id=entity,origin=o,target_end=o+H-1,y=x[o+H-1,head],anchor=x[o-1,head])))
    return pd.concat(records,ignore_index=True)
def split(frame,lo,hi):
    positions=[]
    for _,ix in frame.groupby('entity_id',sort=False).indices.items():positions.extend(ix[int(len(ix)*lo):int(len(ix)*hi)])
    return np.asarray(positions,int)
def earlier(frame,ev,lag,ends='target_end'):
    first=frame.iloc[ev].groupby('entity_id').origin.min().to_dict()
    mask=np.array([e in first and end<first[e]-lag for e,end in zip(frame.entity_id,frame[ends])]);return np.flatnonzero(mask)
def folds(frame):
    return [(earlier(frame,ev,L),ev) for ev in [split(frame,a,b) for a,b in [(.4,.6),(.6,.8),(.8,1.)]]]
def history(data,meta,frame,head,E,w):
    own=np.empty((len(frame),5));u=np.empty((len(frame),10));flow=np.empty(len(frame));latest=np.empty((len(frame),2))
    for e,ix in frame.groupby('entity_id',sort=False).indices.items():
        o=frame.iloc[ix].origin.to_numpy();x=data[e];assert (o>=L).all()
        own[ix]=forward(x[o[:,None]-LAGS,head],w);latest[ix]=x[o-1,1:3];flow[ix]=meta[e]['flow']
        if E=='sparse':v=x[o[:,None]-LAGS,1:3]
        else:v=np.stack([x[o[:,None]-np.arange(a+1,b+1),1:3].mean(1) for a,b in [(0,1),(1,2),(2,4),(4,8),(8,16)]],axis=1)
        u[ix]=v.reshape(len(ix),-1)
    anchor=own[:,0].copy();own-=anchor[:,None];own[:,0]=anchor
    return own,u,flow,latest,anchor
def design(data,meta,frame,head,E,kind,w,recipe=None):
    own,u,flow,last,anchor=history(data,meta,frame,head,E,w)
    if kind=='AR':return own,dict(kind=kind,E=E,link=w),[5,0,0],anchor
    linear=np.c_[own,u,flow]
    if kind=='ARX':return linear,dict(kind=kind,E=E,link=w),[5,11,0],anchor
    if recipe is None:
        spline=SplineTransformer(n_knots=3,degree=2,include_bias=False,extrapolation='linear').fit(last)
        z=np.c_[last,flow,anchor];recipe=dict(kind=kind,E=E,link=w,spline=spline,mean=z.mean(0),scale=np.maximum(z.std(0),1e-8))
    z=(np.c_[last,flow,anchor]-recipe['mean'])/recipe['scale'];cross=np.c_[z[:,0]*z[:,1],z[:,0]*z[:,2],z[:,1]*z[:,2],z[:,2]*z[:,3]]
    return np.c_[linear,recipe['spline'].transform(last),cross],recipe,[5,11,10],anchor
def fit_kcw(data,meta,T,head,H,variant='full',family=None,link_grid=None):
    families=[family] if family else ['AR'] if variant=='no_KC' else ['ARX','NONLINEAR_KC']
    links=link_grid if link_grid is not None else ['identity'] if variant=='no_W' else LINKS
    ledger=[];cv=folds(T)
    for kind in families:
        for E in (['sparse','blocks'] if kind=='NONLINEAR_KC' else ['sparse']):
            for link in links:
                records=[]
                for fi,ei in cv:
                    fit=T.iloc[fi];ev=T.iloc[ei];w=link_fit(fit.y,link);x,recipe,dims,anchor=design(data,meta,fit,head,E,kind,w);z,_,_,ea=design(data,meta,ev,head,E,kind,w,recipe)
                    records.append((fit,ev,x,z,recipe,dims,anchor,ea))
                for a in [1.,100.,1000.]:
                    for b in ([1.] if kind=='AR' else [1.,100.,10000.]):
                        error=0.;count=0;proof=[]
                        try:
                            for fit,ev,x,z,recipe,dims,anchor,ea in records:
                                penalty=np.r_[np.full(dims[0],a),np.full(dims[1],b),np.full(dims[2],10*b)]
                                m=ridge(x,forward(fit.y.to_numpy(),recipe['link'])-anchor,penalty);p=inverse(ea+rp(m,z),recipe['link'])
                                error+=float(np.sum((ev.y.to_numpy()-p)**2));count+=len(ev)
                                assert fit.target_end.max()<ev.origin.min()-L
                                proof.append(dict(fit_rows=len(fit),last_fit_label=int(fit.target_end.max()),first_score_dependency=int(ev.origin.min())-L,entities=fit.entity_id.nunique()))
                            ledger.append(dict(kind=kind,E=E,link=link,AR_alpha=a,input_alpha=b,MSE=error/count,status='AVAILABLE',proof=proof))
                        except SupportLimited as err:ledger.append(dict(kind=kind,E=E,link=link,AR_alpha=a,input_alpha=b,status='SUPPORT_LIMITED',reason=str(err)))
    legal=[r for r in ledger if r['status']=='AVAILABLE']
    if not legal:raise SupportLimited('No pooled KCW recipe meets all3 forward folds')
    best=min(legal,key=lambda r:(r['MSE'],r['link']!='identity',r['kind']=='NONLINEAR_KC',-r['input_alpha'],-r['AR_alpha'],r['E']))
    w=link_fit(T.y,best['link']);x,recipe,dims,anchor=design(data,meta,T,head,best['E'],best['kind'],w)
    m=ridge(x,forward(T.y.to_numpy(),w)-anchor,np.r_[np.full(dims[0],best['AR_alpha']),np.full(dims[1],best['input_alpha']),np.full(dims[2],10*best['input_alpha'])])
    return dict(recipe=recipe,model=m,selected=best,ledger=ledger,fit_last_label=int(T.target_end.max()))
def predict_kcw(data,meta,V,head,m):
    r=m['recipe'];x,_,_,anchor=design(data,meta,V,head,r['E'],r['kind'],r['link'],r)
    return inverse(anchor+rp(m['model'],x),r['link'])
def source(T,p,split_id):
    return pd.DataFrame(dict(entity_id=T.entity_id.to_numpy(),origin=T.origin.to_numpy(),target_stop_exclusive=T.target_end.to_numpy()+1,
        latest_available_target_index=T.origin.to_numpy()-1,residual=T.y.to_numpy()-p,upstream_realization_id='one_frozen_pooled_parent',split_id=split_id,residual_source_role='DEPLOYED_W_ONLINE'))
def a_score_indices(f,H):return np.flatnonzero(f.groupby('entity_id',sort=False).cumcount().to_numpy()>=PMAX+H-1)
def a_fit(f,H,report=False):
    lengths=[1,2,4,8];forgets=[.99,.995,.999] if report else [.99,.995,.999,1.];gains=[1.] if report else [0.,.25,.5,1.]
    selections=[];outer=[];parent=[];child=[]
    def select(g):
        designs={p:rls.features(g,p) for p in lengths};candidates=[]
        for p in lengths:
            errors={(a,b):[] for a in forgets for b in gains};proof=[]
            try:
                for lo,hi in [(.6,.8),(.8,1.)]:
                    ei=split(g,lo,hi);fi=earlier(g,ei,L+PMAX+H,ends='target_stop_exclusive');fi=fi[designs[p][1][fi]]
                    if not len(fi):raise SupportLimited('Empty A initializer')
                    init=rls.initialize(designs[p][0][fi],g.residual.to_numpy()[fi],fit_last_target_end=int(g.iloc[fi].target_stop_exclusive.max()-1))
                    ev=g.iloc[ei];x,native,_=rls.features(ev,p);common=a_score_indices(ev,H)
                    if len(common)<32:raise SupportLimited('Fewer than32 pooled common score rows')
                    bound=int(ev.origin.min())-L-PMAX-H
                    assert init['fit_last_target_end']<bound
                    static,_=rls.replay(ev,x,native,init,1.,update=False)
                    for forget in forgets:
                        adaptive,_=rls.replay(ev,x,native,init,forget)
                        for gain in gains:errors[(forget,gain)].extend((ev.residual.to_numpy()-static-gain*(adaptive-static))[common]**2)
                    proof.append(dict(fit_rows=len(fi),initializer_last_label=init['fit_last_target_end'],first_score_recursive_dependency=bound,score_rows=len(common),entities=ev.entity_id.nunique()))
                for (forget,gain),error in errors.items():candidates.append(dict(p=p,forget=forget,gain=gain,MSE=float(np.mean(error)),status='AVAILABLE',proof=proof))
            except (SupportLimited,rls.SupportLimited) as err:candidates.append(dict(p=p,status='SUPPORT_LIMITED',reason=str(err)))
        legal=[m for m in candidates if m['status']=='AVAILABLE']
        if not legal:raise SupportLimited('No policy has two legal pooled A inner initializers')
        best=min(legal,key=lambda m:(m['MSE'],m['gain'],-m['forget'],m['p']));selections.append(dict(fit_rows=len(g),selected=best,candidates=candidates));return best
    designs={p:rls.features(f,p) for p in lengths}
    for lo,hi in [(.6,.75),(.75,.9),(.9,1.)]:
        ei=split(f,lo,hi);fi=earlier(f,ei,L+PMAX+H,ends='target_stop_exclusive')
        try:
            best=select(f.iloc[fi]);p=best['p'];fit=fi[designs[p][1][fi]];init=rls.initialize(designs[p][0][fit],f.residual.to_numpy()[fit],fit_last_target_end=int(f.iloc[fit].target_stop_exclusive.max()-1))
            ev=f.iloc[ei];x,native,_=rls.features(ev,p);ix=a_score_indices(ev,H);static,_=rls.replay(ev,x,native,init,1.,update=False);adaptive,_=rls.replay(ev,x,native,init,best['forget'])
            correction=static+best['gain']*(adaptive-static);parent.extend(ev.residual.to_numpy()[ix]**2);child.extend((ev.residual.to_numpy()-correction)[ix]**2)
            bound=int(ev.origin.min())-L-PMAX-H;assert init['fit_last_target_end']<bound
            outer.append(dict(status='COMPLETED',selected=best,initializer_last_label=init['fit_last_target_end'],first_score_recursive_dependency=bound,score_rows=len(ix)))
        except (SupportLimited,rls.SupportLimited) as err:outer.append(dict(status='SUPPORT_LIMITED',reason=str(err)))
    best=select(f);p=best['p'];fit=np.flatnonzero(designs[p][1]);init=rls.initialize(designs[p][0][fit],f.residual.to_numpy()[fit],fit_last_target_end=int(f.iloc[fit].target_stop_exclusive.max()-1))
    pr=float(np.mean(parent)) if parent else None;ch=float(np.mean(child)) if child else None
    admitted=sum(e['status']=='COMPLETED' for e in outer)>=3 and pr-ch>16*np.finfo(float).eps*(pr+ch)
    return dict(selected=best,model=init,admitted=bool(admitted),outer=outer,selections=selections,parent_risk=pr,child_risk=ch)
def a_predict(f,policy):
    s=policy['selected'];x,native,feature_audit=rls.features(f,s['p']);static,_=rls.replay(f,x,native,policy['model'],1.,update=False);adaptive,log=rls.replay(f,x,native,policy['model'],s['forget'])
    delta=static+s['gain']*(adaptive-static)
    return delta if policy['admitted'] else np.zeros(len(f)),dict(latest_feature_label=feature_audit['latest_feature_target_end'],latest_parameter_label=log['latest_parameter_target_end'],updates=log['total_updates'],resets=log['resets'])
