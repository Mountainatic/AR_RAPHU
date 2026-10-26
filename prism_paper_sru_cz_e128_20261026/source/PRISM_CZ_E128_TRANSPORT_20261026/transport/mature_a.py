"""Explicit A redesign: low-variance mature bias observer or shrunk report RLS.

All labels delivered through the same stored-issued queue; no current y access.
"""
import numpy as np
import rls_core as rls
class SupportLimited(Exception):pass
PMAX=32;L=8;DELAY=4
SPECS_RLS=[dict(family='RLS',p=p,window=w,forget=f,gain=g) for p in [1,2,4,8] for w in [256,512,0] for f in [.99,.995,.999] for g in [.25,.5,1.]]
SPECS_BIAS=[dict(family='BIAS',p=0,window=w,tau=t,gain=g) for w in [256,512,0] for t in [32,128,512,2048] for g in [.25,.5,1.]]
SPECS_Q=[dict(family='Q_RLS',p=p,window=w,forget=f,gain=g) for p in [0,1,4] for w in [256,512,0] for f in [.99,.995,.999] for g in [.25,.5,1.]]
RESPONSE_FAMILIES=['Q_RLS','QD8_RLS','QD32_RLS','QC8_RLS','QC32_RLS']
SPECS_DYNAMIC=[dict(family=k,p=p,window=w,forget=f,gain=g) for k in ['QD8_RLS','QD32_RLS'] for p in [0,1] for w in [256,0] for f in [.995,.999] for g in [.5,1.]]
SPECS_CONDITIONAL=[dict(family=k,p=1,window=w,forget=f,gain=g) for k in ['QC8_RLS','QC32_RLS'] for w in [512,0] for f in [.995,.999] for g in [.5,1.]]

def configure(delay,support):
    global DELAY,L
    DELAY=delay;L=support

def split(T,lo,hi):
    return np.concatenate([ix[int(len(ix)*lo):int(len(ix)*hi)] for _,ix in T.groupby('entity_id',sort=False).indices.items()])

def earlier(T,ev):
    first=T.iloc[ev].groupby('entity_id').origin.min().to_dict();gap=L+PMAX+DELAY
    return np.flatnonzero([e in first and end<first[e]-gap for e,end in zip(T.entity_id,T.target_stop_exclusive)])

def common(f):return np.flatnonzero(f.groupby('entity_id',sort=False).cumcount().to_numpy()>=PMAX+DELAY-1)

def issued_features(f,p):
    q=f.issued_q.to_numpy(float)[:,None];assert np.isfinite(q).all()
    if not p:return q,np.ones(len(f),bool),dict(latest_feature_target_end=np.full(len(f),-1,int))
    x,n,a=rls.features(f,p);return np.c_[q,x],n,a

def dynamic_features(f,p,family):
    if family=='Q_RLS':return issued_features(f,p)
    if family in ['QC8_RLS','QC32_RLS']:
        assert p==1
        base='QD8_RLS' if family=='QC8_RLS' else 'QD32_RLS';x,n,a=dynamic_features(f,1,base);cols=[0,2,3 if family=='QC8_RLS' else 5]
        # Feedback gain varies with current response and transient direction.
        return np.c_[x,x[:,-1,None]*x[:,cols]],n,a
    q=f.issued_q.to_numpy(float);assert np.isfinite(q).all();lags=[1,4,8] if family=='QD8_RLS' else [1,4,8,16,32]
    groups=rls.runs(f);x=np.full((len(f),1+len(lags)+(1 if family=='QD8_RLS' else 2)),np.nan);native=np.zeros(len(f),bool)
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group);local=np.arange(max(lags),len(ix));take=ix[local];delta=q[take,None]-q[ix[local[:,None]-np.array(lags)]]
        # Interactions vanish in a steady response; they model transient gain.
        cross=q[take,None]*delta[:,[-1] if family=='QD8_RLS' else [-2,-1]]
        x[take]=np.c_[q[take],delta,cross];native[take]=True
    audit=dict(latest_feature_target_end=np.full(len(f),-1,int))
    if p:
        r,n,audit=rls.features(f,p);x=np.c_[x,r];native&=n
    return x,native,audit

def all_designs(f):
    d={p:rls.features(f,p) for p in [1,2,4,8]}
    d.update({('Q_RLS',p):issued_features(f,p) for p in [0,1,4]})
    d.update({(k,p):dynamic_features(f,p,k) for k in ['QD8_RLS','QD32_RLS'] for p in [0,1]})
    d.update({(k,1):dynamic_features(f,1,k) for k in ['QC8_RLS','QC32_RLS']});return d

def initialize(f,ix,spec,designs):
    p=spec['p'];take=ix.copy()
    key=(spec['family'],p) if spec['family'] in RESPONSE_FAMILIES else p
    if p or spec['family'] in RESPONSE_FAMILIES:take=take[designs[key][1][take]]
    if spec['window']:
        # Keep the most recent legal training examples per physical stream.
        keep=[]
        for _,loc in f.iloc[take].groupby('entity_id',sort=False).indices.items():keep.extend(take[loc[-spec['window']:]])
        take=np.asarray(keep,int)
    if not len(take):raise SupportLimited('Empty legal initializer')
    x=designs[key][0][take] if p or spec['family'] in RESPONSE_FAMILIES else np.zeros((len(take),1))
    obj=rls.initialize(x,f.residual.to_numpy()[take],fit_last_target_end=int(f.iloc[take].target_stop_exclusive.max()-1))
    return obj

def replay_bias(f,obj,tau,*,update=True):
    # Scalar local-level observer. State initialized only on legal mature labels.
    groups=rls.runs(f);o=f.origin.to_numpy();end=f.target_stop_exclusive.to_numpy()-1;latest=f.latest_available_target_index.to_numpy();r=f.residual.to_numpy();out=np.zeros(len(f));audit=np.full(len(f),-1,int);updates=0;resets=0
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group);state=float(obj['ym']);loc=0;last=int(obj['fit_last_target_end']);resets+=1
        if last>latest[ix[0]]:raise ValueError('Bias initializer label not yet mature')
        for t in ix:
            while loc<len(ix) and ix[loc]<t and end[ix[loc]]<=latest[t]:
                j=ix[loc]
                if update:
                    state+=(r[j]-state)/tau;last=max(last,int(end[j]));updates+=1
                loc+=1
            if not np.isfinite(state) or abs(state)>1e6:raise rls.NumericalFailure('Invalid bias observer state')
            assert last<=latest[t];out[t]=state;audit[t]=last
    return out,dict(latest_parameter_label=audit,latest_feature_label=np.full(len(f),-1,int),updates=updates,resets=resets,stored_issued_labels=True)

def predict_spec(f,obj,spec,*,update=True):
    if spec['family']=='BIAS':p,log=replay_bias(f,obj,spec['tau'],update=update)
    else:
        x,native,fa=dynamic_features(f,spec['p'],spec['family']) if spec['family'] in RESPONSE_FAMILIES else rls.features(f,spec['p']);p,a=rls.replay(f,x,native,obj,spec['forget'],update=update);log=dict(latest_feature_label=fa['latest_feature_target_end'],latest_parameter_label=a['latest_parameter_target_end'],updates=a['total_updates'],resets=a['resets'],stored_issued_features=True,current_issued_response=spec['family'] in RESPONSE_FAMILIES)
    return spec['gain']*p,log

def fit(f):
    specs=SPECS_RLS+SPECS_BIAS+SPECS_Q+SPECS_DYNAMIC+SPECS_CONDITIONAL;outer=[];selections=[];parent=[];child=[]
    def select(g):
        designs=all_designs(g);ledger=[];errors={i:[] for i in range(len(specs))};proof={i:[] for i in range(len(specs))};failed={}
        for lo,hi in [(.6,.8),(.8,1.)]:
            ei=split(g,lo,hi);fi=earlier(g,ei);ev=g.iloc[ei];score=common(ev)
            if len(score)<32:raise SupportLimited('Fewer than32 common mature inner scores')
            cache={}
            for i,s in enumerate(specs):
                if i in failed:continue
                key=tuple((k,v) for k,v in s.items() if k!='gain')
                try:
                    if key not in cache:
                        obj=initialize(g,fi,s,designs);unit=dict(s,gain=1.);p,log=predict_spec(ev,obj,unit)
                        bound=int(ev.origin.min())-L-PMAX-DELAY
                        assert obj['fit_last_target_end']<bound
                        assert (log['latest_parameter_label']<=ev.latest_available_target_index.to_numpy()).all()
                        cache[key]=(obj,p,dict(initializer_last_label=obj['fit_last_target_end'],first_score_recursive_dependency=bound,fit_rows=obj['fit_rows'],parameters=len(obj['theta']),score_rows=len(score)))
                    obj,p,pr=cache[key];errors[i].extend((ev.residual.to_numpy()-s['gain']*p)[score]**2);proof[i].append(pr)
                except (rls.SupportLimited,SupportLimited) as e:failed[i]=str(e)
        for i,s in enumerate(specs):ledger.append(dict(spec=s,status='SUPPORT_LIMITED',reason=failed[i]) if i in failed else dict(spec=s,status='AVAILABLE',MSE=float(np.mean(errors[i])),proof=proof[i]))
        legal=[q for q in ledger if q['status']=='AVAILABLE'];assert legal
        best=min(legal,key=lambda q:(q['MSE'],q['spec']['gain'],q['spec']['family']!='BIAS',q['spec'].get('p',0),-q['spec'].get('tau',0)))
        selections.append(dict(fit_rows=len(g),selected=best,ledger=ledger));return best['spec']
    designs=all_designs(f)
    for lo,hi in [(.6,.75),(.75,.9),(.9,1.)]:
        ei=split(f,lo,hi);fi=earlier(f,ei)
        try:
            s=select(f.iloc[fi]);obj=initialize(f,fi,s,designs);ev=f.iloc[ei];ix=common(ev);p,log=predict_spec(ev,obj,s);parent.extend(ev.residual.to_numpy()[ix]**2);child.extend((ev.residual.to_numpy()-p)[ix]**2)
            bound=int(ev.origin.min())-L-PMAX-DELAY;assert obj['fit_last_target_end']<bound
            outer.append(dict(status='COMPLETED',spec=s,initializer_last_label=obj['fit_last_target_end'],first_score_recursive_dependency=bound,score_rows=len(ix),parent_MSE=float(np.mean(ev.residual.to_numpy()[ix]**2)),child_MSE=float(np.mean((ev.residual.to_numpy()-p)[ix]**2))))
        except (rls.SupportLimited,SupportLimited) as e:outer.append(dict(status='SUPPORT_LIMITED',reason=str(e)))
    s=select(f);obj=initialize(f,np.arange(len(f)),s,designs);pr=float(np.mean(parent)) if parent else None;ch=float(np.mean(child)) if child else None
    admitted=sum(q['status']=='COMPLETED' for q in outer)==3 and pr-ch>16*np.finfo(float).eps*(pr+ch)
    return dict(selected=s,model=obj,admitted=bool(admitted),parent_risk=pr,child_risk=ch,outer=outer,selections=selections,family='MATURE_ISSUED_Q_BIAS_RLS')

def predict(f,obj,*,update=True):
    p,log=predict_spec(f,obj['model'],obj['selected'],update=update)
    return p if obj['admitted'] else np.zeros(len(f)),log

# Fixed operator candidates: no online parameter updates, only lawful mature state.
SPECS_RLS += [dict(family='TRANSPORT',p=p,window=0,forget=1.,gain=g) for p in [1,4] for g in [.25,.5,.75,1.]]
_original_predict_spec=predict_spec
def predict_spec(f,obj,spec,*,update=True):
    if spec['family']!='TRANSPORT':return _original_predict_spec(f,obj,spec,update=update)
    x,native,audit=rls.features(f,spec['p']);p=np.zeros(len(f));p[native]=x[native].mean(1)
    return spec['gain']*p,dict(latest_feature_label=audit['latest_feature_target_end'],latest_parameter_label=np.full(len(f),obj['fit_last_target_end'],int),updates=0,resets=len(np.unique(rls.runs(f))),stored_issued_features=True,current_issued_response=False)
