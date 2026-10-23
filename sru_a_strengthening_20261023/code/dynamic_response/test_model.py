import numpy as np,pandas as pd
import model as m

m.configure(4)

def sample(n=1600):
    a=np.arange(n*4,dtype=float).reshape(n,4)/100.;return {'e':a},m.frame({'e':a},m.SUPPORT,n)

def test_current_input_and_past_target_offsets():
    D,V=sample();V=V.head(3);own,u=m.raw(D,V);o=V.origin.to_numpy();np.testing.assert_array_equal(own[:,0],D['e'][o-m.DELAY,0]);np.testing.assert_array_equal(u[:,0],D['e'][o,1:]);np.testing.assert_array_equal(u[:,-1],D['e'][o-7,1:])

def test_current_target_never_enters_features():
    D,T=sample();V=T.iloc[300:301];Q={'e':D['e'].copy()};Q['e'][int(V.origin.iloc[0]):,0]+=999.
    for kind in ['OWN','DENSE','ARX','LINEAR','URYSOHN']:
        c=dict(kind=kind,own=True,memory=True);_,r=m.design(D,T.head(200),c);x,_=m.design(D,V,c,r);q,_=m.design(Q,V,c,r);np.testing.assert_array_equal(x,q)

def test_raw_support_purge():
    D,V=sample()
    for fi,ei in m.folds(V):assert V.iloc[fi].target_end.max()<V.iloc[ei].origin.min()-m.SUPPORT

def test_short_A_geometry_and_exact_report_core():
    rng=np.random.default_rng(9)
    for delay in [4,16]:
        m.configure(delay);D,V=sample(6048);cal=m.frame(D,3328+8+2*delay,6048-delay+1);f=m.source(cal,cal.y.to_numpy()-rng.normal(size=len(cal)),'cal');a=m.a_fit(f,report=True)
        assert len(a['outer'])==3 and all(r['status']=='COMPLETED' for r in a['outer'])
        assert a['selected']['gain']==1. and a['selected']['forget'] in [.99,.995,.999]
        for sel in a['selections']:
            for c in sel['candidates']:
                if c['status']=='AVAILABLE':assert all(p['fit_rows']>=max(200,20*(c['p']+1)) and p['initializer_last_label']<p['first_score_recursive_dependency'] for p in c['proof'])
    m.configure(4)

def test_current_residual_not_available_before_issue():
    rng=np.random.default_rng(3);D,V=sample(700);f=m.source(V.head(400),rng.normal(size=400),'test');g=f.copy();g.loc[g.index[-m.DELAY]:,'residual']+=1000.
    for p in [1,2,4,8]:
        x,n,a=m.rls.features(f,p);xx,nn,aa=m.rls.features(g,p);np.testing.assert_array_equal(x,xx);assert (a['latest_feature_target_end']<=f.origin.to_numpy()-m.DELAY).all()

def test_bias_and_shrunk_RLS_only_consume_mature_labels():
    m.configure(8);rng=np.random.default_rng(32);D,T=sample(1200);train=m.frame(D,50,350);f=m.source(train,train.y.to_numpy()-rng.normal(size=len(train)),'train');V=m.frame(D,500,900);s=m.source(V,V.y.to_numpy()-rng.normal(size=len(V)),'eval');q=s.copy();q.loc[q.index[-m.DELAY]:,'residual']+=1000.
    x,n,_=m.rls.features(f,2)
    for family in ['RLS','BIAS']:
        spec=dict(family=family,p=2 if family=='RLS' else 0,window=256,forget=.995,tau=128,gain=.5)
        obj=m.rls.initialize(x[n],f.residual.to_numpy()[n],fit_last_target_end=int(f.target_stop_exclusive.max()-1)) if family=='RLS' else m.rls.initialize(np.zeros((len(f),1)),f.residual.to_numpy(),fit_last_target_end=int(f.target_stop_exclusive.max()-1))
        a,log=m.mature_a.predict_spec(s,obj,spec);b,_=m.mature_a.predict_spec(q,obj,spec);np.testing.assert_array_equal(a,b);assert (log['latest_parameter_label']<=s.latest_available_target_index).all()
    m.configure(4)

def test_bias_observer_entity_reset_and_frozen_update_control():
    m.configure(4);rng=np.random.default_rng(82);D,T=sample(1000);V=m.frame(D,500,700);f=m.source(V,V.y.to_numpy()-rng.normal(size=len(V)),'eval');g=f.copy();g['entity_id']='other';both=pd.concat([f,g],ignore_index=True)
    obj=m.rls.initialize(np.zeros((256,1)),rng.normal(size=256),fit_last_target_end=300);spec=dict(family='BIAS',p=0,window=256,tau=32,gain=.5)
    a,log=m.mature_a.predict_spec(both,obj,spec);np.testing.assert_array_equal(a[:len(f)],a[len(f):]);assert log['resets']==2
    frozen,fl=m.mature_a.predict_spec(both,obj,spec,update=False);np.testing.assert_array_equal(frozen,np.full(len(both),.5*obj['ym']));assert fl['updates']==0

def test_bias_initializer_does_not_relax_support_gate():
    import pytest
    rng=np.random.default_rng(3);D,T=sample(800);V=m.frame(D,200,500);f=m.source(V,V.y.to_numpy()-rng.normal(size=len(V)),'cal');designs={p:m.rls.features(f,p) for p in [1,2,4,8]}
    with pytest.raises(m.rls.SupportLimited):m.mature_a.initialize(f,np.arange(199),dict(family='BIAS',p=0,window=256),designs)

def test_current_issued_response_feature_and_stored_phi_updates():
    m.configure(8);rng=np.random.default_rng(74);D,T=sample(1300);train=m.frame(D,100,400);f=m.source(train,rng.normal(size=len(train)),'train');V=m.frame(D,600,900);s=m.source(V,rng.normal(size=len(V)),'eval')
    for p in [0,1,4]:
        designs=m.mature_a.all_designs(f);spec=dict(family='Q_RLS',p=p,window=256,forget=.999,gain=.5);obj=m.mature_a.initialize(f,np.arange(len(f)),spec,designs);x,n,a=m.mature_a.issued_features(s,p);np.testing.assert_array_equal(x[:,0],s.issued_q)
        pred,log=m.mature_a.predict_spec(s,obj,spec);q=s.copy();q.loc[q.index[-m.DELAY]:,'residual']+=1000.;other,_=m.mature_a.predict_spec(q,obj,spec);np.testing.assert_array_equal(pred,other)
        _,audit=m.rls.replay(s,x,n,obj,.999,return_states=True)
        for ev in audit['events'][:20]:np.testing.assert_array_equal(ev['phi'],np.r_[1.,(x[ev['u']]-np.array(obj['mean']))/np.array(obj['scale'])])
        assert (log['latest_parameter_label']<=s.latest_available_target_index).all()
    m.configure(4)

def test_dynamic_response_features_formula_and_reset():
    m.configure(8);D,T=sample(1200);V=m.frame(D,600,800);q=np.sin(np.arange(len(V))/11.);s=m.source(V,q,'eval')
    for family,lags in [('QD8_RLS',[1,4,8]),('QD32_RLS',[1,4,8,16,32])]:
        x,n,a=m.mature_a.dynamic_features(s,0,family);assert not n[:max(lags)].any() and n[max(lags):].all()
        t=60;delta=q[t]-q[t-np.array(lags)];cross=q[t]*delta[[-1] if family=='QD8_RLS' else [-2,-1]];np.testing.assert_array_equal(x[t],np.r_[q[t],delta,cross])
        second=s.copy();second['entity_id']='other';both=pd.concat([s,second],ignore_index=True);xx,nn,_=m.mature_a.dynamic_features(both,0,family);np.testing.assert_array_equal(xx[:len(s)],xx[len(s):]);assert not nn[len(s):len(s)+max(lags)].any()
    m.configure(4)

def test_dynamic_response_maturity_and_stored_phi():
    m.configure(8);rng=np.random.default_rng(123);D,T=sample(1500);train=m.frame(D,100,500);f=m.source(train,rng.normal(size=len(train)),'train');V=m.frame(D,700,1050);s=m.source(V,rng.normal(size=len(V)),'eval')
    for family in ['QD8_RLS','QD32_RLS']:
        spec=dict(family=family,p=1,window=256,forget=.999,gain=1.);obj=m.mature_a.initialize(f,np.arange(len(f)),spec,m.mature_a.all_designs(f));a,log=m.mature_a.predict_spec(s,obj,spec);q=s.copy();q.loc[q.index[-m.DELAY]:,'residual']+=1000.;b,_=m.mature_a.predict_spec(q,obj,spec);np.testing.assert_array_equal(a,b)
        x,n,_=m.mature_a.dynamic_features(s,1,family);_,audit=m.rls.replay(s,x,n,obj,.999,return_states=True)
        for ev in audit['events'][:20]:np.testing.assert_array_equal(ev['phi'],np.r_[1.,(x[ev['u']]-np.array(obj['mean']))/np.array(obj['scale'])])
        future=s.copy();future.loc[future.index[100]:,'issued_q']+=1000.;xp,_,_=m.mature_a.dynamic_features(future,1,family);np.testing.assert_array_equal(x[:100],xp[:100]);assert (log['latest_parameter_label']<=s.latest_available_target_index).all()
    m.configure(4)

def test_expanded_baseline_current_X_past_Y_cutoff():
    import run
    m.configure(8);D,T=sample(1000);V=m.frame(D,400,401);o=int(V.origin.iloc[0]);Q={'e':D['e'].copy()};Q['e'][o:,0]+=1000.
    for dense in [False,True]:
        x=run.process_long_x(D,V,dense=dense);np.testing.assert_array_equal(x,run.process_long_x(Q,V,dense=dense));assert x[0,0]==D['e'][o-8,0] and x[0,127]==D['e'][o-8-127,0]
        np.testing.assert_array_equal(x[0,128:131],D['e'][o,1:])
    m.configure(4)
