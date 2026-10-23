import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,pickle,time,sys,traceback
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import model as m
import data_view as d
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else str(v)))
def data(case):return {case:np.load(ROOT/'TRAINING_DATA.npz')[case]}
def binding():
    r=read(ROOT/'REGISTRATION_BEFORE_FITS.json');assert all(sha(ROOT/n)==v for n,v in r['code_sha256'].items()) and sha(ROOT/'TRAINING_DATA.npz')==r['training_sha256'];return r

def prepare():
    assert not (ROOT/'REGISTRATION_BEFORE_FITS.json').exists() and '12 passed' in (ROOT/'CORE_TESTS_ATTEMPT_1.log').read_text()
    view=read(ROOT/'DATA_VIEW_BEFORE_FITS.json')
    reg=dict(schema='SRU_DYNAMIC_ISSUED_RESPONSE_A_V8',cases=list(view['records']),variants=['full','no_KC','no_W','no_A','report_A'],training_sha256=sha(ROOT/'TRAINING_DATA.npz'),view_sha256=sha(ROOT/'DATA_VIEW_BEFORE_FITS.json'),
        task='Nowcasting H2S y(t) using measured process x(t), past x(t-1..t-7), mature y(t-delay..t-delay-7), delay4/8/16 steps. No current target or other output channel. User explicitly accepts concurrent industrial soft sensing and A/model improvement.',
        availability='Label maturity4 or16 observation steps after issue, explicitly experimental, not asserted real analyzer timing. This is an experimental availability assumption, not reconstructed actual assay delivery latency. Every comparator has identical current process X and past target information. Current target can only enter later residuals/parameter updates after maturity.',
        source='Adaptive followup after V6/V7 development. Training-only lag diagnosis found residual dependence on past16/32 response changes. Same repeatedly reused SRU H2S development, no new physical source or independent confirmation. All earlier negative and point-positive results retained.',
        model='Explicit structural redesign, not unchanged v0.6: raw current/past input coordinates; KC own8/linear7-or5/finite Urysohn amplitude response with2 degree1 spline basis coefficients per input, either current or current+lag1-7 bin. Optional2 past own states. Ridge C. Joint scalar inverse W identity/signedsqrt/asinh/signedlog or q-only residual spline learned on2 forward parameter-CF streams. W/C scales fold-local; raw inputs never transformed with output.',
        deletion='Independent upstream retrain and downstream retuning full/noKC/noW/noA; noKC keeps own8 response and all W/A choices, removes process-response K/C. Full also includes this bypass. noW identity. noA independently retunes upstream by mature parent outer risk.',
        joint_selection='Same V2 identity/nonidentity recipe per OWN/LINEAR/URYSOHN and joint candidate A outer-risk selection. 15 final configurations,12 selected main A policies plus3 selected ADAPTIVE_ARX A policies. At most57 candidate A search calls (39 new-family,18 report controls). Exact fits counted in review. Each A stream uses one frozen KCW parent.',selection='KC/W two outer purged forward folds70-85/85-100%. Learned W two subfolds60-80/80-100%. All choices training-only. 200rows/20per coefficient retained. Candidates unsupported/nonfinite retained as unavailable, no clipping.',
        A='Retain225 V6 specs, add32 dynamic issued-response specs. QD8 features q,deltaq1/4/8,q*deltaq8; QD32 q,deltaq1/4/8/16/32,q*deltaq16/32; optionally one mature residual. Initialization last256/all,forget.995/.999,gain.5/1. No current y; all derivative features use only issued q from same stream, never cross generator/split/gaps. Original ridge100/stored-phi mature RLS and200/20/numerical gates unchanged. Pmax32 conservative raw support and fixed32+delay-1 startup common across variants/control. Three outer60/75/90 and twoinner60/80, each residual stream one frozen KCW. Current response transformation and calibration explicitly structural redesign, not unchanged report A. ADAPTIVE_ARX gets identical257-spec A; frozen-update control retained.',
        splits={k:dict(upper=v['upper'],delay=v['delay'],calibration=v['upper']+8+2*v['delay'],train=v['train'],last_available_training_label=v['train']-v['delay'],development=v['dev']) for k,v in view['records'].items()},
        baselines='Existing12 baselines including ADAPTIVE_ARX with same257-spec A. Add ARX163,warpedARX163,onlineARX163 using128 mature own-target lags+process lags0/1/4/7/16/32/39; add long-memory HGB and RBF-SVR using128 own-target+40 dense process lags. Same current X and label maturity, full original training budget; more process history covers q32 plus original7-step source span. Total17 baselines, all tuned only on training.',
        metrics='Recorded units RMSE. Paired circular block2500 length192 covers the expanded finite raw/target span; full and fixed32+delay-1 startup sensitivity. Frozen-update and baseline comparisons retained. Adaptive descriptive intervals, not independent/search-adjusted.',
        final='No final reader or final parsing in this branch. Existing ledger references hash-bound and never reset. Any qualified setting requires separate freeze/audit before one-shot eligible final; historically exposed sources never relabeled fresh.',
        code_sha256={n:sha(ROOT/n) for n in ['model.py','mature_a.py','legacy.py','rls_core.py','run.py','data_view.py','test_model.py','launch.py']},source_final_numeric_read=False)
    dump(ROOT/'REGISTRATION_BEFORE_FITS.json',reg);print('SRU_MATURE_BIAS_REGISTERED15_CONFIGS12_MAIN_A_AND3_BASELINE_A' ,flush=True)

def online_fit(D,T):
    x=m.design(D,T,dict(kind='ARX'))[0];f=m.source(T,T.anchor.to_numpy(),'online_train');ledger=[]
    for alpha in [1.,100.,1000.]:
        for forget in [.99,.995,.999,1.]:
            err=[];proof=[]
            try:
                for fi,ei in m.folds(T):
                    obj=m.rls.initialize(x[fi],f.residual.to_numpy()[fi],fit_last_target_end=int(T.iloc[fi].target_end.max()),alpha=alpha)
                    p,log=m.rls.replay(f.iloc[ei],x[ei],np.ones(len(ei),bool),obj,forget);err.extend((f.residual.to_numpy()[ei]-p)**2)
                    assert obj['fit_last_target_end']<T.iloc[ei].origin.min()-m.SUPPORT and (log['latest_parameter_target_end']<=T.iloc[ei].origin.to_numpy()-m.DELAY).all()
                    proof.append(dict(last_fit_label=obj['fit_last_target_end'],first_score_dependency=int(T.iloc[ei].origin.min())-m.SUPPORT))
                ledger.append(dict(alpha=alpha,forget=forget,MSE=float(np.mean(err)),status='AVAILABLE',proof=proof))
            except m.rls.NumericalFailure as e:ledger.append(dict(alpha=alpha,forget=forget,status='NUMERICALLY_REJECTED',reason=str(e)))
    best=min([r for r in ledger if r['status']=='AVAILABLE'],key=lambda x:(x['MSE'],-x['forget'],-x['alpha']));obj=m.rls.initialize(x,f.residual.to_numpy(),fit_last_target_end=int(T.target_end.max()),alpha=best['alpha']);return dict(model=obj,selected=best,ledger=ledger)

def tree_fit(x,y,leaves,L2):return HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=L2,early_stopping=False,random_state=20261023).fit(x,y)

def long_x(D,T,kind='ARX'):
    a=D[T.entity_id.iloc[0]];o=T.origin.to_numpy();own=a[o[:,None]-np.arange(m.DELAY,m.DELAY+128)[None,:],0]
    return own if kind=='OWN' else np.c_[own,a[o[:,None]-np.array([0,1,4,7])[None,:],1:].reshape(len(T),-1)]

def long_folds(T):return [(m.earlier(T,ei,128+m.DELAY-1),ei) for ei in [m.split(T,a,b) for a,b in [(.7,.85),(.85,1.)]]]

def process_long_x(D,T,kind='ARX',dense=False):
    a=D[T.entity_id.iloc[0]];o=T.origin.to_numpy();own=a[o[:,None]-np.arange(m.DELAY,m.DELAY+128)[None,:],0]
    if kind=='OWN':return own
    lags=np.arange(40) if dense else np.array([0,1,4,7,16,32,39]);return np.c_[own,a[o[:,None]-lags[None,:],1:].reshape(len(T),-1)]

def long_baseline_fit(D,T,kind,warped=False,feature_fn=None):
    x=(feature_fn or long_x)(D,T,kind);ledger=[]
    for target in ['level','delta']:
        for link in (m.LINKS if warped else ['identity']):
            for alpha in [.01,1.,100.,1000.]:
                err=[];proof=[]
                for fi,ei in long_folds(T):
                    fit=T.iloc[fi];ev=T.iloc[ei];w=m.old.link_fit(fit.y,link);base=m.old.forward(fit.anchor.to_numpy(),w) if target=='delta' else 0.
                    obj=m.old.ridge(x[fi],m.old.forward(fit.y.to_numpy(),w)-base,np.full(x.shape[1],alpha));q=m.old.rp(obj,x[ei])+(m.old.forward(ev.anchor.to_numpy(),w) if target=='delta' else 0.);p=m.old.inverse(q,w);err.extend((ev.y.to_numpy()-p)**2)
                    assert fit.target_end.max()<ev.origin.min()-(128+m.DELAY-1);proof.append(dict(last_fit_label=int(fit.target_end.max()),first_score_dependency=int(ev.origin.min())-(128+m.DELAY-1),certificate=dict(rows=obj['fit_rows'],parameters=obj['parameter_count'],**obj['certificate'])))
                ledger.append(dict(target=target,link=link,alpha=alpha,MSE=float(np.mean(err)),proof=proof))
    best=min(ledger,key=lambda v:(v['MSE'],v['link']!='identity',-v['alpha']));w=m.old.link_fit(T.y,best['link']);y=m.old.forward(T.y.to_numpy(),w)-(m.old.forward(T.anchor.to_numpy(),w) if best['target']=='delta' else 0.);obj=m.old.ridge(x,y,np.full(x.shape[1],best['alpha']));return dict(kind=kind,model=obj,selected=best,coordinate=w,ledger=ledger,long_process=feature_fn is not None)

def long_baseline_predict(D,T,obj):
    x=process_long_x(D,T,obj['kind']) if obj.get('long_process') else long_x(D,T,obj['kind']);q=m.old.rp(obj['model'],x)+(m.old.forward(T.anchor.to_numpy(),obj['coordinate']) if obj['selected']['target']=='delta' else 0.);return m.old.inverse(q,obj['coordinate'])

def long_online_fit(D,T,feature_fn=None):
    x=(feature_fn or long_x)(D,T);f=m.source(T,T.anchor.to_numpy(),'long_online_training');ledger=[]
    for forget in [.99,.999,1.]:
        err=[];proof=[]
        for fi,ei in long_folds(T):
            obj=m.rls.initialize(x[fi],f.residual.to_numpy()[fi],fit_last_target_end=int(T.iloc[fi].target_end.max()));p,log=m.rls.replay(f.iloc[ei],x[ei],np.ones(len(ei),bool),obj,forget);err.extend((f.residual.to_numpy()[ei]-p)**2)
            assert obj['fit_last_target_end']<T.iloc[ei].origin.min()-(128+m.DELAY-1) and (log['latest_parameter_target_end']<=T.iloc[ei].origin.to_numpy()-m.DELAY).all();proof.append(dict(last_fit_label=obj['fit_last_target_end'],first_score_dependency=int(T.iloc[ei].origin.min())-(128+m.DELAY-1)))
        ledger.append(dict(forget=forget,MSE=float(np.mean(err)),proof=proof))
    best=min(ledger,key=lambda v:(v['MSE'],-v['forget']));return dict(model=m.rls.initialize(x,f.residual.to_numpy(),fit_last_target_end=int(T.target_end.max())),selected=best,ledger=ledger,long_process=feature_fn is not None)

def expanded_baselines(D,T):
    out={};out['ARX163']=long_baseline_fit(D,T,'ARX',feature_fn=process_long_x);out['WARX163']=long_baseline_fit(D,T,'ARX',True,feature_fn=process_long_x);out['ONLINE_ARX163']=long_online_fit(D,T,feature_fn=process_long_x)
    x=process_long_x(D,T,dense=True);ledger=[]
    for target in ['level','delta']:
        y=T.y.to_numpy()-(T.anchor.to_numpy() if target=='delta' else 0.)
        for leaves in [7,15]:
            for L2 in [1.,10.]:
                errors=[]
                for fi,ei in long_folds(T):
                    obj=tree_fit(x[fi],y[fi],leaves,L2);errors.extend((y[ei]-obj.predict(x[ei]))**2)
                ledger.append(dict(target=target,leaves=leaves,L2=L2,MSE=float(np.mean(errors))))
    best=min(ledger,key=lambda v:v['MSE']);y=T.y.to_numpy()-(T.anchor.to_numpy() if best['target']=='delta' else 0.);out['HGB_LONG_MEMORY']=dict(model=tree_fit(x,y,best['leaves'],best['L2']),selected=best,ledger=ledger)
    ledger=[]
    for C in [1.,10.,100.]:
        for gamma in ['scale',.01]:
            errors=[]
            for fi,ei in long_folds(T):
                ym=float(T.y.to_numpy()[fi].mean());ys=max(float(T.y.to_numpy()[fi].std()),1e-8);obj=make_pipeline(StandardScaler(),SVR(C=C,gamma=gamma,epsilon=.01)).fit(x[fi],(T.y.to_numpy()[fi]-ym)/ys);p=ym+ys*obj.predict(x[ei]);errors.extend((T.y.to_numpy()[ei]-p)**2)
            ledger.append(dict(C=C,gamma=gamma,MSE=float(np.mean(errors))))
    best=min(ledger,key=lambda v:v['MSE']);ym=float(T.y.mean());ys=max(float(T.y.std(ddof=0)),1e-8);out['SVR_LONG_MEMORY']=dict(model=make_pipeline(StandardScaler(),SVR(C=best['C'],gamma=best['gamma'],epsilon=.01)).fit(x,(T.y.to_numpy()-ym)/ys),ym=ym,ys=ys,selected=best,ledger=ledger)
    return out

def baselines(D,T):
    out={k:m.fit_kcw(D,T,variant='no_W',family=k) for k in ['OWN','ARX']};out['WARX']=m.fit_kcw(D,T,family='ARX');x=m.design(D,T,dict(kind='DENSE'))[0];ledger=[]
    for target in ['level','delta']:
        y=T.y.to_numpy()-(T.anchor.to_numpy() if target=='delta' else 0.)
        for leaves in [7,15]:
            for L2 in [1.,10.]:
                err=[]
                for fi,ei in m.folds(T):
                    obj=tree_fit(x[fi],y[fi],leaves,L2);err.extend((y[ei]-obj.predict(x[ei]))**2)
                ledger.append(dict(target=target,leaves=leaves,L2=L2,MSE=float(np.mean(err))))
    best=min(ledger,key=lambda v:v['MSE']);y=T.y.to_numpy()-(T.anchor.to_numpy() if best['target']=='delta' else 0.);out['HGB']=dict(model=tree_fit(x,y,best['leaves'],best['L2']),selected=best,ledger=ledger)
    ledger=[]
    for C in [1.,10.,100.]:
        for gamma in ['scale',.01]:
            err=[]
            for fi,ei in m.folds(T):
                ym=float(T.y.to_numpy()[fi].mean());ys=max(float(T.y.to_numpy()[fi].std()),1e-8);obj=make_pipeline(StandardScaler(),SVR(C=C,gamma=gamma,epsilon=.01)).fit(x[fi],(T.y.to_numpy()[fi]-ym)/ys);p=ym+ys*obj.predict(x[ei]);err.extend((T.y.to_numpy()[ei]-p)**2)
            ledger.append(dict(C=C,gamma=gamma,MSE=float(np.mean(err))))
    best=min(ledger,key=lambda v:v['MSE']);ym=float(T.y.mean());ys=max(float(T.y.std(ddof=0)),1e-8);out['SVR']=dict(model=make_pipeline(StandardScaler(),SVR(C=best['C'],gamma=best['gamma'],epsilon=.01)).fit(x,(T.y.to_numpy()-ym)/ys),ym=ym,ys=ys,selected=best,ledger=ledger)
    out['ONLINE_ARX']=online_fit(D,T)
    longT=T[T.origin>=128+m.DELAY-1].reset_index(drop=True)
    out['OWN_AR128']=long_baseline_fit(D,longT,'OWN');out['ARX148']=long_baseline_fit(D,longT,'ARX');out['WARX148']=long_baseline_fit(D,longT,'ARX',True);out['ONLINE_ARX148']=long_online_fit(D,longT)
    upper=m.frame(D,m.SUPPORT,3328);cal=m.frame(D,3328+8+2*m.DELAY,int(T.target_end.max())+1);out['ADAPTIVE_ARX']=m.joint_train(D,upper,cal,'full',family='ARX')
    out.update(expanded_baselines(D,longT))
    return out

def train(case):
    r=binding();v=d.records()[case];m.configure(v['delay']);D=data(case);dest=ROOT/'cases'/case;dest.mkdir(parents=True,exist_ok=False);T=m.frame(D,m.SUPPORT,v['upper']);cal=m.frame(D,v['upper']+8+2*m.DELAY,v['train']-m.DELAY+1);files={}
    assert T.target_end.max()<cal.origin.min()-m.SUPPORT
    for variant in r['variants']:
        obj=m.joint_train(D,T,cal,variant);obj.update(case=case,variant=variant);path=dest/(variant+'_FROZEN.pkl');path.write_bytes(pickle.dumps(obj,protocol=5));files[path.name]=sha(path)
        dump(dest/(variant+'_TRAINING.json'),dict(KCW_selected=obj['upper']['selected'],KCW_ledger=obj['KCW_ledger'],A=obj['A'],joint_candidates=obj['joint_candidates'],joint_selected_index=obj['joint_selected_index']));print('SOFTSENSOR_DELAYED_FROZEN',case,variant,obj['upper']['selected']['config'],obj['upper']['selected']['mode'],None if obj['A'] is None else [obj['A']['admitted'],obj['A']['selected']['p']],flush=True)
    obj=baselines(D,m.frame(D,m.SUPPORT,v['train']-m.DELAY+1));path=dest/'BASELINES_FROZEN.pkl';path.write_bytes(pickle.dumps(obj,protocol=5));files[path.name]=sha(path);dump(dest/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json',dict(files=files,final_numeric_read=False));print('SOFTSENSOR_DELAYED_CASE_ALL_FROZEN',case,flush=True)

def effect(y,a,b,skip):
    ea=(y[skip:]-a[skip:])**2;eb=(y[skip:]-b[skip:])**2;n=len(ea);rng=np.random.default_rng(20261023);ratios=[];diff=[]
    for _ in range(2500):
        starts=rng.integers(0,n,size=(n+191)//192);ix=((starts[:,None]+np.arange(192))%n).ravel()[:n];m0=ea[ix].mean();m1=eb[ix].mean();assert m0>0;ratios.append(float(100*(np.sqrt(m1/m0)-1)));diff.append(float(m1-m0))
    return dict(skip=skip,gain_percent=float(100*(np.sqrt(eb.mean()/ea.mean())-1)),CI95=np.quantile(ratios,[.025,.975]).tolist(),MSE_difference_CI95=np.quantile(diff,[.025,.975]).tolist(),block=192)

def evaluate(case):
    reg=binding();v=d.records()[case];m.configure(v['delay']);dest=ROOT/'cases'/case;frozen=read(dest/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json');assert all(sha(dest/n)==h for n,h in frozen['files'].items());D=d.prefix(case,True)
    np.testing.assert_array_equal(D[case][:v['train']],data(case)[case]);V=m.frame(D,v['train'],v['dev']);pred=V.copy();prefix=V.head(80).copy();cut=int(prefix.origin.max());poison={case:D[case].copy()};poison[case][cut+1:]+=1000.;rows=[];QA=[];before=m.FITS
    for variant in reg['variants']:
        obj=pickle.loads((dest/(variant+'_FROZEN.pkl')).read_bytes());p=m.predict_kcw(D,V,obj['upper']);sf=m.source(V,p,'development_cold');a=np.zeros(len(V)) if obj['A'] is None else m.a_predict(sf,obj['A'])[0];pred[variant]=p+a;pred[variant+'_upper']=p
        if variant=='full':pred['full_A_frozen_updates']=p+m.mature_a.predict(sf,obj['A'],update=False)[0]
        up=m.predict_kcw(D,prefix,obj['upper']);np.testing.assert_array_equal(up,m.predict_kcw(poison,prefix,obj['upper']))
        # Separate current-target poison: current X stays available, y(t) must not enter.
        first=prefix.head(1);o=int(first.origin.iloc[0]);target_poison={case:D[case].copy()};target_poison[case][o:,0]+=1000.;np.testing.assert_array_equal(m.predict_kcw(D,first,obj['upper']),m.predict_kcw(target_poison,first,obj['upper']))
        if obj['A']:
            f=m.source(prefix,up,'development_cold');fp=f.copy();fp.loc[fp.target_stop_exclusive-1>cut-m.DELAY,'residual']+=1000.;aa,log=m.a_predict(f,obj['A']);ab,_=m.a_predict(fp,obj['A']);np.testing.assert_array_equal(aa,ab);np.testing.assert_allclose(up+aa,pred[variant].to_numpy()[:80],rtol=1e-12,atol=1e-12)
            assert (log['latest_feature_label']<=f.origin.to_numpy()-m.DELAY).all() and (log['latest_parameter_label']<=f.origin.to_numpy()-m.DELAY).all() and log['resets']==1
        rows.append(dict(variant=variant,RMSE=float(np.sqrt(m.mse(V.y,pred[variant]))),KCW_selected=obj['upper']['selected'],A_admitted=None if obj['A'] is None else obj['A']['admitted'],A_selected=None if obj['A'] is None else obj['A']['selected']));QA.append(dict(variant=variant,current_y_poison_bitwise=True,future_X_poison_bitwise=True,unmatured_feedback_poison_bitwise=True,only_mature_label=True))
    b=pickle.loads((dest/'BASELINES_FROZEN.pkl').read_bytes());pred['Persistence']=V.anchor
    for k in ['OWN','ARX','WARX']:pred[k]=m.predict_kcw(D,V,b[k])
    x=m.design(D,V,dict(kind='DENSE'))[0];pred['HGB']=b['HGB']['model'].predict(x)+(V.anchor.to_numpy() if b['HGB']['selected']['target']=='delta' else 0.);pred['SVR']=b['SVR']['ym']+b['SVR']['ys']*b['SVR']['model'].predict(x)
    x=m.design(D,V,dict(kind='ARX'))[0];sf=m.source(V,V.anchor.to_numpy(),'online_development');q,log=m.rls.replay(sf,x,np.ones(len(V),bool),b['ONLINE_ARX']['model'],b['ONLINE_ARX']['selected']['forget']);pred['ONLINE_ARX']=V.anchor.to_numpy()+q;assert (log['latest_parameter_target_end']<=V.origin.to_numpy()-m.DELAY).all()
    pf=sf.head(80).copy();pf.loc[pf.target_stop_exclusive-1>cut-m.DELAY,'residual']+=1000.;qq,_=m.rls.replay(pf,x[:80],np.ones(80,bool),b['ONLINE_ARX']['model'],b['ONLINE_ARX']['selected']['forget']);np.testing.assert_array_equal(qq,q[:80]);np.testing.assert_array_equal(x[:80],m.design(poison,prefix,dict(kind='ARX'))[0])
    for k in ['OWN_AR128','ARX148','WARX148']:pred[k]=long_baseline_predict(D,V,b[k])
    ad=b['ADAPTIVE_ARX'];ap=m.predict_kcw(D,V,ad['upper']);af=m.source(V,ap,'adaptive_ARX_dev');ac,al=m.a_predict(af,ad['A']);pred['ADAPTIVE_ARX']=ap+ac
    assert (al['latest_parameter_label']<=V.origin.to_numpy()-m.DELAY).all();pa=m.predict_kcw(poison,prefix,ad['upper']);np.testing.assert_array_equal(pa,ap[:80]);paf=m.source(prefix,pa,'adaptive_ARX_dev');paf.loc[paf.target_stop_exclusive-1>cut-m.DELAY,'residual']+=1000.;pac,_=m.a_predict(paf,ad['A']);np.testing.assert_array_equal(pac,ac[:80])
    lx=long_x(D,V);lsf=m.source(V,V.anchor.to_numpy(),'long_online_dev');lq,llog=m.rls.replay(lsf,lx,np.ones(len(V),bool),b['ONLINE_ARX148']['model'],b['ONLINE_ARX148']['selected']['forget']);pred['ONLINE_ARX148']=V.anchor.to_numpy()+lq
    assert (llog['latest_parameter_target_end']<=V.origin.to_numpy()-m.DELAY).all();np.testing.assert_array_equal(lx[:80],long_x(poison,prefix));lpf=lsf.head(80).copy();lpf.loc[lpf.target_stop_exclusive-1>cut-m.DELAY,'residual']+=1000.;lqq,_=m.rls.replay(lpf,lx[:80],np.ones(80,bool),b['ONLINE_ARX148']['model'],b['ONLINE_ARX148']['selected']['forget']);np.testing.assert_array_equal(lqq,lq[:80])
    for k in ['ARX163','WARX163']:pred[k]=long_baseline_predict(D,V,b[k])
    xl=process_long_x(D,V,dense=True);np.testing.assert_array_equal(xl[:80],process_long_x(poison,prefix,dense=True));pred['HGB_LONG_MEMORY']=b['HGB_LONG_MEMORY']['model'].predict(xl)+(V.anchor.to_numpy() if b['HGB_LONG_MEMORY']['selected']['target']=='delta' else 0.);pred['SVR_LONG_MEMORY']=b['SVR_LONG_MEMORY']['ym']+b['SVR_LONG_MEMORY']['ys']*b['SVR_LONG_MEMORY']['model'].predict(xl)
    xs=process_long_x(D,V);fs=m.source(V,V.anchor.to_numpy(),'online163_dev');ps,ls=m.rls.replay(fs,xs,np.ones(len(V),bool),b['ONLINE_ARX163']['model'],b['ONLINE_ARX163']['selected']['forget']);pred['ONLINE_ARX163']=V.anchor.to_numpy()+ps;assert (ls['latest_parameter_target_end']<=V.origin.to_numpy()-m.DELAY).all();np.testing.assert_array_equal(xs[:80],process_long_x(poison,prefix));fps=fs.head(80).copy();fps.loc[fps.target_stop_exclusive-1>cut-m.DELAY,'residual']+=1000.;pps,_=m.rls.replay(fps,xs[:80],np.ones(80,bool),b['ONLINE_ARX163']['model'],b['ONLINE_ARX163']['selected']['forget']);np.testing.assert_array_equal(pps,ps[:80])
    baseline={k:float(np.sqrt(m.mse(V.y,pred[k]))) for k in ['Persistence','OWN','ARX','WARX','HGB','SVR','ONLINE_ARX','OWN_AR128','ARX148','WARX148','ONLINE_ARX148','ADAPTIVE_ARX','ARX163','WARX163','ONLINE_ARX163','HGB_LONG_MEMORY','SVR_LONG_MEMORY']};effects=[dict(control=k,**effect(V.y.to_numpy(),pred.full.to_numpy(),pred[k].to_numpy(),skip)) for k in ['no_KC','no_W','no_A'] for skip in [0,m.PMAX+m.DELAY-1]]
    frozen_update_control=[effect(V.y.to_numpy(),pred.full.to_numpy(),pred.full_A_frozen_updates.to_numpy(),s) for s in [0,m.PMAX+m.DELAY-1]]
    positive=all(e['gain_percent']>0 and e['CI95'][0]>0 for e in effects);below=rows[0]['RMSE']<min(baseline.values());assert m.FITS==before and all(sha(dest/n)==h for n,h in frozen['files'].items())
    pred.to_parquet(dest/'DEVELOPMENT_predictions.parquet',index=False);dump(dest/'DEVELOPMENT_RESULT.json',dict(case=case,status='COMPLETED',task='CURRENT_X_NOWCASTING',rows=rows,baselines=baseline,effects=effects,positive_all3_CI=positive,below_all_baselines=below,goal_achieved=bool(positive and below),frozen_update_control=frozen_update_control,best_baseline_comparison=effect(V.y.to_numpy(),pred.full.to_numpy(),pred[min(baseline,key=baseline.get)].to_numpy(),0),adaptive_ARX_selected=dict(KCW=ad['upper']['selected'],A=ad['A']['selected']),QA=QA,no_offline_refit=True,noA_independent_joint_upstream_retuning=True,online_baseline_current_X_and_feedback_QA=True,models_unchanged=True,source_final_numeric_read=False,label_delay=m.DELAY,historical_exposure=v['historical_exposure'],rows_scored=len(V)));print('SRU_MATURE_BIAS_DEVELOPMENT',case,rows[0]['RMSE'],min(baseline.values()),positive,positive and below,flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='train':train(sys.argv[2])
        elif sys.argv[1]=='evaluate':evaluate(sys.argv[2])
    except BaseException:dump(ROOT/('FAILURE_'+'_'.join(sys.argv[1:])+'.json'),dict(error=traceback.format_exc(),time=time.time()));raise
