"""Train-only PRISM window selection; no development scoring until all choices freeze."""
from pathlib import Path
import os,sys,json,hashlib,pickle,time,importlib.metadata
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prism'))
import model as m
WINDOWS=[8,16,32,64,128];DELAYS=[4,8,16];MAX_L=128
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def configure(delay,window):
    assert delay in DELAYS and window in WINDOWS
    m.L=window;m.configure(delay)
    # Identical conservative dependency embargo for every candidate window.
    m.SUPPORT=MAX_L+delay-1;m.old.L=m.SUPPORT;m.mature_a.configure(delay,m.SUPPORT)
def reg():
    r=read(ROOT/'REGISTRATION.json')
    assert all(sha(ROOT/k)==v for k,v in r['code_sha256'].items())
    assert sha(ROOT/'data/train.npz')==r['train_sha256']
    return r
def data(delay,development=False):
    a=np.load(ROOT/'data/train.npz')['values']
    if development:a=np.vstack([a,np.load(ROOT/'data/development.npz')['values']])
    return {f'SRU_H2S_D{delay}':a}
def geometry(D,delay):
    T=m.frame(D,192,3328);cal=m.frame(D,3328+MAX_L+2*delay,6048-delay+1)
    score=np.concatenate([m.split(cal,a,b)[m.PMAX+delay-1:] for a,b in [(.6,.75),(.75,.9),(.9,1.)]])
    assert T.target_end.max()<cal.origin.min()-m.SUPPORT
    return T,cal,cal.origin.to_numpy()[score]
def prepare():
    assert not (ROOT/'REGISTRATION.json').exists()
    code={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}
    baseline={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'baseline_reference').rglob('*')) if p.is_file()}
    dump(ROOT/'REGISTRATION.json',dict(status='COMPLETED',task='PRISM train-selected windows vs fixed128 baselines',
        source_branch='sru-conditional-feedback-a-20261023',source_commit='a4cdf17d31ae8956dffe20f8745d45802bd084f0',
        candidate_windows=WINDOWS,label_delays=DELAYS,selection='Minimum joint training-calibration nested MSE per delay and variant; ties choose shorter window',
        common_geometry=dict(parent_targets=[192,3328],calibration_start='3328+128+2*delay',calibration_stop_exclusive='6048-delay+1',
            common_dependency_embargo='127+delay for all windows',score='calibration blocks .6-.75/.75-.9/.9-1, each drops32+delay-1 cold rows'),
        variants=['full','no_A'],unchanged_A_candidates=273,numerical_and_support_gates='Unchanged original PRISM',
        direct_X='x(t-L+1)..x(t)',direct_y='y(t-delay-L+1)..y(t-delay)',
        state_compression='Original structure: two y endpoints0/L-1, current X and optional mean past X response; candidate may omit own or memory',
        online_A='Full online A retained; issued q and matured residual/RLS history may exceed chosen direct window',
        development_targets=[6048,8064],reserved_tail='[8064,10081) NEVER READ',development_role='Previously reused development comparison, not independent final test',
        baseline_direct_length=128,baseline_action='Reuse exact frozen predictions and seed metrics; no baseline retraining',
        baseline_archive_sha256='ca4d91e9fbcee61311ab6719ef82675ecf18efc367876aaca6f22198252f2fda',
        train_sha256=sha(ROOT/'data/train.npz'),development_file_sha256=sha(ROOT/'data/development.npz'),
        code_sha256=code,baseline_files_sha256=baseline,parallel_workers=3,threads_per_worker=1,
        environment=dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','pyarrow']})))
    print('REGISTERED',flush=True)
def smoke():
    reg();records=[]
    for delay in DELAYS:
        D=data(delay);origins=None
        for window in WINDOWS:
            configure(delay,window);T,cal,score=geometry(D,delay)
            if origins is None:origins=score
            np.testing.assert_array_equal(score,origins)
            V=m.frame(D,192,196);own,u=m.raw(D,V);assert own.shape==(4,window) and u.shape==(4,window,5)
            config=dict(kind='LINEAR',alpha=1.,own=True,memory=True);x,_=m.design(D,V,config)
            o=V.origin.to_numpy();case=V.entity_id.iloc[0]
            np.testing.assert_array_equal(x[:,:2],D[case][o[:,None]-delay-np.array([0,window-1]),0])
            cut=int(o[-1]);poison={case:D[case].copy()};poison[case][cut-delay+1:,0]+=1234;poison[case][cut+1:,1:]-=1234
            np.testing.assert_array_equal(m.design(poison,V,config)[0],x)
            obj=m.fit_parent(D,T,config,'identity');assert np.isfinite(m.predict_parent(D,V,obj)).all()
            records.append(dict(delay=delay,window=window,score_count=len(score),score_origins_sha256=hashlib.sha256(score.tobytes()).hexdigest(),certificate=obj['certificate']))
    dump(ROOT/'SMOKE.json',dict(status='COMPLETED',records=records,no_development_data_read=True))
    print('SMOKE_PASSED',flush=True)
def train(delay,window):
    reg();configure(delay,window);D=data(delay);T,cal,score=geometry(D,delay)
    dest=ROOT/'runs'/f'D{delay}'/f'L{window}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'FROZEN.json').exists();start=time.time();files={};selected={};counts={}
    for variant in ['full','no_A']:
        before=m.FITS;obj=m.joint_train(D,T,cal,variant)
        path=dest/f'{variant}.pkl';path.write_bytes(pickle.dumps(obj,protocol=5));files[path.name]=sha(path)
        chosen=obj['joint_candidates'][obj['joint_selected_index']]
        selected[variant]=dict(selection_MSE=float(chosen['selection_MSE']),KCW=obj['upper']['selected'],
            A=None if obj['A'] is None else dict(selected=obj['A']['selected'],admitted=obj['A']['admitted'],parent_risk=obj['A']['parent_risk'],child_risk=obj['A']['child_risk']))
        counts[variant]=m.FITS-before
        ledger=dest/f'{variant}_TRAINING.json'
        dump(ledger,dict(KCW_ledger=obj['KCW_ledger'],joint_candidates=obj['joint_candidates'],joint_selected_index=obj['joint_selected_index']))
        files[ledger.name]=sha(ledger)
        print('CANDIDATE_FROZEN',delay,window,variant,selected[variant]['selection_MSE'],flush=True)
    dump(dest/'FROZEN.json',dict(status='COMPLETED',delay=delay,window=window,files=files,selected=selected,
        score_origins=score.tolist(),common_support=m.SUPPORT,latest_fit_target=6048-delay,development_read_during_fit=False,
        staged_ridge_fit_counts=counts,runtime_seconds=time.time()-start))
def select():
    reg();receipts={};choices={};rows=[]
    for delay in DELAYS:
        records=[];origins=None
        for window in WINDOWS:
            path=ROOT/'runs'/f'D{delay}'/f'L{window}'/'FROZEN.json';f=read(path)
            assert f['status']=='COMPLETED' and not f['development_read_during_fit']
            assert all(sha(path.parent/k)==v for k,v in f['files'].items())
            if origins is None:origins=f['score_origins']
            assert origins==f['score_origins'] and f['common_support']==MAX_L+delay-1
            receipts[path.relative_to(ROOT).as_posix()]=sha(path);records.append(f)
        choices[str(delay)]={}
        for variant in ['full','no_A']:
            best=min(records,key=lambda f:(f['selected'][variant]['selection_MSE'],f['window']))
            choices[str(delay)][variant]=dict(window=best['window'],selection_MSE=best['selected'][variant]['selection_MSE'])
            for f in records:rows.append(dict(label_delay=delay,variant=variant,window=f['window'],
                selection_MSE=f['selected'][variant]['selection_MSE'],selected=f['window']==best['window'],
                score_count=len(origins),score_origin_sha256=hashlib.sha256(np.array(origins,dtype=np.int64).tobytes()).hexdigest(),
                KCW_config=json.dumps(f['selected'][variant]['KCW']['config'],sort_keys=True),W=f['selected'][variant]['KCW']['mode'],
                A_admitted=False if f['selected'][variant]['A'] is None else f['selected'][variant]['A']['admitted']))
    pd.DataFrame(rows).to_csv(ROOT/'WINDOW_CANDIDATES.csv',index=False)
    dump(ROOT/'ALL_SELECTIONS_FROZEN.json',dict(status='COMPLETED',all_fits_frozen_before_evaluation=True,
        choices=choices,receipts=receipts,ledger_sha256=sha(ROOT/'WINDOW_CANDIDATES.csv'),
        development_values_read_for_window_selection=False,selection_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('ALL_WINDOWS_SELECTED',json.dumps(choices),flush=True)
def evaluate(delay):
    r=reg();gate=read(ROOT/'ALL_SELECTIONS_FROZEN.json');assert gate['all_fits_frozen_before_evaluation']
    assert all(sha(ROOT/k)==v for k,v in gate['receipts'].items())
    assert sha(ROOT/'data/development.npz')==r['development_file_sha256']
    D=data(delay,True);case=f'SRU_H2S_D{delay}';V=m.frame(D,6048,8064);before=m.FITS
    result=pd.DataFrame(dict(origin=V.origin,y_true=V.y,label_delay=delay));audits=[]
    for variant,col in [('full','PRISM'),('no_A','PRISM_no_A')]:
        window=gate['choices'][str(delay)][variant]['window'];configure(delay,window)
        dest=ROOT/'runs'/f'D{delay}'/f'L{window}';frozen=read(dest/'FROZEN.json')
        assert all(sha(dest/k)==v for k,v in frozen['files'].items())
        obj=pickle.loads((dest/f'{variant}.pkl').read_bytes());q=m.predict_kcw(D,V,obj['upper']);f=m.source(V,q,'selected_window_evaluation')
        a=np.zeros(len(V)) if obj['A'] is None else m.a_predict(f,obj['A'])[0];p=q+a;result[col]=p
        for n in [1,delay,delay+1,33,65,257,2016]:
            cut=6048+n-1;poison={case:D[case].copy()};poison[case][cut-delay+1:,0]+=1234;poison[case][cut+1:,1:]-=1234
            W=m.frame(poison,6048,cut+1);qq=m.predict_kcw(poison,W,obj['upper']);fp=m.source(W,qq,'selected_window_evaluation')
            aa=np.zeros(n) if obj['A'] is None else m.a_predict(fp,obj['A'])[0]
            np.testing.assert_allclose(qq+aa,p[:n],rtol=1e-12,atol=1e-12)
        audits.append(dict(variant=variant,window=window,poison_prefixes=[1,delay,delay+1,33,65,257,2016],tolerance=1e-12,passed=True))
    assert before==m.FITS
    result.to_parquet(ROOT/'data'/f'prism_D{delay}.parquet',index=False)
    dump(ROOT/'audits'/f'D{delay}_CAUSAL_REPLAY.json',dict(status='COMPLETED',records=audits,no_fits=True,source_final_numeric_read=False))
    print('PRISM_EVALUATED',delay,flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='smoke':smoke()
    elif mode=='train':train(int(sys.argv[2]),int(sys.argv[3]))
    elif mode=='select':select()
    elif mode=='evaluate':evaluate(int(sys.argv[2]))
