"""Registered fixed-window ablations and bounded-innovation RLS extension.

Training and development evaluation are separate commands. All new fits must
freeze before any new development predictions are generated. Existing archives
are read-only. No source raw workbook or reserved-tail reader is present.
"""
from pathlib import Path
import os, sys, json, hashlib, pickle, time, subprocess
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['AR_RAPHU_RUNTIME_MANAGER']='uv'
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=Path('/root/autodl-tmp')
SCI=BASE/'PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020/.venv/bin/python'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else str(v)),encoding='utf-8')
    tmp.replace(p)

def geometry(entity):
    if entity=='SRU_H2S':
        return dict(root=str(BASE/'PRISM_SRU_WINDOW_SELECTION_20261024'),parent_end=3328,train_end=6048,dev_end=8064,windows={4:8,8:16,16:16})
    r=BASE/'cz_sru_matched_20261024'/entity;g=read(r/'GEOMETRY.json')
    table=pd.read_csv(BASE/'cz_sru_matched_20261024/results/selected_windows.csv')
    w={int(x.label_delay):int(x.window) for x in table.itertuples() if x.rod==entity and x.variant=='full'}
    return dict(root=str(r),parent_end=g['parent_end'],train_end=g['train_end'],dev_end=g['development_end'],windows=w)

def reference(g,entity,d,variant='full'):
    return Path(g['root'])/'runs'/f'D{d}'/f"L{g['windows'][d]}"/f'{variant}.pkl'

def runtime(name):
    sys.path.insert(0,str(ROOT/name))
    import model as m
    return m

def configure(m,g,d):
    m.L=getattr(m,'E_HISTORY',g['windows'][d]);g['windows'][d]=m.L;m.configure(d)
    m.SUPPORT=127+d;m.old.L=m.SUPPORT;m.mature_a.configure(d,m.SUPPORT)

def train_data(g,entity,d):
    a=np.load(Path(g['root'])/'data/train.npz')['values']
    assert len(a)==g['train_end']
    return {f'{entity}_D{d}':a}

def frames(m,D,g,d):
    T=m.frame(D,192,g['parent_end'])
    cal=m.frame(D,g['parent_end']+128+2*d,g['train_end']-d+1)
    assert T.target_end.max()<cal.origin.min()-m.SUPPORT
    return T,cal

def prepare():
    assert not (ROOT/'REGISTRATION.json').exists()
    cases={e:geometry(e) for e in ('CZ_Rod1','CZ_Rod2')}
    for g in cases.values():g['windows']={4:128,8:128,16:128}
    for g in cases.values():g['windows']={str(k):v for k,v in g['windows'].items()}
    jobs=[]
    for name in ('fixed128','E128','E16'):
        for e in cases:
            for d in (4,8,16):
                jobs.extend(dict(runtime=name,entity=e,delay=d,variant=v) for v in ('full','no_A'))
    sources={}
    for e,g0 in cases.items():
        g=dict(g0,windows={int(k):v for k,v in g0['windows'].items()})
        for d in (4,8,16):
            for v in ('full','no_A'):
                p=reference(g,e,d,v);sources[str(p)]=sha(p)
        p=Path(g['root'])/'data/train.npz';sources[str(p)]=sha(p)
    code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')}
    dump(ROOT/'REGISTRATION.json',dict(status='FROZEN_BEFORE_NEW_FITS',cases=cases,jobs=jobs,source_hashes=sources,code_hashes=code,
        task='Concurrent y(t) soft sensing; same current X and strictly matured y as original. Label delays are experimental availability assumptions.',
        ablations='CZ main128 imposed by user; E16 is an explicit ablation, common support127+delay and original training geometry. no_KC/no_W independently refit and reselect KCW/A; all36 full/no_A chains are newly fitted. This is conditional fixed-window ablation, not reselecting the entire window-search procedure.',
        robust_extension='CZ fixed128 main input history, corrected degree1 spline. E128 uses raw block averages on lag bins[0,1),[1,2),[2,4),[4,8),[8,16),[16,32),[32,64),[64,128), then per-channel amplitude basis and learned ridge coefficients. This is FIXED_MULTIRESOLUTION E, not full original channel-specific profile/rank search. E16 is a short-history refit control only, never selected as the CZ main model. Candidate process families require memory=True; OWN bypass and exact-zero A remain. Original273 A policies and gates retained. No Huber extension.',
        selection='Original training-only nested A MSE and three outer admission folds, joint KCW/A risk. No development-driven model selection, task/window changes, or repeated grid edits.',
        evaluation='Generate all new development predictions only after every36 new job freezes. Previously reused development, exploratory, not independent test.',
        resources=dict(workers=3,threads=1,GPU_training=False),reserved_tail_read=False,raw_workbook_read=False))

def binding():
    r=read(ROOT/'REGISTRATION.json')
    assert all(sha(ROOT/k)==v for k,v in r['code_hashes'].items())
    assert all(sha(k)==v for k,v in r['source_hashes'].items())
    return r

def train(job):
    r=binding();g=r['cases'][job['entity']];g['windows']={int(k):v for k,v in g['windows'].items()}
    d=job['delay'];m=runtime(job['runtime']);configure(m,g,d)
    D=train_data(g,job['entity'],d);assert sha(Path(g['root'])/'data/train.npz')==r['source_hashes'][str(Path(g['root'])/'data/train.npz')]
    T,cal=frames(m,D,g,d)
    dest=ROOT/'runs'/job['runtime']/job['entity']/f'D{d}'/job['variant'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'FROZEN.json').exists():
        assert sha(dest/'model.pkl')==read(dest/'FROZEN.json')['model_sha256'];return
    start=time.monotonic();before=m.FITS
    obj=m.joint_train(D,T,cal,job['variant']);p=dest/'model.pkl';p.write_bytes(pickle.dumps(obj,protocol=5))
    chosen=obj['joint_candidates'][obj['joint_selected_index']]
    dump(dest/'TRAINING_LEDGER.json',dict(KCW_ledger=obj['KCW_ledger'],joint_candidates=obj['joint_candidates'],joint_selected_index=obj['joint_selected_index']))
    dump(dest/'FROZEN.json',dict(status='COMPLETED',job=job,model_sha256=sha(p),ledger_sha256=sha(dest/'TRAINING_LEDGER.json'),
        seconds=time.monotonic()-start,ridge_fits=m.FITS-before,selection_MSE=chosen['selection_MSE'],KCW=obj['upper']['selected'],
        A=None if obj['A'] is None else dict(selected=obj['A']['selected'],admitted=obj['A']['admitted']),development_read=False))
    print('FROZEN',job,chosen['selection_MSE'],flush=True)

def queue():
    r=binding();procs=[];todo=list(r['jobs']);done=[]
    while todo or procs:
        while todo and len(procs)<3:
            j=todo.pop(0);logpath=ROOT/'logs'/f"{j['runtime']}_{j['entity']}_{j['delay']}_{j['variant']}.log";logpath.parent.mkdir(exist_ok=True)
            f=logpath.open('a');p=subprocess.Popen([str(SCI),'-u',str(ROOT/'suite.py'),'train',json.dumps(j)],stdout=f,stderr=subprocess.STDOUT)
            procs.append((j,p,f,time.monotonic()))
        keep=[]
        for j,p,f,start in procs:
            rc=p.poll()
            if rc is None:keep.append((j,p,f,start));continue
            f.close();done.append(dict(job=j,returncode=rc,seconds=time.monotonic()-start))
        procs=keep;dump(ROOT/'QUEUE_PROGRESS.json',done);time.sleep(1)
    assert all(x['returncode']==0 for x in done),done
    receipts={}
    for j in r['jobs']:
        p=ROOT/'runs'/j['runtime']/j['entity']/f"D{j['delay']}"/j['variant']/'FROZEN.json'
        f=read(p);assert f['status']=='COMPLETED' and not f['development_read'];receipts[str(p.relative_to(ROOT))]=sha(p)
    dump(ROOT/'ALL_NEW_FITS_FROZEN.json',dict(status='COMPLETED',receipts=receipts,development_read=False,job_count=len(done)))
    for mode in ('fixed128','E128','E16'):
        for entity in ('CZ_Rod1','CZ_Rod2'):
            if mode=='standard' and entity!='SRU_H2S':continue
            subprocess.run([str(SCI),'-u',str(ROOT/'suite.py'),'evaluate',mode,entity],check=True)
    dump(ROOT/'QUEUE_COMPLETED.json',dict(status='COMPLETED',new_fits=len(done),reserved_tail_read=False))

def effect(y,full,other,skip=0):
    a=(y[skip:]-full[skip:])**2;b=(y[skip:]-other[skip:])**2;n=len(a);rng=np.random.default_rng(20261026);gain=[]
    for _ in range(2500):
        starts=rng.integers(0,n,size=(n+191)//192);ix=((starts[:,None]+np.arange(192))%n).ravel()[:n]
        gain.append(100*(np.sqrt(b[ix].mean()/a[ix].mean())-1))
    return dict(skip=skip,increase_percent=float(100*(np.sqrt(b.mean()/a.mean())-1)),CI95=np.quantile(gain,[.025,.975]).tolist(),block=192,bootstrap=2500,role='adaptive-development descriptive interval, not search adjusted')

def evaluate(mode,entity):
    r=binding();gate=read(ROOT/'ALL_NEW_FITS_FROZEN.json');assert all(sha(ROOT/k)==v for k,v in gate['receipts'].items())
    g=r['cases'][entity];g['windows']={int(k):v for k,v in g['windows'].items()};m=runtime(mode)
    metrics=[];audits=[];effects=[]
    a=np.vstack([np.load(Path(g['root'])/'data/train.npz')['values'],np.load(Path(g['root'])/'data/development.npz')['values']])
    assert len(a)==g['dev_end']
    for d in (4,8,16):
        configure(m,g,d);D={f'{entity}_D{d}':a};V=m.frame(D,g['train_end'],g['dev_end']);before=m.FITS
        table=pd.DataFrame(dict(origin=V.origin,y_true=V.y))
        variants=('full','no_KC','no_W','no_A') if entity=='SRU_H2S' else ('full','no_A')
        for v in variants:
            p=ROOT/'runs'/mode/entity/f'D{d}'/v/'model.pkl'
            if not p.exists():p=reference(g,entity,d,v)
            obj=pickle.loads(p.read_bytes());q=m.predict_kcw(D,V,obj['upper']);f=m.source(V,q,'optimization_eval')
            corr=np.zeros(len(V)) if obj['A'] is None else m.a_predict(f,obj['A'])[0];pred=q+corr;table[v]=pred
            for n in sorted(set([1,d,d+1,33,65,257,len(V)])):
                cut=g['train_end']+n-1;poison=a.copy();poison[cut-d+1:,0]+=1234;poison[cut+1:,1:]-=1234
                DD={f'{entity}_D{d}':poison};VV=m.frame(DD,g['train_end'],cut+1);qq=m.predict_kcw(DD,VV,obj['upper']);ff=m.source(VV,qq,'optimization_eval')
                aa=np.zeros(n) if obj['A'] is None else m.a_predict(ff,obj['A'])[0]
                np.testing.assert_allclose(qq+aa,pred[:n],rtol=1e-12,atol=1e-12)
            if mode=='standard' and v=='full':
                old=pd.read_parquet(Path(g['root'])/'data'/f'prism_D{d}.parquet')['PRISM'].to_numpy()
                np.testing.assert_allclose(pred,old,rtol=1e-12,atol=1e-12)
            e=V.y.to_numpy()-pred
            metrics.append(dict(runtime=mode,entity=entity,delay=d,window=g['windows'][d],variant=v,RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(np.abs(e))),n=len(V),model_sha256=sha(p),source_model=str(p),A=None if obj['A'] is None else dict(selected=obj['A']['selected'],admitted=obj['A']['admitted'])))
            audits.append(dict(entity=entity,delay=d,variant=v,prefix_and_unmatured_label_poison_passed=True))
        assert before==m.FITS
        if entity=='SRU_H2S':
            for v in ('no_KC','no_W','no_A'):
                for skip in (0,32+d-1):effects.append(dict(entity=entity,delay=d,control=v,**effect(table.y_true.to_numpy(),table.full.to_numpy(),table[v].to_numpy(),skip)))
        dest=ROOT/'results'/mode/entity;dest.mkdir(parents=True,exist_ok=True);table.to_parquet(dest/f'D{d}_predictions.parquet',index=False)
    dump(ROOT/'results'/mode/entity/'RESULTS.json',dict(status='COMPLETED',metrics=metrics,effects=effects,audits=audits,fit_calls_during_evaluation=0,reserved_tail_read=False))
    pd.DataFrame([{k:v for k,v in x.items() if k!='A'} for x in metrics]).to_csv(ROOT/'results'/mode/entity/'metrics.csv',index=False)
    print('EVALUATED',mode,entity,flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='train':train(json.loads(sys.argv[2]))
    elif mode=='queue':queue()
    elif mode=='evaluate':evaluate(sys.argv[2],sys.argv[3])
