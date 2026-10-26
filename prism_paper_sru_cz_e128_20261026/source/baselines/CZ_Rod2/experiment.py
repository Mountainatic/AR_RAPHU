"""Separated train / evaluate SRU benchmark with training-only purged selection."""
from pathlib import Path
import os,sys,json,time,hashlib,copy,pickle,platform,importlib.metadata
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='3'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['AR_RAPHU_RUNTIME_MANAGER']='uv'
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from baseline_models import build
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(3);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def reg():
    r=read(ROOT/'REGISTRATION.json')
    assert all(sha(ROOT/k)==v for k,v in r['code_sha256'].items())
    assert sha(ROOT/'data/train.npz')==r['train_sha256']
    return r
def origins(delay):return np.arange(192,12376-delay+1,dtype=np.int64)
def folds(delay):
    o=origins(delay);out=[]
    for lo,hi in [(.7,.85),(.85,1.)]:
        ev=o[int(len(o)*lo):int(len(o)*hi)]
        fit=o[o<ev.min()-(128+delay)]
        assert fit.max()<ev.min()-(128+delay)
        out.append((fit,ev))
    return out
def sequence(a,o,L,delay):
    o=np.asarray(o,dtype=np.int64);xidx=o[:,None]-np.arange(L-1,-1,-1)[None,:]
    yidx=xidx-delay
    assert yidx.min()>=0 and (yidx.max(1)<=o-delay).all() and (xidx.max(1)==o).all()
    return np.concatenate([a[xidx,1:],a[yidx,0,None]],axis=2).astype(np.float32)
def flat(a,o,c,delay,kind):
    own=a[o[:,None]-delay-np.arange(c['Ly'])[None,:],0]
    if kind=='AR':return own
    x=a[o[:,None]-np.array(c['Xlags'])[None,:],1:].reshape(len(o),-1)
    return np.c_[own,x]
def candidates():
    return [dict(L=L,d=d,lr=lr,target=target) for L,d,lr in [(128,32,.001),(128,64,.0003)] for target in ['level','delta']]
def metric(y,p):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);assert y.shape==p.shape and np.isfinite(p).all()
    e=y-p;den=np.abs(y);nz=den>1e-8
    return dict(n=len(y),R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),RMSE=float(np.sqrt(np.mean(e*e))),
        MAE=float(np.mean(np.abs(e))),MAPE_pct=float(100*np.mean(np.abs(e[nz])/den[nz])) if nz.any() else None,
        MAPE_excluded_nearzero=int((~nz).sum()),sMAPE_pct=float(200*np.mean(np.abs(e)/np.maximum(np.abs(y)+np.abs(p),1e-8))),
        WAPE_pct=float(100*np.sum(np.abs(e))/np.sum(den)),bias=float(np.mean(p-y)))
def make_batch(a,o,c,delay,sc=None):
    x=sequence(a,o,c['L'],delay);anchor=a[o-delay,0]
    target=a[o,0]-(anchor if c['target']=='delta' else 0)
    if sc is None:
        sc=dict(xmean=x.mean((0,1),dtype=np.float64).astype(np.float32),
            xstd=np.maximum(x.std((0,1),dtype=np.float64),1e-6).astype(np.float32),
            ymean=float(target.mean()),ystd=max(float(target.std()),1e-6))
    x=(x-sc['xmean'])/sc['xstd'];y=(target-sc['ymean'])/sc['ystd']
    return torch.as_tensor(x,device='cuda'),torch.as_tensor(y.astype(np.float32),device='cuda'),anchor,sc
def infer(model,x,sc,anchor,c,batch=512):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):
            block=x[i:i+batch];n=len(block)
            if n<batch:block=torch.cat([block,torch.zeros((batch-n,*block.shape[1:]),device=block.device,dtype=block.dtype)],dim=0)
            parts.append(model(block)[:n])
    p=torch.cat(parts).cpu().numpy().astype(np.float64)
    return p*sc['ystd']+sc['ymean']+(anchor if c['target']=='delta' else 0.)

def fit_deep(name,delay,c,seed,fit,ev,a,epochs=None):
    torch.manual_seed(seed);np.random.seed(seed)
    model=build(name,6,c['d'],c['L']).cuda()
    x,y,_,sc=make_batch(a,fit,c,delay)
    if ev is not None:xe,ye,anchor,_=make_batch(a,ev,c,delay,sc)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=1e-4)
    best=float('inf');best_state=None;best_epoch=0;trace=[];last=0
    for epoch in range(1,(epochs or 80)+1):
        model.train();order=torch.randperm(len(x),device='cuda');total=0.
        for i in range(0,len(x),256):
            ix=order[i:i+256];opt.zero_grad(set_to_none=True)
            pred=model(x[ix]);loss=torch.mean((pred-y[ix])**2)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();total+=float(loss.detach())*len(ix)
        row=dict(epoch=epoch,train_MSE_standardized=total/len(x))
        if ev is not None:
            p=infer(model,xe,sc,anchor,c);risk=float(np.mean((a[ev,0]-p)**2));row['validation_MSE']=risk
            if risk<best:
                best=risk;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};last=epoch
        trace.append(row)
        if ev is not None and epoch-last>=12:break
    if ev is None:
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epochs
    obj=dict(state=best_state,scaler=sc,config=c,seed=seed,epochs=best_epoch,parameters=sum(p.numel() for p in model.parameters()),
        latest_fit_target=int(fit.max()),delay=delay,name=name,precision='FP32',trace=trace)
    if ev is not None:obj['validation_MSE']=best
    del model,x,y,opt
    torch.cuda.empty_cache()
    return obj
def train_deep(name,delay):
    r=reg();dest=ROOT/'runs'/f'D{delay}'/name;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'FROZEN.json').exists()
    a=np.load(ROOT/'data/train.npz')['values'];assert a.shape==(12376,6)
    start=time.time();torch.cuda.reset_peak_memory_stats();ledger=[]
    for ci,c in enumerate(candidates()):
        rows=[]
        for fi,(fit,ev) in enumerate(folds(delay)):
            obj=fit_deep(name,delay,c,20261023,fit,ev,a)
            row=dict(fold=fi,MSE=obj['validation_MSE'],epochs=obj['epochs'],fit_rows=len(fit),score_rows=len(ev),
                latest_fit_target=int(fit.max()),first_validation_origin=int(ev.min()),purge_steps=128+delay,
                parameter_count=obj['parameters'],trace=obj['trace'])
            rows.append(row)
        item=dict(candidate=ci,config=c,MSE=float(np.average([x['MSE'] for x in rows],weights=[x['score_rows'] for x in rows])),folds=rows)
        ledger.append(item);dump(dest/'SELECTION_PROGRESS.json',ledger)
        print('CV',name,delay,ci,item['MSE'],[x['epochs'] for x in rows],flush=True)
    best=min(ledger,key=lambda v:(v['MSE'],v['candidate']));c=best['config']
    epochs=max(1,int(round(np.median([x['epochs'] for x in best['folds']]))));files={}
    for seed in [0,1,2]:
        obj=fit_deep(name,delay,c,seed,origins(delay),None,a,epochs)
        p=dest/f'seed_{seed}.pt';torch.save(obj,p);files[p.name]=sha(p)
    dump(dest/'FROZEN.json',dict(status='COMPLETED',name=name,delay=delay,selected=best,epochs=epochs,
        ledger=ledger,files=files,latest_fit_target=12376-delay,development_read_during_fit=False,
        runtime_seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),workers_per_device=2,cpu_threads=3))
    print('FROZEN',name,delay,'seconds',round(time.time()-start),flush=True)
def train_linear(delay):
    reg();a=np.load(ROOT/'data/train.npz')['values'];dest=ROOT/'runs'/f'D{delay}'/'linear';dest.mkdir(parents=True,exist_ok=True)
    result={};ledger={};start=time.time()
    for name in ['AR','ARX']:
        configs=[dict(Ly=L,Xlags=[]) for L in [128]] if name=='AR' else [dict(Ly=128,Xlags=list(range(128)))]
        scores=[]
        for c in configs:
            for alpha in [.01,1.,100.,1000.,10000.]:
                for target in ['level','delta']:
                    err=[];proof=[]
                    for fit,ev in folds(delay):
                        x=flat(a,fit,c,delay,name);y=a[fit,0]-(a[fit-delay,0] if target=='delta' else 0)
                        assert len(fit)>=1000 and x.shape[1] in [128,768]  # Regularized dense ARX, not PRISM's staged-support rule.
                        ym=float(y.mean());ys=max(float(y.std()),1e-8)
                        obj=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky')).fit(x,(y-ym)/ys)
                        p=ym+ys*obj.predict(flat(a,ev,c,delay,name))+(a[ev-delay,0] if target=='delta' else 0)
                        err.extend((a[ev,0]-p)**2);proof.append(dict(last_fit_target=int(fit.max()),first_score=int(ev.min()),rows=len(fit),parameters=x.shape[1]+1))
                    scores.append(dict(config=c,alpha=alpha,target=target,MSE=float(np.mean(err)),proof=proof))
        best=min(scores,key=lambda v:v['MSE']);o=origins(delay);y=a[o,0]-(a[o-delay,0] if best['target']=='delta' else 0)
        ym=float(y.mean());ys=max(float(y.std()),1e-8)
        obj=make_pipeline(StandardScaler(),Ridge(alpha=best['alpha'],solver='cholesky')).fit(flat(a,o,best['config'],delay,name),(y-ym)/ys)
        xx=obj[0].transform(flat(a,o,best['config'],delay,name));gram=xx.T@xx;alpha=best['alpha']
        eig=np.maximum(np.linalg.eigvalsh(gram),0.);df=float(1+np.sum(eig/(eig+alpha)))
        rhs=xx.T@((y-ym)/ys-obj[1].intercept_);g=gram+alpha*np.eye(gram.shape[0]);coef=obj[1].coef_
        kkt=float(np.linalg.norm(g@coef-rhs)/max(1.,np.linalg.norm(rhs),np.linalg.norm(g)*np.linalg.norm(coef)));assert kkt<1e-8
        result[name]=dict(model=obj,ym=ym,ys=ys,selected=best,regularization_certificate=dict(rows=len(o),raw_regressors=xx.shape[1],effective_degrees_of_freedom=df,relative_kkt=kkt));ledger[name]=scores
    p=dest/'linear.pkl';p.write_bytes(pickle.dumps(result,protocol=5))
    dump(dest/'FROZEN.json',dict(status='COMPLETED',files={p.name:sha(p)},selected={k:v['selected'] for k,v in result.items()},ledger=ledger,
        latest_fit_target=12376-delay,development_read_during_fit=False,runtime_seconds=time.time()-start))
    print('LINEAR_FROZEN',delay,flush=True)
def prepare():
    raise RuntimeError('Use root prepare.py; scientific registration must precede all fits')

def smoke():
    reg();a=np.load(ROOT/'data/train.npz')['values'];checks=[]
    for delay in [4,8,16]:
        for L in [128]:
            o=np.array([192,193,399]);x=sequence(a,o,L,delay)
            for t in o:
                poisoned=a.copy();poisoned[t-delay+1:,0]+=1234;poisoned[t+1:,1:]-=1234
                np.testing.assert_array_equal(sequence(a,[t],L,delay),sequence(poisoned,[t],L,delay))
        for fit,ev in folds(delay):assert fit.max()<ev.min()-128-delay
    for name in ['LSTM','Transformer','iTransformer','TimeXer']:
        model=build(name,6,32,128).cuda();x=torch.tensor(sequence(a,[192,193],128,4),device='cuda');model.train()
        p=model(x);assert p.shape==(2,) and torch.isfinite(p).all();p.sum().backward()
        assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
        checks.append(name)
    dump(ROOT/'SMOKE_TESTS.json',dict(status='COMPLETED',models_forward_backward=checks,unmatured_y_and_future_x_poison=True,
        folds_purged=True,source_final_numeric_read=False));print('SMOKE_PASSED',checks,flush=True)
def evaluate():
    r=reg();gate=read(ROOT/'ALL_FITS_FROZEN.json');assert gate['all_fits_frozen_before_evaluation']
    a=np.vstack([np.load(ROOT/'data/train.npz')['values'],np.load(ROOT/'data/development.npz')['values']]);assert a.shape==(16501,6)
    out=ROOT/'results';out.mkdir(exist_ok=True);rows=[];seedrows=[];long=[];audits=[]
    for delay in [4,8,16]:
        o=np.arange(12376,16501);y=a[o,0];ref=pd.read_parquet(ROOT/'data'/f'prism_D{delay}.parquet')
        np.testing.assert_array_equal(o,ref.origin);np.testing.assert_array_equal(y,ref.y_true)
        wide=pd.DataFrame(dict(origin=o,label_delay=delay,latest_available_y_index=o-delay,y_true=y))
        wide['Persistence']=a[o-delay,0]
        for name in ['PRISM','PRISM_no_A','PRISM_fixed128','PRISM_fixed128_no_A']:wide[name]=ref[name].to_numpy()
        dest=ROOT/'runs'/f'D{delay}'/'linear';receipt=read(dest/'FROZEN.json');assert all(sha(dest/k)==v for k,v in receipt['files'].items())
        lin=pickle.loads((dest/'linear.pkl').read_bytes())
        for name,v in lin.items():
            b=v['selected'];wide[name]=v['ym']+v['ys']*v['model'].predict(flat(a,o,b['config'],delay,name))+(a[o-delay,0] if b['target']=='delta' else 0.)
        for name in ['LSTM','Transformer','iTransformer','TimeXer']:
            dest=ROOT/'runs'/f'D{delay}'/name;receipt=read(dest/'FROZEN.json');assert all(sha(dest/k)==v for k,v in receipt['files'].items())
            ps=[]
            for seed in [0,1,2]:
                obj=torch.load(dest/f'seed_{seed}.pt',map_location='cpu',weights_only=False);c=obj['config'];model=build(name,6,c['d'],c['L']).cuda();model.load_state_dict(obj['state'])
                x,_,anchor,sc=make_batch(a,o,c,delay,obj['scaler']);p=infer(model,x,sc,anchor,c);ps.append(p)
                seedrows.append(dict(label_delay=delay,model=name,seed=seed,**metric(y,p)))
                pd.DataFrame(dict(origin=o,label_delay=delay,model=name,seed=seed,y_true=y,y_pred=p)).to_parquet(out/f'D{delay}_{name}_seed{seed}.parquet',index=False)
                for n in [1,delay,delay+1,65,257]:
                    t=int(o[n-1]);poison=a.copy();poison[t-delay+1:,0]+=1234;poison[t+1:,1:]-=1234
                    np.testing.assert_array_equal(sequence(poison,o[:n],c['L'],delay),sequence(a,o[:n],c['L'],delay))
                    xp,_,ap,_=make_batch(poison,o[:n],c,delay,sc)
                    pp=infer(model,xp,sc,ap,c)
                    np.testing.assert_allclose(pp,p[:n],rtol=3e-5,atol=2e-6)
                audits.append(dict(model=name,delay=delay,seed=seed,unmatured_y_and_future_X_poison_passed=True,
                    latest_fitted_label=obj['latest_fit_target'],first_evaluation_origin=int(o.min()),batch_prefix_consistency=True))
            wide[name]=np.mean(ps,axis=0)
        for name in [c for c in wide.columns if c not in ['origin','label_delay','latest_available_y_index','y_true']]:
            p=wide[name].to_numpy();rows.append(dict(label_delay=delay,model=name,prediction='3_seed_ensemble' if name in ['LSTM','Transformer','iTransformer','TimeXer'] else 'deterministic',**metric(y,p)))
            long.append(pd.DataFrame(dict(origin=o,label_delay=delay,model=name,y_true=y,y_pred=p,residual=y-p,absolute_error=np.abs(y-p),squared_error=(y-p)**2,
                absolute_percentage_error_pct=np.where(np.abs(y)>1e-8,100*np.abs(y-p)/np.maximum(np.abs(y),1e-8),np.nan))))
        wide.to_csv(out/f'point_predictions_D{delay}.csv',index=False);wide.to_parquet(out/f'point_predictions_D{delay}.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(out/'metrics_by_seed.csv',index=False)
    pd.concat(long,ignore_index=True).to_parquet(out/'point_predictions_long.parquet',index=False)
    pd.concat(long,ignore_index=True).to_csv(out/'point_predictions_long.csv',index=False)
    dump(out/'CAUSAL_AUDIT.json',dict(status='COMPLETED',checks=audits,source_final_numeric_read=False))
    dump(out/'COMPLETED.json',dict(status='COMPLETED',rows_metrics=len(rows),direct_input_length=128,rows_per_model=4125,delays=[4,8,16],
        source_final_numeric_read=False,all_new_fits_frozen_before_evaluation=True,registration_sha256=sha(ROOT/'REGISTRATION.json')))
    print(pd.DataFrame(rows)[['label_delay','model','R2','RMSE','MAE','MAPE_pct']].to_string(index=False),flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='smoke':smoke()
    elif mode=='linear':train_linear(int(sys.argv[2]))
    elif mode=='train':train_deep(sys.argv[2],int(sys.argv[3]))
    elif mode=='evaluate':evaluate()
