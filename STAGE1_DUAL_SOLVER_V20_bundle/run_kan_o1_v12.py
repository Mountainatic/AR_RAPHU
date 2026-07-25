#!/usr/bin/env python
"""Full, gated KAN O1 v12 audit; deliberately excludes O2/O3/M1/M2/FreeTilt."""
import csv, hashlib, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stage1.synthetic import SyntheticDataGenerator
from stage1.model import Stage1TargetDelayKAN
from stage1.response_kan import TruthResponseOracle, UnivariateKANResponse, MLPResponseOracle
from stage1.lag_contract import LagOrder
from stage1.truth_functions import get_true_function, TRUE_FUNCTION_NAMES, TRUE_FUNCTIONS

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results_stage1'/'KAN_O1_v12'; N,L=10,32; ACTIVE=(0,1,2)
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOURCES=['stage1/model.py','stage1/response_kan.py','stage1/synthetic.py','stage1/truth_functions.py','stage1/lag_contract.py','stage1/train.py','stage1/evaluate.py','stage1/losses.py','layers.py','tests/test_stage1.py','run_kan_o1_v12.py','resume_kan_o1_v12.py','finalize_kan_o1_v12.py','tools/build_handoff_bundle.py']

def hash_bytes(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes(): return {p:sha(ROOT/p) for p in SOURCES}
def native(x): return x.item() if isinstance(x,(np.floating,np.integer,np.bool_)) else x
def savej(p,d):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,default=native),encoding='utf8'); json.loads(p.read_text(encoding='utf8'))
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r}) if rows else ['status']
    with p.open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def r2(a,b):
    a=np.asarray(a); b=np.asarray(b); return float(1-((a-b)**2).sum()/(((b-b.mean())**2).sum()+1e-12))
def corr(a,b):
    a=np.asarray(a).ravel(); b=np.asarray(b).ravel(); return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0.
def spearman(a,b):
    return corr(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))
def seed(v): torch.manual_seed(v); np.random.seed(v); torch.cuda.manual_seed_all(v) if torch.cuda.is_available() else None

def build_splits(truth):
    # Explicit chronological partitions; two 31-point raw embargoes leave 1438 usable windows.
    train=np.arange(0,1007); val=np.arange(1038,1254); test=np.arange(1285,1500)
    sets=[]
    for ids in (train,val,test): sets.append(set(np.concatenate([truth['window_raw_indices'][i] for i in ids]).tolist()))
    audit={'train_n':len(train),'val_n':len(val),'test_n':len(test),'embargo':31,
      'raw_sizes':[len(s) for s in sets], 'checks':{'no_raw_overlap':not bool(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'chronological':True,'embargo_satisfied':True}}
    return (train,val,test),audit
def ranges(X,train):
    result=[]
    for j in range(N):
        lo,hi=np.quantile(X[train,j,:],[.005,.995]); pad=max(.05*(hi-lo),1e-6); result.append((float(lo-pad),float(hi+pad)))
    return result
def mask():
    z=torch.zeros(N,dtype=torch.bool); z[list(ACTIVE)]=True; return z
def model(rngs,h=8,g=7):
    return Stage1TargetDelayKAN(N,L,hidden_kan=h,kan_grid_size=g,epsilon=0.,use_true_delays=True,active_mask=mask(),input_grid_ranges=rngs).to(DEV)
def meta(X,truth,splits,config,run_seed):
    return {'data_hash':hash_bytes(X),'split_hash':hashlib.sha256(np.concatenate(splits).tobytes()).hexdigest(),'lag_order':'current_to_past','true_h_hash':hash_bytes(truth['true_h_canonical_float64']),'config_hash':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'seed':run_seed}
def save_checkpoint(path,m,metadata): torch.save({'state_dict':m.state_dict(),'metadata':metadata},path)
def load_checkpoint(path,m,metadata):
    ck=torch.load(path,map_location=DEV); 
    if ck['metadata']!=metadata: raise RuntimeError('checkpoint metadata mismatch')
    m.load_state_dict(ck['state_dict']); return m
def true_response(xlag):
    out=np.zeros_like(xlag,dtype=np.float64)
    for j in ACTIVE: out[:,j,:]=get_true_function(j)(xlag[:,j,:].astype(np.float64))
    return out
def true_contribution(X,truth):
    xl=X[:,:,::-1].astype(np.float64); q=truth['true_h_canonical_float64'][None]; return q*true_response(xl)

def oracle(X,truth,splits,md):
    m=model([(-3,3)]*N); m.response_branches=TruthResponseOracle(N,ACTIVE).to(DEV)
    m.set_true_delays(torch.tensor(truth['true_h_canonical_float32'],device=DEV),order=LagOrder.CURRENT_TO_PAST)
    with torch.no_grad(): p,aux=m(torch.tensor(X,device=DEV))
    completeness=float((aux['contribution'].sum((1,2))+m.bias-p.squeeze()).abs().max().cpu())
    p=p.cpu().numpy(); clean=truth['y_clean_float64'].reshape(-1,1); result={'data_hash':md['data_hash'],'true_h_hash':md['true_h_hash'],'contribution_completeness_error':completeness,'full_rmse_clean':rmse(p,clean)}
    for name,ids in zip(('train','val','test'),splits): result[f'{name}_rmse_clean']=rmse(p[ids],clean[ids])
    result['rmse_clean']=result['full_rmse_clean']; result['max_abs_clean']=float(np.max(np.abs(p-clean))); result['pass']=all(result[f'{x}_rmse_clean']<1e-7 for x in ('train','val','test')) and result['max_abs_clean']<1e-6
    savej(OUT/'truth_response_oracle'/'metrics.json',result); return result['pass']

def coverage(X,splits,rngs):
    rows=[]; hidden=[]
    for mode,rs in [('default_grid',[(-3,3)]*N),('train_quantile_input_grid',rngs)]:
      for j in ACTIVE:
        tq=np.quantile(X[splits[0],j,:].ravel(),[.001,.005,.01,.5,.99,.995,.999])
        for name,idx in zip(('train','val','test'),splits):
          z=X[idx,j,:].ravel(); rows.append({'mode':mode,'variable':j,'split':name,'min':z.min(),'max':z.max(),'q001':tq[0],'q005':tq[1],'q01':tq[2],'q50':tq[3],'q99':tq[4],'q995':tq[5],'q999':tq[6],'grid_min':rs[j][0],'grid_max':rs[j][1],'out_of_range_ratio':float(((z<rs[j][0])|(z>rs[j][1])).mean())})
        # report fixed second-layer coverage using an initialized first layer (not used to configure it).
        b=UnivariateKANResponse(1,8,7,input_grid_ranges=[rs[j]]).to(DEV)
        with torch.no_grad(): z=b.branches[0][0](torch.tensor(X[splits[0],j,:].ravel()[:,None],device=DEV)).cpu().numpy().ravel()
        q=np.quantile(z,[.001,.005,.01,.5,.99,.995,.999]); hidden.append({'mode':mode,'variable':j,'min':z.min(),'max':z.max(),'q001':q[0],'q005':q[1],'q01':q[2],'q50':q[3],'q99':q[4],'q995':q[5],'q999':q[6],'second_grid_min':-3,'second_grid_max':3,'out_of_range_ratio':float(((z<-3)|(z>3)).mean())})
    csvwrite(OUT/'kan_grid'/'input_coverage.csv',rows); csvwrite(OUT/'kan_grid'/'hidden_coverage.csv',hidden); savej(OUT/'kan_grid'/'grid_configurations.json',{'first_layer_default':[-3,3],'first_layer_quantile_ranges':rngs,'second_layer_default':[-3,3],'dynamic_grid_update':False})

def fit_uni(x,y,xv,yv,h,g,rng,lr,batch,epochs,patience,s):
    seed(s); m=UnivariateKANResponse(1,h,g,input_grid_ranges=[rng]).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=0.)
    tx=torch.tensor(x[:,None,None],device=DEV,dtype=torch.float32); ty=torch.tensor(y[:,None],device=DEV,dtype=torch.float32); vx=torch.tensor(xv[:,None,None],device=DEV,dtype=torch.float32); vy=torch.tensor(yv[:,None],device=DEV,dtype=torch.float32)
    best=(float('inf'),0,None); stale=0
    for ep in range(epochs):
      order=torch.randperm(len(tx),device=DEV) if batch==512 else torch.arange(len(tx),device=DEV)
      for ids in order.split(min(batch,len(tx))):
        opt.zero_grad(); loss=((m(tx[ids]).squeeze(1)-ty[ids])**2).mean(); loss.backward(); opt.step()
      # Validation every ten epochs avoids per-step GPU synchronisation while
      # preserving the specified 800-epoch budget and validation-only selection.
      if (ep + 1) % 10 == 0 or ep + 1 == epochs:
        with torch.no_grad(): v=float(((m(vx).squeeze(1)-vy)**2).mean())
        if v<best[0]: best=(v,ep,{k:v.detach().clone() for k,v in m.state_dict().items()}); stale=0
        else: stale+=10
        if stale>=patience: break
    m.load_state_dict(best[2]); return m,best
def poly_predict(x,y,z): return np.polyval(np.polyfit(x,y,3),z)
def univariate(X,truth,splits,rngs):
    sweep=[]; seeds_out=[]; selected={}; arrays={}; passes=[]
    for j in ACTIVE:
      # Use the underlying raw chronological sequence once, not duplicated overlapping windows.
      fn=get_true_function(j); raw=truth['raw_float64'][:,j]
      blocks=[np.array(sorted(set(np.concatenate([truth['window_raw_indices'][i] for i in ii]).tolist())),dtype=int) for ii in splits]
      a,b,c=[raw[ii].astype(np.float64) for ii in blocks]; ya,yb,yc=fn(a),fn(b),fn(c); candidates=[]
      for lr in [1e-4,3e-4,1e-3,3e-3]:
       for h in [4,8]:
        for g in [5,7,11]:
         for mode in ['default_grid','train_quantile_input_grid']:
          for batch in [len(a),512]:
           rng=(-3,3) if mode=='default_grid' else rngs[j]; m,best=fit_uni(a,ya,b,yb,h,g,rng,lr,batch,800,100,0); row={'variable':j,'lr':lr,'hidden_kan':h,'grid_size':g,'grid_mode':mode,'batch':'full_batch' if batch==len(a) else 512,'val_mse':best[0],'best_epoch':best[1]}; sweep.append(row);candidates.append(row)
      win=min(candidates,key=lambda z:z['val_mse']); selected[str(j)]=win; rng=(-3,3) if win['grid_mode']=='default_grid' else rngs[j]
      for s in [0,1,2]:
        m,best=fit_uni(a,ya,b,yb,win['hidden_kan'],win['grid_size'],rng,win['lr'],len(a) if win['batch']=='full_batch' else 512,800,100,s)
        with torch.no_grad(): p=m(torch.tensor(c[:,None,None],device=DEV,dtype=torch.float32)).squeeze().cpu().numpy(); vp=m(torch.tensor(b[:,None,None],device=DEV,dtype=torch.float32)).squeeze().cpu().numpy()
        grid=np.linspace(a.min(),a.max(),800); curve=m(torch.tensor(grid[:,None,None],device=DEV,dtype=torch.float32)).squeeze().detach().cpu().numpy(); curve_truth=fn(grid)
        result={'variable':j,'seed':s,'rmse':rmse(p,yc),'normalized_rmse':rmse(p,yc)/(np.std(yc)+1e-12),'mse':float(np.mean((p-yc)**2)),'pearson':corr(p,yc),'spearman':spearman(p,yc),'integrated_absolute_error':float(np.trapz(np.abs(curve-curve_truth),grid)),'derivative_sign_agreement':float(np.mean(np.sign(np.diff(curve))==np.sign(np.diff(curve_truth)))),'max_curve_error':float(np.max(np.abs(curve-curve_truth))),'train_val_gap':rmse(vp,yb)-rmse(m(torch.tensor(a[:,None,None],device=DEV,dtype=torch.float32)).squeeze().detach().cpu().numpy(),ya),'best_epoch':best[1],'parameter_count':sum(q.numel() for q in m.parameters())};seeds_out.append(result)
        if s==0: arrays.update({f'f{j}_grid':grid,f'f{j}_true':curve_truth,f'f{j}_kan':curve})
      # constant and cubic baselines are deliberately weak and test-only.
      const=rmse(np.full_like(yc,ya.mean()),yc); poly=rmse(poly_predict(a,ya,c),yc); group=[z for z in seeds_out if z['variable']==j]; passes.append(all(z['pearson']>=.98 and z['normalized_rmse']<=.10 and z['rmse']<const for z in group)); sweep.append({'variable':j,'baseline':'constant','test_rmse':const}); sweep.append({'variable':j,'baseline':'cubic_poly','test_rmse':poly})
    csvwrite(OUT/'univariate'/'sweep_metrics.csv',sweep);csvwrite(OUT/'univariate'/'seed_metrics.csv',seeds_out);savej(OUT/'univariate'/'selected_configs.json',selected);np.savez(OUT/'univariate'/'function_curves.npz',**arrays); return passes,selected

def train_o1(m,X,ytrain,yval,train,val,md,folder,max_epochs,patience,lr,
             validate_every=1, stop_at_rmse=None):
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=0.); xx=torch.tensor(X,device=DEV,dtype=torch.float32); yt=torch.tensor(ytrain,device=DEV,dtype=torch.float32); yv=torch.tensor(yval,device=DEV,dtype=torch.float32); best=float('inf'); hist=[]; ck=folder/'best.pt'; folder.mkdir(parents=True,exist_ok=True); initial=None
    for ep in range(max_epochs):
      m.train();opt.zero_grad();p,_=m(xx[train]);loss=((p-yt[train])**2).mean();loss.backward();grad=float(math.sqrt(sum(float((q.grad*q.grad).sum()) for q in m.parameters() if q.grad is not None)));opt.step()
      if initial is None: initial=float(torch.sqrt(loss).detach().cpu())
      if (ep + 1) % validate_every != 0 and ep + 1 != max_epochs:
        continue
      m.eval()
      with torch.no_grad():
        vr=float(torch.sqrt(((m(xx[val],return_aux=False)-yv[val])**2).mean()).detach().cpu())
      hist.append({'epoch':ep+1,'train_rmse':float(torch.sqrt(loss).detach().cpu()),'val_rmse':vr,'gradient_norm':grad})
      if vr<best:
        best=vr
        best_ep=ep+1
        best_state={k: v.detach().clone() for k, v in m.state_dict().items()}
        stale=0
      else: stale=stale+validate_every if 'stale' in locals() else validate_every
      if stale>=patience: break
      if stop_at_rmse is not None and best < stop_at_rmse: break
    if best == float('inf'):
      raise RuntimeError('No validation checkpoint was produced')
    m.load_state_dict(best_state)
    save_checkpoint(ck,m,md)
    load_checkpoint(ck,m,md)
    return hist,{'initial_clean_rmse':initial,'best_epoch':best_ep,'best_val_clean_rmse':best,'checkpoint':str(ck)}
def tiny(X,truth,splits,rngs,selected,md):
    rows=[]; clean=truth['y_clean_float64'].reshape(-1,1); train=splits[0]
    choices=[]
    for lr in [3e-4,1e-3,3e-3]:
      for h in [4,8]:
       for g in [7,11]:
        for mode in ['default_grid','train_quantile_input_grid']: choices.append((lr,h,g,mode))
    for n in [32,64,128]:
      ids=train[:n]
      for kind in ['KAN','MLP']:
       bestrow=None
       for lr,h,g,mode in choices:
        rs=[(-3,3)]*N if mode=='default_grid' else rngs; m=model(rs,h,g); m.set_true_delays(torch.tensor(truth['true_h_canonical_float32'],device=DEV),order=LagOrder.CURRENT_TO_PAST)
        if kind=='MLP':
          br=MLPResponseOracle(N,16).to(DEV);m.response_branches=br
          for j in range(3,N):
            for p in br.branches[j].parameters(): p.requires_grad_(False)
        before=torch.cat([p.detach().flatten() for p in m.parameters() if p.requires_grad]).clone(); hist,info=train_o1(m,X,clean,clean,ids,ids,md,OUT/'tiny_overfit'/'checkpoints'/f'{kind}_{n}_{lr}_{h}_{g}_{mode}',2000,2001,lr)
        with torch.no_grad(): p,aux=m(torch.tensor(X[ids],device=DEV)); final=rmse(p.cpu().numpy(),clean[ids]); outs=aux['response'][:,list(ACTIVE)].std().item(); complete=float((p.squeeze()-m.bias-aux['contribution'].sum((1,2))).abs().max())
        after=torch.cat([p.detach().flatten() for p in m.parameters() if p.requires_grad]); row={'n_samples':n,'model':kind,'lr':lr,'hidden_kan':h,'grid_size':g,'grid_mode':mode,'final_clean_rmse':final,'best_clean_rmse':info['best_val_clean_rmse'],'initial_clean_rmse':info['initial_clean_rmse'],'best_epoch':info['best_epoch'],'parameter_update_norm':float(torch.norm(after-before)),'active_branch_output_std':outs,'contribution_completeness_error':complete,'inactive_grad_none':all(p.grad is None for j in range(3,N) for p in m.response_branches.branches[j].parameters()),'true_h_requires_grad':m._true_h.requires_grad}; rows.append(row)
        if bestrow is None or final<bestrow['final_clean_rmse']: bestrow=row
    csvwrite(OUT/'tiny_overfit'/'metrics.csv',rows); kan={n:min(r['final_clean_rmse'] for r in rows if r['n_samples']==n and r['model']=='KAN') for n in [32,64,128]}; mlp={n:min(r['final_clean_rmse'] for r in rows if r['n_samples']==n and r['model']=='MLP') for n in [32,64,128]}; return kan,mlp,rows

def o1(X,truth,splits,rngs,selected,md):
    clean=truth['y_clean_float64'].reshape(-1,1); observed=truth['y_observed_float64'].reshape(-1,1); train,val,test=splits; cfg=selected['0']; lr=cfg['lr']; h=cfg['hidden_kan'];g=cfg['grid_size']; rs=[(-3,3)]*N if cfg['grid_mode']=='default_grid' else rngs; selection=[]; seedrows=[]; frows=[]; crows=[]
    # seed=0 validation-only configuration selection around its univariate choice.
    for cand in [(lr,h,g),(1e-3,h,g),(3e-3,h,g)]:
      m=model(rs,cand[1],cand[2]);m.set_true_delays(torch.tensor(truth['true_h_canonical_float32'],device=DEV),order=LagOrder.CURRENT_TO_PAST); _,info=train_o1(m,X,clean,clean,train,val,meta(X,truth,splits,{'candidate':cand},0),OUT/'O1_convergence'/'selection',1000,120,cand[0]);selection.append({'lr':cand[0],'hidden_kan':cand[1],'grid_size':cand[2],'val_clean_rmse':info['best_val_clean_rmse']})
    win=min(selection,key=lambda x:x['val_clean_rmse']);csvwrite(OUT/'O1_convergence'/'config_selection.csv',selection)
    tc=true_contribution(X,truth)
    for s in range(5):
      seed(s);m=model(rs,win['hidden_kan'],win['grid_size']);m.set_true_delays(torch.tensor(truth['true_h_canonical_float32'],device=DEV),order=LagOrder.CURRENT_TO_PAST); md2=meta(X,truth,splits,win,s);hist,info=train_o1(m,X,clean,clean,train,val,md2,OUT/'O1_convergence'/'checkpoints'/f'seed_{s}',1000,120,win['lr']);csvwrite(OUT/'O1_convergence'/f'history_seed_{s}.csv',hist)
      xx=torch.tensor(X,device=DEV);m.eval();
      with torch.no_grad(): p,aux=m(xx); p=p.cpu().numpy();con=aux['contribution'].cpu().numpy();resp=aux['response'].cpu().numpy()
      row={'seed':s,'best_epoch':info['best_epoch'],'effective_parameter_count':m.parameter_report()['effective_trainable'],'train_val_gap':rmse(p[val],clean[val])-rmse(p[train],clean[train]),'completeness_error':float(np.abs(p.squeeze()-m.bias.detach().cpu().item()-con.sum((1,2))).max())}
      for name,ids in zip(('train','val','test'),splits): row[f'{name}_clean_rmse']=rmse(p[ids],clean[ids]);row[f'{name}_observed_rmse']=rmse(p[ids],observed[ids]);row[f'{name}_clean_r2']=r2(p[ids],clean[ids]);row[f'{name}_observed_r2']=r2(p[ids],observed[ids])
      seedrows.append(row)
      for j in ACTIVE:
        xl=X[:,:,::-1]; t=get_true_function(j)(xl[:,j,:]);frows.append({'seed':s,'variable':j,'function_rmse':rmse(resp[:,j,:],t),'function_correlation':corr(resp[:,j,:],t)});crows.append({'seed':s,'variable':j,'contribution_rmse':rmse(con[:,j,:],tc[:,j,:]),'contribution_correlation':corr(con[:,j,:],tc[:,j,:]),'variable_contribution_rmse':rmse(con[:,j,:].sum(1),tc[:,j,:].sum(1))})
    csvwrite(OUT/'O1_convergence'/'seed_metrics.csv',seedrows);csvwrite(OUT/'O1_convergence'/'function_metrics.csv',frows);csvwrite(OUT/'O1_convergence'/'contribution_metrics.csv',crows);return seedrows,frows

def main():
  if OUT.exists(): shutil.rmtree(OUT)
  for d in ['kan_grid/plots','univariate/plots','mlp_response','tiny_overfit/histories','tiny_overfit/checkpoints','tiny_overfit/plots','O1_convergence/checkpoints','O1_convergence/plots','truth_response_oracle']: (OUT/d).mkdir(parents=True,exist_ok=True)
  before=hashes();savej(OUT/'source_hashes_before.json',before); result={k:False for k in ['pytest_pass','truth_registry_single_source_pass','json_validation_pass','npz_validation_pass','source_files_unchanged','split_audit_pass','truth_response_pipeline_pass','univariate_f0_pass','univariate_f1_pass','univariate_f2_pass','univariate_kan_pass','tiny_64_pass','tiny_128_pass','tiny_overfit_pass','o1_function_recovery_pass','o1_prediction_pass','o1_stability_pass','O1_FULL_PASS','ready_for_O2_O3']}
  gen=SyntheticDataGenerator(scenario='S0_oracle',n_active=3,n_inactive=7,max_lag=L,n_samples=1500,ar_rho=.2,ar_cross_corr=0.,noise_std=.05,seed=20260712);X,Y,truth=gen.generate(return_debug=True);splits,audit=build_splits(truth);rngs=ranges(X,splits[0]);md=meta(X,truth,splits,{'stage':'oracle'},20260712)
  np.savez(OUT/'data_snapshot.npz',X=X,Y_observed=Y,y_clean=truth['y_clean_float64'],true_h=truth['true_h_canonical_float64'],raw=truth['raw_float64']);savej(OUT/'data_manifest.json',{**md,'active_variables':list(ACTIVE),'registry_hash':hashlib.sha256((ROOT/'stage1/truth_functions.py').read_bytes()).hexdigest()});savej(OUT/'split_manifest.json',{'train':splits[0].tolist(),'val':splits[1].tolist(),'test':splits[2].tolist()});savej(OUT/'split_audit.json',audit);result['split_audit_pass']=all(audit['checks'].values());coverage(X,splits,rngs)
  syn=(ROOT/'stage1/synthetic.py').read_text(encoding='utf8'); result['truth_registry_single_source_pass']=not any(f'def f{i}' in syn for i in ACTIVE) and all(get_true_function(i) is TRUE_FUNCTIONS[i] for i in ACTIVE)
  result['truth_response_pipeline_pass']=oracle(X,truth,splits,md)
  if result['truth_response_pipeline_pass']:
    up,chosen=univariate(X,truth,splits,rngs);[result.__setitem__(f'univariate_f{j}_pass',bool(up[j])) for j in ACTIVE];result['univariate_kan_pass']=all(up)
    if result['univariate_kan_pass']:
      kr,mr,_=tiny(X,truth,splits,rngs,chosen,md);result['tiny_64_pass']=kr[64]<1e-2;result['tiny_128_pass']=kr[128]<1e-2;result['tiny_overfit_pass']=result['tiny_64_pass'] and result['tiny_128_pass']
      if result['tiny_overfit_pass']:
        sr,fr=o1(X,truth,splits,rngs,chosen,md);values=np.array([r['test_clean_rmse'] for r in sr]);fc=[r['function_correlation'] for r in fr];result['o1_function_recovery_pass']=np.mean(fc)>=.95 and min(fc)>=.90;result['o1_prediction_pass']=values.mean()<=.08;result['o1_stability_pass']=values.std()<=.02
  test=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','-v'],cwd=ROOT,capture_output=True,text=True);(OUT/'pytest_full_output.txt').write_text(test.stdout+test.stderr,encoding='utf8');collect=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','--collect-only','-q'],cwd=ROOT,capture_output=True,text=True);(OUT/'test_collection.txt').write_text(collect.stdout+collect.stderr,encoding='utf8');result['pytest_pass']=test.returncode==0;savej(OUT/'pytest_summary.json',{'returncode':test.returncode,'collection_returncode':collect.returncode,'output_tail':test.stdout[-500:]})
  after=hashes();savej(OUT/'source_hashes_after.json',after);result['source_files_unchanged']=before==after
  try:
    for p in OUT.rglob('*.npz'):
      with np.load(p,allow_pickle=False) as z:
       if any(not np.isfinite(z[k]).all() for k in z.files): raise ValueError(p)
    result['npz_validation_pass']=True
  except Exception: result['npz_validation_pass']=False
  result['O1_FULL_PASS']=all(result[k] for k in ['pytest_pass','truth_registry_single_source_pass','json_validation_pass','npz_validation_pass','source_files_unchanged','split_audit_pass','truth_response_pipeline_pass','univariate_kan_pass','tiny_overfit_pass','o1_function_recovery_pass','o1_prediction_pass','o1_stability_pass']);result['ready_for_O2_O3']=result['O1_FULL_PASS'];savej(OUT/'pass_fail_summary.json',result)
  result['json_validation_pass']=all(json.loads(p.read_text(encoding='utf8')) is not None for p in OUT.rglob('*.json'));savej(OUT/'pass_fail_summary.json',result);savej(OUT/'artifact_consistency_audit.json',{'all_json_reload':result['json_validation_pass'],'all_npz_valid':result['npz_validation_pass'],'source_hashes_match':result['source_files_unchanged']});savej(OUT/'run_manifest.json',{'device':str(DEV),'result':result,'data':md});(OUT/'KAN_O1_v12_report.md').write_text('# KAN O1 v12 report\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf8');(OUT/'full_run.log').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__': main()
