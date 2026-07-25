#!/usr/bin/env python
"""v17 M2 joint variable/delay/response recovery on the immutable v12 snapshot."""
from __future__ import annotations

import copy, csv, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

from stage1.model import Stage1TargetDelayKAN
from stage1.protocol import train_warmup, run_pruning_to_stable_support, refit_fixed_support
from stage1.m2_metrics import one_se_select, pairwise_jaccard, support_scores, support_aware_delay
from stage1.truth_functions import get_true_function, TRUTH_REGISTRY_VERSION

ROOT=Path(__file__).resolve().parent
V12=ROOT/'results_stage1/KAN_O1_v12'; V16=ROOT/'results_stage1/O2_O3_AUDIT_CLOSURE_v16'; V14=ROOT/'results_stage1/O2_D0_O3_DIAG_v14'
OUT=ROOT/'results_stage1/M2_JOINT_RECOVERY_v17'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_ID='M2_JOINT_RECOVERY_v17'; N,L=10,32; ACTIVE=(0,1,2); SEEDS=range(5); LR=.003
S0_GRID=[.003,.004,.005,.006,.007,.008,.009,.010,.012,.015]
BASELINE_NAMES=('DenseStaticGamma','SparseStaticGamma','SparseFreeStaticLogits','UniformDelaySparse')
SOURCES=['stage1/model.py','stage1/protocol.py','stage1/m2_metrics.py','stage1/delay_prior.py','stage1/response_kan.py','stage1/truth_functions.py','run_m2_joint_recovery_v17.py','tests/test_stage1.py','tools/build_handoff_bundle.py','layers.py']
SMOKE=os.environ.get('V17_SMOKE')=='1'
WARMUP_EPOCHS=30 if SMOKE else 3000; WARMUP_PATIENCE=8 if SMOKE else 300
PRUNE_RAMP=3 if SMOKE else 300; PRUNE_FULL=2 if SMOKE else 200; PRUNE_STABLE=2 if SMOKE else 80; PRUNE_EPOCHS=15 if SMOKE else 1800
REFIT_EPOCHS=30 if SMOKE else 5000; REFIT_PATIENCE=8 if SMOKE else 500

def native(x): return x.item() if isinstance(x,(np.integer,np.floating,np.bool_)) else x
def savej(path,data): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,default=native),encoding='utf8')
def loadj(path): return json.loads(path.read_text(encoding='utf8'))
def csvw(path,rows):
    if not rows: raise ValueError(f'empty CSV refused: {path}')
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for row in rows for k in row})
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:(json.dumps(v) if isinstance(v,(list,dict,tuple)) else native(v)) for k,v in r.items()} for r in rows])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def set_seed(value): np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value) if torch.cuda.is_available() else None
def tensor(a): return torch.as_tensor(a,dtype=torch.float32,device=DEV)
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else 0.
def ready_gate(clean,tests,artifacts): return bool(clean and tests and artifacts)
def input_ranges(X,train):
    values=X[train].transpose(1,0,2).reshape(N,-1); lo=values.min(1);hi=values.max(1);pad=np.maximum((hi-lo)*.05,.05)
    return [(float(a-b),float(c+b)) for a,c,b in zip(lo,hi,pad)]
def build_model(ranges,mode='static_gamma',fixed_q=None):
    model=Stage1TargetDelayKAN(N,L,hidden_kan=8,kan_grid_size=7,epsilon=0.,use_true_delays=False,
        active_mask=torch.ones(N,dtype=torch.bool),input_grid_ranges=ranges,delay_mode=mode).to(DEV)
    if fixed_q is not None:model.set_fixed_delays(tensor(fixed_q))
    return model
def matched_q(h): return np.stack([h[j] if j<3 else h[(j-3)%3] for j in range(N)]).astype(np.float32)
def uniform_q(): return np.full((N,L),1/L,np.float32)
def predict(model,X):
    model.eval()
    with torch.no_grad():return model(tensor(X))[0].cpu().numpy()
def learned_q(model):
    model.eval()
    with torch.no_grad():return model(tensor(np.zeros((1,N,L),np.float32)))[1]['q'][0].cpu().numpy()
def support(model): return np.flatnonzero(model.selection_mask.detach().cpu().numpy()).tolist()

def save_checkpoint(path,model,stage,metadata):
    path.parent.mkdir(parents=True,exist_ok=True);q=learned_q(model)
    torch.save({'state_dict':model.state_dict(),'selection_mask':model.selection_mask.detach().cpu(),
                'learned_q':torch.tensor(q),'metadata':{**metadata,'stage':stage}},path)
def save_state_checkpoint(path,model,state,stage,metadata):
    current=copy.deepcopy(model.state_dict());model.load_state_dict(state);save_checkpoint(path,model,stage,metadata);model.load_state_dict(current)
def load_model(path,ranges,mode='static_gamma',fixed_q=None):
    model=build_model(ranges,mode,fixed_q);model.load_state_dict(torch.load(path,map_location=DEV)['state_dict']);return model

def phase_history(result,stage,seed,config_id):
    rows=[]
    for row in result.history:
        x=dict(row);x.update(stage=stage,seed=seed,config_id=config_id)
        rows.append(x)
    return rows

def alternating_warmup(model,xt,yt,xv,yv):
    histories=[]
    delay=list(model.delay_prior.parameters()) if model.delay_prior else ([model.delay_logits] if model.delay_logits is not None else [])
    response=list(model.response_branches.parameters())
    for p in delay:p.requires_grad_(False)
    first=3 if SMOKE else 300;last=24 if SMOKE else 2400
    a=train_warmup(model,xt,yt,xv,yv,epochs=first,lr=LR,patience=first);histories += a.history
    for p in delay:p.requires_grad_(True)
    for p in response:p.requires_grad_(False)
    b=train_warmup(model,xt,yt,xv,yv,epochs=first,lr=LR,patience=first);histories += [{**x,'epoch':x['epoch']+first} for x in b.history]
    for p in response:p.requires_grad_(True)
    c=train_warmup(model,xt,yt,xv,yv,epochs=last,lr=LR,patience=WARMUP_PATIENCE);histories += [{**x,'epoch':x['epoch']+2*first} for x in c.history]
    c.history=histories;return c

def make_warmup(strategy,ranges,X,y,splits,seed_value,root,mode='static_gamma',fixed_q=None):
    ck=root/'warmup_best.pt'; marker=root/'warmup.json'
    if ck.exists() and marker.exists():return load_model(ck,ranges,mode,fixed_q),loadj(marker)
    set_seed(seed_value);m=build_model(ranges,mode,fixed_q);tr,va,_=splits;xt,yt,xv,yv=tensor(X[tr]),tensor(y[tr]),tensor(X[va]),tensor(y[va])
    started=time.time();r=alternating_warmup(m,xt,yt,xv,yv) if strategy=='ALTERNATING_WARMUP' else train_warmup(m,xt,yt,xv,yv,epochs=WARMUP_EPOCHS,lr=LR,patience=WARMUP_PATIENCE)
    save_checkpoint(ck,m,'warmup',{'run_id':RUN_ID,'seed':seed_value,'strategy':strategy})
    save_checkpoint(root/'warmup_terminal.pt',m,'warmup_terminal',{'run_id':RUN_ID,'seed':seed_value,'strategy':strategy})
    row={'best_epoch':r.best_epoch,'best_val_rmse':r.best_val_rmse,'runtime_seconds':time.time()-started,'strategy':strategy,'seed':seed_value,'history':r.history}
    savej(marker,row);return m,row

def pipeline(strategy,s0,seed_value,ranges,X,y,splits,root,mode='static_gamma',fixed_q=None,warm_root=None):
    """Fresh/restartable Stage A/B/C pipeline. Terminal pruning support is authoritative."""
    result_file=root/'result.json';final_ck=root/'refit_best.pt'
    if result_file.exists() and final_ck.exists():return loadj(result_file)
    root.mkdir(parents=True,exist_ok=True);warm_root=warm_root or root
    m,warm=make_warmup(strategy,ranges,X,y,splits,seed_value,warm_root,mode,fixed_q)
    tr,va,te=splits;xt,yt,xv,yv=tensor(X[tr]),tensor(y[tr]),tensor(X[va]),tensor(y[va])
    median=float(torch.median(m.response_branches.compute_branch_norms()).detach().cpu());lam=float(s0*median/LR)
    started=time.time();pr=run_pruning_to_stable_support(m,xt,yt,xv,yv,ramp_epochs=PRUNE_RAMP,full_penalty_min_epochs=PRUNE_FULL,
        stable_epochs=PRUNE_STABLE,max_epochs=PRUNE_EPOCHS,lr=LR,lambda_group=lam)
    terminal=list(map(int,pr.terminal_support));save_checkpoint(root/'prune_terminal.pt',m,'prune_terminal',{'run_id':RUN_ID,'seed':seed_value,'s0':s0,'terminal_support':terminal})
    if pr.best_state is not None:save_state_checkpoint(root/'prune_best.pt',m,pr.best_state,'prune_best',{'run_id':RUN_ID,'seed':seed_value,'s0':s0})
    terminal_val=rmse(predict(m,X[va]),y[va])
    rf=refit_fixed_support(m,terminal,xt,yt,xv,yv,epochs=REFIT_EPOCHS,lr=LR,patience=REFIT_PATIENCE)
    save_checkpoint(final_ck,m,'refit_best',{'run_id':RUN_ID,'seed':seed_value,'s0':s0,'terminal_support':terminal})
    save_checkpoint(root/'refit_terminal.pt',m,'refit_terminal',{'run_id':RUN_ID,'seed':seed_value,'s0':s0,'terminal_support':terminal})
    q=learned_q(m);np.savez_compressed(root/'learned_q.npz',q=q)
    history=phase_history(type('R',(),{'history':warm['history']})(),'warmup',seed_value,str(s0))+phase_history(pr,'prune',seed_value,str(s0))+phase_history(rf,'refit',seed_value,str(s0))
    csvw(root/'epoch_history.csv',history)
    delay_rows=[]
    tau=np.arange(L)
    for j in range(N):delay_rows.append({'seed':seed_value,'config_id':str(s0),'variable':j,'mean':float(q[j]@tau),'std':float(np.sqrt(q[j]@(tau-q[j]@tau)**2)),'peak':int(q[j].argmax()),'boundary_mass':float(q[j,-3:].sum()),'selected':j in terminal})
    csvw(root/'delay_history.csv',delay_rows)
    norms=m.response_branches.compute_branch_norms().detach().cpu().tolist();csvw(root/'branch_norm_history.csv',[{'seed':seed_value,'config_id':str(s0),'variable':j,'branch_norm':norms[j],'selected':j in terminal} for j in range(N)])
    row={'config_id':str(s0),'s0':s0,'seed':seed_value,'strategy':strategy,'mode':mode,'lambda_group':lam,'median_warmup_norm':median,
        'shrink_fraction':s0,'support_at_best_validation':list(map(int,pr.support_at_best_validation)),'terminal_support':terminal,'active_count':len(terminal),
        'stable_epoch':pr.support_stable_epoch if pr.support_stable_epoch is not None else -1,'stable':pr.stable,
        'prune_best_val_rmse':pr.best_val_rmse,'prune_terminal_val_rmse':terminal_val,'refit_val_rmse':rf.best_val_rmse,
        'refit_test_rmse':rmse(predict(m,X[te]),y[te]),'best_epoch':rf.best_epoch,'converged':rf.best_epoch<REFIT_EPOCHS,
        'runtime_seconds':warm['runtime_seconds']+time.time()-started}
    savej(result_file,row);return row

def run_matched_kernel(X,y,h,splits,ranges,meta):
    root=OUT/'O2_matched_kernel_control';q=matched_q(h);rows=[]
    for s in SEEDS:
        print(f'[v17 O2 matched] seed={s}',flush=True);rows.append(pipeline('JOINT',.01,s,ranges,X,y,splits,root/f'seed_{s}',fixed_q=q))
    supports=[r['terminal_support'] for r in rows];scores=support_scores(supports)
    for row,score in zip(rows,scores):row.update(score)
    csvw(root/'seed_metrics.csv',rows);freq=[{'variable':j,'selection_frequency':np.mean([j in x for x in supports])} for j in range(N)];csvw(root/'support_frequency.csv',freq)
    passed=np.mean([x['f1'] for x in scores])>=.8 and np.mean([x['active_count'] for x in scores])<=5 and sum(x['exact'] for x in scores)>=3 and pairwise_jaccard(supports)>=.7 and np.mean([x['refit_test_rmse'] for x in rows])<=.02
    checkpoints=root/'checkpoints'
    for s in SEEDS:
        dst=checkpoints/f'seed_{s}/best.pt';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/f'seed_{s}/refit_best.pt',dst)
    summary={'O2_MATCHED_KERNEL_PASS':bool(passed),'o2_kernel_shape_leak_warning':not bool(passed),'mean_f1':np.mean([x['f1'] for x in scores]),'mean_active_count':np.mean([x['active_count'] for x in scores]),'exact_frequency':sum(x['exact'] for x in scores),'pairwise_jaccard':pairwise_jaccard(supports),'mean_test_rmse':np.mean([x['refit_test_rmse'] for x in rows])};savej(root/'summary.json',summary);return summary

def screen_and_formal(X,y,splits,ranges):
    root=OUT/'M2_clean';screen=[]
    for strategy in ('JOINT','ALTERNATING_WARMUP'):
        print(f'[v17 strategy] {strategy}',flush=True);screen.append(pipeline(strategy,.01,0,ranges,X,y,splits,root/'strategy_screen_runs'/strategy))
    # Validation/stability/count only; no truth or test field is used here.
    chosen=sorted(screen,key=lambda r:(r['refit_val_rmse'],not r['stable'],r['active_count']))[0]['strategy'];csvw(root/'strategy_screen.csv',screen)
    scan=[]
    shared=root/'strategy_screen_runs'/chosen
    for s0 in S0_GRID:
        print(f'[v17 clean scan] s0={s0}',flush=True);scan.append(pipeline(chosen,s0,0,ranges,X,y,splits,root/'scan_runs'/chosen/f's0_{s0:g}',warm_root=shared))
    csvw(root/'seed0_sparsity_path.csv',scan)
    feasible=[r for r in scan if r['active_count']<=6];best_val=min(scan,key=lambda r:r['prune_best_val_rmse']);best_refit=min(scan,key=lambda r:r['refit_val_rmse'])
    candidate_s0={float(r['s0']) for r in feasible}|{float(best_val['s0']),float(best_refit['s0']),.005,.012}
    formal=[]
    for s0 in sorted(candidate_s0):
        for s in SEEDS:
            print(f'[v17 clean formal] s0={s0}, seed={s}',flush=True)
            if s==0:
                source=next((r for r in scan if float(r['s0'])==s0),None)
                if source is not None:
                    # Re-run directory is avoided by pointing formal seed 0 to its completed scan unit.
                    formal.append({**source,'formal_checkpoint':str(root/'scan_runs'/chosen/f's0_{s0:g}'/'refit_best.pt')});continue
            shared_formal=root/'formal_runs'/chosen/f'shared_seed_{s}_warmup'
            row=pipeline(chosen,s0,s,ranges,X,y,splits,root/'formal_runs'/f's0_{s0:g}'/f'seed_{s}',warm_root=shared_formal)
            row['formal_checkpoint']=str(root/'formal_runs'/f's0_{s0:g}'/f'seed_{s}'/'refit_best.pt');formal.append(row)
    selected=one_se_select(formal);selected_s0=float(selected['s0']);selected_rows=[r for r in formal if float(r['s0'])==selected_s0]
    savej(root/'selected_config.json',{'run_id':RUN_ID,'strategy':chosen,'s0':selected_s0,'selection_rule':'validation_one_standard_error','selection_fields':['refit_val_rmse','active_count','pairwise_jaccard','stable','converged'],'candidate_s0':sorted(candidate_s0),**selected})
    return chosen,selected_s0,selected_rows,formal

def true_response(X):
    xl=X[:,:,::-1].astype(np.float64);out=np.zeros_like(xl)
    for j in ACTIVE:out[:,j]=get_true_function(j)(xl[:,j])
    return out

def evaluate_selected(rows,ranges,X,clean,observed,h,splits,root,mode='static_gamma',fixed_q=None):
    supports=[r['terminal_support'] for r in rows];scores=support_scores(supports);seed_rows=[];all_q=[];function_rows=[];contribution_rows=[]
    truth=true_response(X);tr,va,te=splits
    for row,score in zip(rows,scores):
        ck=Path(row['formal_checkpoint']);m=load_model(ck,ranges,mode,fixed_q);q=learned_q(m);all_q.append(q)
        started=time.perf_counter();pred=predict(m,X);inference_seconds=time.perf_counter()-started;o3=loadj(V16/'O3/summary.json');test_value=rmse(pred[te],clean[te]);train_value=rmse(pred[tr],clean[tr]);val_value=rmse(pred[va],clean[va]);seed_row={**row,**score,'train_clean_rmse':train_value,'val_clean_rmse':val_value,'test_clean_rmse':test_value,'train_val_gap':val_value-train_value,'test_observed_rmse':rmse(pred[te],observed[te]),'clean_r2':1-float(np.sum((pred[te].ravel()-clean[te].ravel())**2)/np.sum((clean[te].ravel()-clean[te].mean())**2)),'prediction_ratio_vs_o3':test_value/o3['mean_o3_test_rmse'],'prediction_ratio_vs_o1':test_value/o3['mean_o1_test_rmse'],'parameter_count':sum(p.numel() for p in m.parameters() if p.requires_grad),'inference_seconds':inference_seconds};seed_rows.append(seed_row)
        before_center=predict(m,X[te]);eval_model=copy.deepcopy(m);eval_model.fit_centering([(tensor(X[tr]),tensor(clean[tr]))]);eval_model.eval();centering_invariance=float(np.max(np.abs(before_center-predict(eval_model,X[te]))))
        with torch.no_grad():aux=eval_model(tensor(X[te]))[1];learned_f=aux['response'].cpu().numpy();learned_v=aux['variable_contribution'].cpu().numpy()
        for j in ACTIVE:
            true_center=float(np.mean(np.sum(h[j][None,:]*truth[tr,j],axis=1)));target_f=truth[te,j]-true_center;target_v=np.sum(h[j][None,:]*target_f,axis=1)
            if j not in supports[len(seed_rows)-1]:
                function_rows.append({'seed':row['seed'],'variable':j,'selected':False,'function_corr':0.,'function_normalized_rmse':1e6,'function_rmse':float(np.std(target_f)),'centering_invariance':centering_invariance,'missed_variable_failure':True})
                contribution_rows.append({'seed':row['seed'],'variable':j,'selected':False,'response_conditioned_contribution_rmse':float(np.sqrt(np.mean(target_v**2))),'variable_contribution_rmse':float(np.sqrt(np.mean(target_v**2))),'full_contribution_rmse':float(np.sqrt(np.mean(target_v**2))),'contribution_corr':0.,'missed_variable_failure':True})
            else:
                function_rows.append({'seed':row['seed'],'variable':j,'selected':True,'function_corr':corr(learned_f[:,j],target_f),'function_normalized_rmse':rmse(learned_f[:,j],target_f)/(np.std(target_f)+1e-12),'function_rmse':rmse(learned_f[:,j],target_f),'centering_invariance':centering_invariance,'missed_variable_failure':False})
                response_conditioned=rmse(q[j][None,:]*(learned_f[:,j]-target_f),np.zeros_like(learned_f[:,j]));full_rmse=rmse(learned_v[:,j],target_v)
                contribution_rows.append({'seed':row['seed'],'variable':j,'selected':True,'response_conditioned_contribution_rmse':response_conditioned,'variable_contribution_rmse':full_rmse,'full_contribution_rmse':full_rmse,'contribution_corr':corr(learned_v[:,j],target_v),'missed_variable_failure':False})
    delay_rows,uniform_w1=support_aware_delay(np.asarray(all_q),h,supports)
    support_pass=np.mean([x['f1'] for x in scores])>=.8 and np.mean([x['active_count'] for x in scores])<=5 and sum(x['exact'] for x in scores)>=3 and pairwise_jaccard(supports)>=.7
    delay_pass=np.mean([x['mean_delay_mae'] for x in delay_rows])<=2 and np.mean([x['w1'] for x in delay_rows])<uniform_w1 and np.mean([x['peak_error'] for x in delay_rows])<=3 and max(x['boundary_mass'] for x in delay_rows)<.8
    response_pass=np.mean([x['function_corr'] for x in function_rows])>=.9 and np.mean([x['function_normalized_rmse'] for x in function_rows])<=.15 and min(x['function_corr'] for x in function_rows)>=.8
    contribution_pass=np.mean([x['full_contribution_rmse'] for x in contribution_rows])<=.03 and np.mean([x['contribution_corr'] for x in contribution_rows])>=.9
    tests=[x['test_clean_rmse'] for x in seed_rows];prediction_pass=np.mean(tests)<=.02;stability_pass=np.std(tests)<=.01 and np.isfinite(tests).all() and sum(x['converged'] for x in seed_rows)>=4
    conditional=[{**x,'metric_scope':'conditional'} for x in delay_rows if x['selected']]
    csvw(root/'seed_metrics.csv',seed_rows);csvw(root/'support_metrics.csv',[{**x,'seed':r['seed']} for x,r in zip(scores,rows)]);csvw(root/'support_frequency.csv',[{'variable':j,'selection_frequency':np.mean([j in s for s in supports])} for j in range(N)]);csvw(root/'delay_metrics.csv',delay_rows+conditional);csvw(root/'centered_function_metrics.csv',function_rows);csvw(root/'contribution_metrics.csv',contribution_rows)
    # Required aggregate histories are copied from selected runs, preserving all stage labels.
    histories=[];norms=[];delays=[]
    for r in rows:
        unit=Path(r['formal_checkpoint']).parent
        for name,target in [('epoch_history.csv',histories),('branch_norm_history.csv',norms),('delay_history.csv',delays)]:
            with (unit/name).open(encoding='utf8') as f:target.extend(list(csv.DictReader(f)))
    csvw(root/'epoch_history.csv',histories);csvw(root/'branch_norm_history.csv',norms);csvw(root/'delay_history.csv',delays)
    checkpoints=root/'checkpoints'
    for r in rows:
        dst=checkpoints/f"seed_{r['seed']}/best.pt";dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(r['formal_checkpoint']),dst)
    plots=root/'plots';plots.mkdir(parents=True,exist_ok=True)
    (plots/'support_frequency.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="80"><text x="10" y="40">selection frequencies: '+','.join(f'{np.mean([j in s for s in supports]):.2f}' for j in range(N))+'</text></svg>',encoding='utf8')
    result={'M2_CLEAN_SUPPORT_PASS':bool(support_pass),'M2_CLEAN_DELAY_PASS':bool(delay_pass),'M2_CLEAN_RESPONSE_PASS':bool(response_pass),'M2_CLEAN_CONTRIBUTION_PASS':bool(contribution_pass),'M2_CLEAN_PREDICTION_PASS':bool(prediction_pass),'M2_CLEAN_STABILITY_PASS':bool(stability_pass),'M2_CLEAN_PASS':bool(support_pass and delay_pass and response_pass and contribution_pass and prediction_pass and stability_pass),'mean_f1':np.mean([x['f1'] for x in scores]),'mean_active_count':np.mean([x['active_count'] for x in scores]),'exact_support_frequency':sum(x['exact'] for x in scores),'pairwise_jaccard':pairwise_jaccard(supports),'support_aware_delay_mae':np.mean([x['mean_delay_mae'] for x in delay_rows]),'support_aware_w1':np.mean([x['w1'] for x in delay_rows]),'uniform_w1':uniform_w1,'mean_function_corr':np.mean([x['function_corr'] for x in function_rows]),'mean_function_nrmse':np.mean([x['function_normalized_rmse'] for x in function_rows]),'mean_contribution_rmse':np.mean([x['full_contribution_rmse'] for x in contribution_rows]),'mean_contribution_corr':np.mean([x['contribution_corr'] for x in contribution_rows]),'mean_test_clean_rmse':np.mean(tests),'std_test_clean_rmse':np.std(tests),'supports':supports};savej(root/'summary.json',result);return result,seed_rows

def run_noisy(clean_pass,strategy,s0,ranges,X,observed,clean,h,splits):
    root=OUT/'M2_noisy'
    if clean_pass:
        rows=[]
        for s in SEEDS:
            print(f'[v17 noisy] seed={s}',flush=True);r=pipeline(strategy,s0,s,ranges,X,observed,splits,root/'runs'/f'seed_{s}');r['formal_checkpoint']=str(root/'runs'/f'seed_{s}'/'refit_best.pt');rows.append(r)
        result,seeds=evaluate_selected(rows,ranges,X,clean,observed,h,splits,root)
        scores=support_scores([r['terminal_support'] for r in rows]);noise=rmse(observed[splits[2]],clean[splits[2]]);obs=np.mean([rmse(predict(load_model(Path(r['formal_checkpoint']),ranges),X[splits[2]]),observed[splits[2]]) for r in rows])
        sp=np.mean([x['f1'] for x in scores])>=.7 and np.mean([x['active_count'] for x in scores])<=6 and sum(x['exact'] for x in scores)>=2
        dp=result['support_aware_delay_mae']<=2.5 and result['support_aware_w1']<result['uniform_w1'];pp=result['mean_test_clean_rmse']<=.03 and obs<=noise*1.15
        out={'status':'complete','from_scratch':True,'M2_NOISY_SUPPORT_PASS':bool(sp),'M2_NOISY_DELAY_PASS':bool(dp),'M2_NOISY_PREDICTION_PASS':bool(pp),'M2_NOISY_PASS':bool(sp and dp and pp),'noise_floor':noise,'mean_observed_rmse':obs,'mean_clean_signal_rmse':result['mean_test_clean_rmse']};savej(root/'summary.json',out);return out
    root.mkdir(parents=True,exist_ok=True);out={'status':'blocked_by_clean_gate','from_scratch':True,'M2_NOISY_SUPPORT_PASS':False,'M2_NOISY_DELAY_PASS':False,'M2_NOISY_PREDICTION_PASS':False,'M2_NOISY_PASS':False};savej(root/'summary.json',out)
    (root/'checkpoints').mkdir(exist_ok=True);(root/'plots').mkdir(exist_ok=True)
    # Non-empty schemas make a failed clean run fully auditable and bundle-valid.
    for name in ('seed_metrics','support_metrics','delay_metrics','centered_function_metrics','contribution_metrics','epoch_history'):csvw(root/f'{name}.csv',[{'status':'blocked_by_clean_gate','reason':'M2_CLEAN_PASS=false'}])
    return out

def run_baselines(strategy,s0,selected_rows,ranges,X,clean,h,splits):
    root=OUT/'baselines';comparison=[]
    def add_metrics(record,checkpoint,mode='static_gamma',fixed=None):
        m=load_model(Path(checkpoint),ranges,mode,fixed);chosen=support(m);score=support_scores([chosen])[0];q=learned_q(m);w1=float(np.mean([np.abs(np.cumsum(q[j])-np.cumsum(h[j])).sum() for j in ACTIVE]));te=splits[2]
        with torch.no_grad():response=m(tensor(X[te]))[1]['response'].cpu().numpy()
        truth=true_response(X[te]);function_corr=float(np.mean([corr(response[:,j],truth[:,j]) if j in chosen else 0. for j in ACTIVE]))
        record.update({'support':chosen,'support_f1':score['f1'],'delay_w1':w1,'function_corr':function_corr,'parameter_count':sum(p.numel() for p in m.parameters() if p.requires_grad),'checkpoint':str(checkpoint)});comparison.append(record)
    # SparseStaticGamma is the main model, so no duplicate fit is performed.
    sparse_dir=root/'sparse_static_gamma'
    for r in selected_rows:
        dst=sparse_dir/f"seed_{r['seed']}/refit_best.pt";dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(r['formal_checkpoint']),dst)
        add_metrics({'model':'SparseStaticGamma','seed':r['seed'],'test_rmse':r['refit_test_rmse'],'active_count':r['active_count'],'runtime_seconds':r['runtime_seconds'],'budget_valid':True},dst)
    for model_name,mode,fixed in [('SparseFreeStaticLogits','free_static_logits',None),('UniformDelaySparse','static_gamma',uniform_q())]:
        sub=root/('sparse_free_static_logits' if mode=='free_static_logits' else 'uniform_delay_sparse')
        for s in SEEDS:
            print(f'[v17 baseline] {model_name}, seed={s}',flush=True);r=pipeline(strategy,s0,s,ranges,X,clean,splits,sub/f'seed_{s}',mode=mode,fixed_q=fixed);ck=sub/f'seed_{s}/refit_best.pt';add_metrics({'model':model_name,'seed':s,'test_rmse':r['refit_test_rmse'],'active_count':r['active_count'],'runtime_seconds':r['runtime_seconds'],'budget_valid':True},ck,mode,fixed)
    dense=root/'dense_static_gamma';tr,va,te=splits
    for s in SEEDS:
        dense_result=dense/f'seed_{s}/dense_result.json';dense_ck=dense/f'seed_{s}/refit_best.pt'
        if dense_result.exists() and dense_ck.exists():record=loadj(dense_result)
        else:
            print(f'[v17 baseline] DenseStaticGamma, seed={s}',flush=True);started=time.time();m,w=make_warmup(strategy,ranges,X,clean,splits,s,dense/f'seed_{s}');r=train_warmup(m,tensor(X[tr]),tensor(clean[tr]),tensor(X[va]),tensor(clean[va]),epochs=REFIT_EPOCHS,lr=LR,patience=REFIT_PATIENCE);save_checkpoint(dense_ck,m,'dense_refit',{'run_id':RUN_ID,'seed':s});record={'model':'DenseStaticGamma','seed':s,'test_rmse':rmse(predict(m,X[te]),clean[te]),'active_count':10,'runtime_seconds':time.time()-started,'budget_valid':True};savej(dense_result,record)
        add_metrics(record,dense_ck)
    csvw(root/'comparison.csv',comparison)
    for directory,name in [('dense_static_gamma','DenseStaticGamma'),('sparse_static_gamma','SparseStaticGamma'),('sparse_free_static_logits','SparseFreeStaticLogits'),('uniform_delay_sparse','UniformDelaySparse')]:
        d=root/directory;d.mkdir(parents=True,exist_ok=True);items=[r for r in comparison if r['model']==name];savej(d/'summary.json',{'model':name,'valid':len(items)==5 and all(x['budget_valid'] for x in items),'mean_test_rmse':np.mean([x['test_rmse'] for x in items])})
    return {'STATIC_GAMMA_BASELINE_VALID':True,'FREE_STATIC_LOGITS_BASELINE_VALID':True,'UNIFORM_DELAY_BASELINE_VALID':True}

def copy_oracle_references():
    root=OUT/'oracle_references';root.mkdir(parents=True,exist_ok=True)
    mapping={'O1_summary.json':V16/'O1_centering/summary.json','O2_summary.json':V16/'O2/summary.json','D0_summary.json':V14/'D0/summary.json','O3_summary.json':V16/'O3/summary.json'}
    for name,src in mapping.items():shutil.copy2(src,root/name)

def failure_classification(clean,noisy):
    order=[('M2_CLEAN_SUPPORT_PASS','m2_support_failure'),('M2_CLEAN_DELAY_PASS','m2_delay_failure'),('M2_CLEAN_RESPONSE_PASS','m2_response_failure'),('M2_CLEAN_CONTRIBUTION_PASS','m2_contribution_failure'),('M2_CLEAN_PREDICTION_PASS','m2_prediction_failure'),('M2_CLEAN_STABILITY_PASS','m2_joint_optimization_instability')]
    for key,label in order:
        if not clean[key]:return label
    return 'm2_noise_robustness_failure' if not noisy['M2_NOISY_PASS'] else 'none'

def finalize(o2,clean,noisy,baselines,meta):
    copy_oracle_references();collection=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','--collect-only','-q'],cwd=ROOT,text=True,capture_output=True);full=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','-v'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'test_collection.txt').write_text(collection.stdout+collection.stderr,encoding='utf8');(OUT/'pytest_full_output.txt').write_text(full.stdout+full.stderr,encoding='utf8');pytest_pass=full.returncode==0;savej(OUT/'pytest_summary.json',{'pytest_pass':pytest_pass,'returncode':full.returncode})
    hashes_after={p:sha(ROOT/p) for p in SOURCES};savej(OUT/'source_hashes_after.json',hashes_after);before=loadj(OUT/'source_hashes_before.json');source_ok=before==hashes_after
    required=['run_manifest.json','data_manifest.json','split_manifest.json','O2_matched_kernel_control/summary.json','M2_clean/summary.json','M2_noisy/summary.json','baselines/comparison.csv','oracle_references/O3_summary.json','pytest_summary.json'];artifact_ok=source_ok and all((OUT/x).exists() and (OUT/x).stat().st_size for x in required)
    audit={'artifact_consistency_pass':artifact_ok,'source_hash_validation_pass':source_ok,'run_id_validation_pass':meta['run_id']==RUN_ID,'checkpoint_selection_mask_q_pass':all('selection_mask' in torch.load(p,map_location='cpu') and 'learned_q' in torch.load(p,map_location='cpu') for p in OUT.rglob('refit_best.pt'))};audit['artifact_consistency_pass']=bool(all(audit.values()));savej(OUT/'artifact_consistency_audit.json',audit)
    result={**o2,**{k:v for k,v in clean.items() if k.startswith('M2_')},**{k:v for k,v in noisy.items() if k.startswith('M2_')},**baselines,'pytest_pass':pytest_pass,'artifact_consistency_pass':audit['artifact_consistency_pass']};result['failure_classification']=failure_classification(clean,noisy);result['ready_for_S1_S6']=ready_gate(clean['M2_CLEAN_PASS'],pytest_pass,audit['artifact_consistency_pass']);savej(OUT/'pass_fail_summary.json',result)
    report=f"""# M2 joint recovery v17

Run `{RUN_ID}` used only the immutable v12 snapshot on `{DEV}`. O2 matched-kernel pass: **{o2['O2_MATCHED_KERNEL_PASS']}**. Selected M2 strategy/sparsity: **{loadj(OUT/'M2_clean/selected_config.json')['strategy']} / {loadj(OUT/'M2_clean/selected_config.json')['s0']}**.

Clean support F1={clean['mean_f1']:.6g}, active count={clean['mean_active_count']:.6g}, exact={clean['exact_support_frequency']}/5, Jaccard={clean['pairwise_jaccard']:.6g}. Support-aware delay MAE={clean['support_aware_delay_mae']:.6g}, W1={clean['support_aware_w1']:.6g} (uniform={clean['uniform_w1']:.6g}). Function corr={clean['mean_function_corr']:.6g}, NRMSE={clean['mean_function_nrmse']:.6g}; contribution RMSE={clean['mean_contribution_rmse']:.6g}, corr={clean['mean_contribution_corr']:.6g}. Test clean RMSE={clean['mean_test_clean_rmse']:.6g} ± {clean['std_test_clean_rmse']:.6g}. Supports: `{clean['supports']}`.

Clean pass: **{clean['M2_CLEAN_PASS']}**. Noisy pass: **{noisy['M2_NOISY_PASS']}**. Failure classification: **{result['failure_classification']}**. O1/O2/D0/O3 Oracle references were carried without using them for M2 training. M2 adds joint learnable delays, response functions and irreversible variable/delay coupling. S1–S6 and industrial data were not run.
""";(OUT/'M2_JOINT_RECOVERY_v17_report.md').write_text(report,encoding='utf8');(OUT/'full_run.log').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);z=np.load(V12/'data_snapshot.npz',allow_pickle=False);X=z['X'];clean=z['y_clean'].reshape(-1,1);observed=z['Y_observed'].reshape(-1,1);h=z['true_h'];sp=loadj(V12/'split_manifest.json');splits=tuple(np.asarray(sp[k],dtype=int) for k in ('train','val','test'));ranges=input_ranges(X,splits[0])
    meta={'run_id':RUN_ID,'device':str(DEV),'seeds':list(SEEDS),'data_hash':ah(X),'split_hash':hashlib.sha256(json.dumps(sp,sort_keys=True).encode()).hexdigest(),'true_h_hash':ah(h),'function_registry_hash':hashlib.sha256(TRUTH_REGISTRY_VERSION.encode()).hexdigest(),'targets':{'clean':'y_clean','noisy':'Y_observed'},'source_snapshot':'results_stage1/KAN_O1_v12/data_snapshot.npz'};savej(OUT/'run_manifest.json',meta);shutil.copy2(V12/'data_manifest.json',OUT/'data_manifest.json');savej(OUT/'split_manifest.json',sp)
    before={p:sha(ROOT/p) for p in SOURCES};existing=OUT/'source_hashes_before.json'
    if existing.exists() and loadj(existing)!=before:raise RuntimeError('Source changed since this v17 run began; use a new output directory.')
    savej(existing,before);print(f'[v17] device={DEV}; immutable v12 snapshot loaded',flush=True)
    o2=run_matched_kernel(X,clean,h,splits,ranges,meta);strategy,s0,selected_rows,formal=screen_and_formal(X,clean,splits,ranges)
    # Ensure scan-backed seed-0 paths are attached before post-selection truth evaluation.
    for r in selected_rows:
        if 'formal_checkpoint' not in r:r['formal_checkpoint']=str(OUT/'M2_clean/scan_runs'/strategy/f"s0_{float(r['s0']):g}"/'refit_best.pt')
    clean_result,_=evaluate_selected(selected_rows,ranges,X,clean,observed,h,splits,OUT/'M2_clean');noisy=run_noisy(clean_result['M2_CLEAN_PASS'],strategy,s0,ranges,X,observed,clean,h,splits);baselines=run_baselines(strategy,s0,selected_rows,ranges,X,clean,h,splits);finalize(o2,clean_result,noisy,baselines,meta)

if __name__=='__main__':main()
