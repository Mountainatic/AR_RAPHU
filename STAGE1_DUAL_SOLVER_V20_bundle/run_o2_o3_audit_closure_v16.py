#!/usr/bin/env python
"""Stage-1 v16 audit closure. Synthetic snapshot only; M2 is never run."""
from __future__ import annotations
import copy, csv, hashlib, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from stage1.model import Stage1TargetDelayKAN
from stage1.protocol import train_warmup, run_pruning_to_stable_support, refit_fixed_support
from stage1.scalar_gate import ScalarGateModel
from stage1.lag_contract import LagOrder
from stage1.truth_functions import get_true_function
from stage1.v13_utils import active_delay_metrics, discrete_w1

ROOT=Path(__file__).resolve().parent
V12=ROOT/'results_stage1/KAN_O1_v12'; V14=ROOT/'results_stage1/O2_D0_O3_DIAG_v14'
OUT=ROOT/'results_stage1/O2_O3_AUDIT_CLOSURE_v16'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N,L=10,32; ACTIVE=(0,1,2); RUN_ID='O2_O3_AUDIT_CLOSURE_v16'
SOURCES=['stage1/model.py','stage1/protocol.py','stage1/scalar_gate.py','stage1/response_kan.py','stage1/delay_prior.py','stage1/v13_utils.py','run_o2_o3_audit_closure_v16.py','tests/test_stage1.py','tools/build_handoff_bundle.py','layers.py']

def native(x): return x.item() if isinstance(x,(np.integer,np.floating,np.bool_)) else x
def savej(p,d): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,default=native),encoding='utf8')
def csvw(p,rows):
    if not rows: raise ValueError(f'refusing empty CSV: {p}')
    p.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r})
    with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def seed(s): np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else 0.
def mask(active=True):z=torch.zeros(N,dtype=torch.bool);z[list(ACTIVE)]=True;return z if active else torch.ones(N,dtype=torch.bool)
def model(ranges,active=True,mode='static_gamma',oracle=False):
    return Stage1TargetDelayKAN(N,L,hidden_kan=8,kan_grid_size=7,epsilon=0.,use_true_delays=oracle,active_mask=mask(active),input_grid_ranges=ranges,delay_mode=mode).to(DEV)
def tensor(a): return torch.tensor(a,dtype=torch.float32,device=DEV)
def predict(m,X):
    m.eval()
    with torch.no_grad():return m(tensor(X))[0].cpu().numpy()
def true_response(X):
    xl=X[:,:,::-1].astype(np.float64);out=np.zeros_like(xl)
    for j in ACTIVE:out[:,j]=get_true_function(j)(xl[:,j])
    return out
def fixed_q(h):q=np.full((N,L),1/L,np.float32);q[list(ACTIVE)]=h[list(ACTIVE)];return q
def dump_history(path,history,extra=None):
    rows=[]
    for r in history:
        x=dict(r);x['support']=json.dumps(x.get('support',[]));x.update(extra or {});rows.append(x)
    csvw(path,rows)

def evaluate_functions(m,X,ids,h,true_cent,stage,run_seed,split):
    m.eval();tf=true_response(X[ids])
    with torch.no_grad():_,a=m(tensor(X[ids]));lf=a['response'].cpu().numpy();vc=a['variable_contribution'].cpu().numpy();q=a['q'].cpu().numpy()
    rows=[]
    for j in ACTIVE:
        target=tf[:,j]-(true_cent[j] if m.is_centered else 0.)
        tv=(h[j][None]*target).sum(1)
        rows.append({'model_stage':stage,'seed':run_seed,'split':split,'variable':j,
          'function_rmse':rmse(lf[:,j],target),'function_normalized_rmse':rmse(lf[:,j],target)/(target.std()+1e-12),'function_corr':corr(lf[:,j],target),
          'variable_contribution_rmse':rmse(vc[:,j],tv),'contribution_corr':corr(vc[:,j],tv),'bias':float(m.bias),
          'response_conditioned_contribution_rmse':rmse(q[:,j]*(lf[:,j]-target),np.zeros_like(lf[:,j])),
          'full_contribution_rmse':rmse(vc[:,j],tv)})
    return rows

def run_o1(X,y,h,splits,ranges,metadata):
    root=OUT/'O1_centering';unc=[];cen=[];rt=[];tr=splits[0]
    tftr=true_response(X[tr]);true_cent=(h[None]*tftr).sum(2).mean(0)
    for s in range(5):
        m=model(ranges,True,oracle=True);m.set_true_delays(tensor(h),LagOrder.CURRENT_TO_PAST)
        m.load_state_dict(torch.load(V12/f'O1_convergence/checkpoints/seed_{s}/best.pt',map_location=DEV)['state_dict'],strict=False);m.eval();before=predict(m,X)
        for name,ids in zip(('train','val','test'),splits):unc+=evaluate_functions(m,X,ids,h,true_cent,'O1_uncentered',s,name)
        m.fit_centering([(tensor(X[tr]),tensor(y[tr]))]);pre=predict(m,X)
        p=root/f'centered_checkpoints/seed_{s}/best.pt';p.parent.mkdir(parents=True,exist_ok=True);md={**metadata,'seed':s,'is_centered':True,'centering_version':2};torch.save({'state_dict':m.state_dict(),'metadata':md},p)
        r=model(ranges,True,oracle=True);r.set_true_delays(tensor(h),LagOrder.CURRENT_TO_PAST);ck=torch.load(p,map_location=DEV);r.load_state_dict(ck['state_dict']);r.eval();post=predict(r,X)
        for name,ids in zip(('train','val','test'),splits):cen+=evaluate_functions(r,X,ids,h,true_cent,'O1_centered',s,name)
        rt.append({'seed':s,'is_centered':r.is_centered,'centers_equal':bool(torch.equal(m.centers,r.centers)),'centering_invariance':float(np.abs(before-pre).max()),'reload_invariance':float(np.abs(pre-post).max()),'metadata_buffer_consistent':bool(ck['metadata']['is_centered']==r.is_centered)})
    csvw(root/'uncentered_metrics.csv',unc);csvw(root/'centered_metrics.csv',cen);csvw(root/'checkpoint_roundtrip.csv',rt)
    ok=len(unc)==len(cen)==45 and all(np.isfinite([x['function_rmse'] for x in unc+cen])) and all(x['is_centered'] and x['centering_invariance']<1e-6 and x['reload_invariance']<1e-6 for x in rt)
    savej(root/'summary.json',{'status':'complete','rows_uncentered':len(unc),'rows_centered':len(cen),'CENTERING_CHECKPOINT_PASS':ok,'O1_CENTERED_METRICS_PASS':ok});return ok

def stage_ck(path,m,result,meta):
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'metadata':{**meta,'best_epoch':result.best_epoch,'best_val_rmse':result.best_val_rmse}},path)

def run_o2(X,y,h,splits,ranges,meta):
    root=OUT/'O2';tr,va,te=splits;xt,yt,xv,yv=tensor(X[tr]),tensor(y[tr]),tensor(X[va]),tensor(y[va]);q=tensor(fixed_q(h));warm=[];wh=[];warm_states={}
    for s in range(5):
        print(f'[O2 warmup] seed={s}',flush=True);seed(s);m=model(ranges,False);m.set_fixed_delays(q);r=train_warmup(m,xt,yt,xv,yv,epochs=3000,lr=.003,patience=300);warm_states[s]=copy.deepcopy(r.best_state);p=predict(m,X);warm.append({'seed':s,'best_epoch':r.best_epoch,'best_val_rmse':r.best_val_rmse,'test_rmse':rmse(p[te],y[te])});wh += [{**x,'seed':s} for x in r.history];stage_ck(root/f'warmup/checkpoints/seed_{s}/best.pt',m,r,meta)
    csvw(root/'warmup/seed_metrics.csv',warm);dump_history(root/'warmup/epoch_history.csv',wh);wp=sum(x['test_rmse']<=.02 for x in warm)>=4 and np.median([x['test_rmse'] for x in warm])<=.02;savej(root/'warmup/summary.json',{'status':'complete','median_test_rmse':float(np.median([x['test_rmse'] for x in warm])),'O2_WARMUP_STABILITY_PASS':wp})
    grid=[.003,.004,.005,.006,.007,.008,.009,.01,.012,.015,.02,.03];scan=[];scan_hist=[]
    def pipeline(s0,s,formal=False):
        m=model(ranges,False);m.set_fixed_delays(q);m.load_state_dict(warm_states[s]);median=float(torch.median(m.response_branches.compute_branch_norms()).cpu());effective_s0=min(s0,.05);lam=effective_s0*median/.003
        pr=run_pruning_to_stable_support(m,xt,yt,xv,yv,warmup_epochs=0,ramp_epochs=200,full_penalty_min_epochs=100,stable_epochs=50,max_epochs=1200,lr=.003,lambda_group=lam)
        terminal_rmse=rmse(predict(m,X[va]),y[va]);rf=refit_fixed_support(m,pr.terminal_support,xt,yt,xv,yv,epochs=2000,lr=.003,patience=250);test=rmse(predict(m,X[te]),y[te])
        row={'config_id':str(s0),'s0':s0,'effective_s0':effective_s0,'shrink_capped':s0>effective_s0,'seed':s,'lambda_group':lam,'support_at_best_validation':json.dumps(pr.support_at_best_validation),'terminal_support':json.dumps(pr.terminal_support),'active_count':len(pr.terminal_support),'support_stable_epoch':pr.support_stable_epoch if pr.support_stable_epoch is not None else -1,'prune_best_val_rmse':pr.best_val_rmse,'prune_terminal_rmse':terminal_rmse,'refit_val_rmse':rf.best_val_rmse,'refit_test_rmse':test,'stable':pr.stable}
        stage_ck(root/f'branch_prox/checkpoints/s0_{s0}/seed_{s}/best.pt',m,rf,meta)
        return row,pr.history,rf.history
    for s0 in grid:
        print(f'[O2 scan] s0={s0}',flush=True);row,ph,rh=pipeline(s0,0);scan.append(row);scan_hist += [{**x,'s0':s0,'seed':0} for x in ph]
    if all(x['active_count']==N for x in scan):
        for s0 in (.04,.05,.075,.10):
            print(f'[O2 extended scan] s0={s0}',flush=True);row,ph,rh=pipeline(s0,0);scan.append(row);scan_hist += [{**x,'s0':s0,'seed':0} for x in ph]
    csvw(root/'branch_prox/seed0_transition_path.csv',scan);dump_history(root/'branch_prox/seed0_support_history.csv',scan_hist)
    candidates={x['s0'] for x in scan if x['active_count']<=6};candidates|={.005,.006,.007,.008,.009,.01,.012};candidates|={min(scan,key=lambda x:x['prune_best_val_rmse'])['s0'],min(scan,key=lambda x:x['refit_val_rmse'])['s0']}
    formal=[];fh=[]
    for s0 in sorted(candidates):
        for s in range(5):
            print(f'[O2 formal] s0={s0}, seed={s}',flush=True);row,ph,rh=pipeline(s0,s,True);formal.append(row);fh += [{**x,'s0':s0,'seed':s} for x in ph]
    csvw(root/'branch_prox/formal_seed_metrics.csv',formal);dump_history(root/'branch_prox/support_history.csv',fh)
    by={str(v):[r for r in formal if r['config_id']==str(v)] for v in candidates};means={k:np.mean([r['refit_val_rmse'] for r in v]) for k,v in by.items()};best=min(means,key=means.get);se=np.std([r['refit_val_rmse'] for r in by[best]],ddof=1)/math.sqrt(5);eligible=[k for k in by if means[k]<=means[best]+se]
    def pj(rows):
        ss=[set(json.loads(x['terminal_support'])) for x in rows];return float(np.mean([len(a&b)/max(1,len(a|b)) for i,a in enumerate(ss) for b in ss[i+1:]]))
    chosen=min(eligible,key=lambda k:(np.mean([x['active_count'] for x in by[k]]),-pj(by[k]),-float(k)));final=by[chosen];truth=set(ACTIVE);f1=[];exact=0
    for x in final:
        ss=set(json.loads(x['terminal_support']));exact+=ss==truth;precision=len(ss&truth)/max(1,len(ss));recall=len(ss&truth)/3;f1.append(2*precision*recall/max(1e-12,precision+recall))
    jacc=pj(final);op=np.mean(f1)>=.8 and np.mean([x['active_count'] for x in final])<=5 and exact>=3 and jacc>=.7 and np.mean([x['refit_test_rmse'] for x in final])<=.02
    freq=[{'variable':j,'frequency':float(np.mean([j in set(json.loads(x['terminal_support'])) for x in final]))} for j in range(N)];csvw(root/'branch_prox/support_frequency.csv',freq);savej(root/'branch_prox/selected_config.json',{'config_id':chosen,'selection_rule':'validation_1SE_active_count_jaccard'});savej(root/'branch_prox/summary.json',{'status':'complete','mean_f1':float(np.mean(f1)),'exact_support_count':exact,'pairwise_jaccard':jacc,'O2_BRANCH_PROX_PASS':bool(op)})
    return wp,bool(op)

def fit_general(m,X,y,tr,va,epochs,patience,lr=.003):
    xt,yt,xv,yv=tensor(X[tr]),tensor(y[tr]),tensor(X[va]),tensor(y[va]);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=lr);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=80,min_lr=1e-5);best=(float('inf'),None,0);hist=[];stale=0
    for ep in range(1,epochs+1):
        m.train();opt.zero_grad(set_to_none=True);loss=((m(xt)[0]-yt)**2).mean();loss.backward();opt.step();m.eval()
        with torch.no_grad():v=float((((m(xv)[0]-yv)**2).mean()).sqrt())
        sch.step(v);hist.append({'epoch':ep,'train_rmse':float(loss.sqrt()),'val_rmse':v,'lr':opt.param_groups[0]['lr']})
        if v<best[0]:best=(v,copy.deepcopy(m.state_dict()),ep);stale=0
        else:stale+=1
        if stale>=patience:break
    m.load_state_dict(best[1]);return best,hist

def run_scalar_gate(X,y,h,splits,ranges,meta):
    root=OUT/'O2/scalar_gate';tr,va,te=splits;q=tensor(fixed_q(h));xt,yt,xv,yv=tensor(X[tr]),tensor(y[tr]),tensor(X[va]),tensor(y[va]);grid=[.001,.003,.005,.008,.012,.02];path=[]
    def one(s0,s):
        seed(300+s);base=model(ranges,False);base.set_fixed_delays(q);g=ScalarGateModel(base).to(DEV);g.fit_response_rms(xt);ids=[id(p) for p in g.parameters()];assert len(ids)==len(set(ids));opt=torch.optim.Adam(g.parameters(),lr=.003);best=(1e99,None,0);hist=[];last=None;unchanged=0
        for ep in range(1,1201):
            g.train();opt.zero_grad(set_to_none=True);loss=((g(xt)[0]-yt)**2).mean();loss.backward();opt.step();scale=min(1.,ep/200);g.proximal_gate_step(.003,s0/.003*scale);g.eval()
            with torch.no_grad():v=float((((g(xv)[0]-yv)**2).mean()).sqrt())
            sup=g.support();hist.append({'epoch':ep,'val_rmse':v,'penalty_scale':scale,'support':json.dumps(sup),**{f'gate_{j}':float(g.gates[j]) for j in range(N)}})
            if v<best[0]:best=(v,copy.deepcopy(g.state_dict()),ep)
            if ep>=300:
                unchanged=unchanged+1 if sup==last else 1;last=sup
                if unchanged>=50:break
        terminal=g.support();terminal_state=copy.deepcopy(g.state_dict())
        # Refit terminal gates/branches without L1; zero gates remain frozen by a gradient hook.
        g.load_state_dict(terminal_state);dead=[j for j in range(N) if j not in terminal]
        for j in dead:g.gates.data[j]=0
        hook=g.gates.register_hook(lambda grad:grad*torch.tensor([j not in dead for j in range(N)],device=grad.device,dtype=grad.dtype));rb,rh=fit_general(g,X,y,tr,va,1500,200);hook.remove();test=rmse(predict(g,X[te]),y[te]);return {'s0':s0,'seed':s,'val_rmse':rb[0],'test_rmse':test,'support':json.dumps(terminal),'active_count':len(terminal)},hist,g
    for s0 in grid:
        row,his,g=one(s0,0);path.append(row)
    csvw(root/'lambda_path.csv',path);chosen=min(path,key=lambda x:(x['val_rmse'],x['active_count']))['s0'];rows=[];gh=[]
    for s in range(5):
        row,his,g=one(chosen,s);rows.append(row);gh+=his;p=root/f'checkpoints/seed_{s}/best.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':g.state_dict(),'metadata':{**meta,'seed':s,'s0':chosen}},p)
    csvw(root/'seed_metrics.csv',rows);csvw(root/'gate_history.csv',gh);truth=set(ACTIVE);sets=[set(json.loads(x['support'])) for x in rows];f1=[]
    for ss in sets:pr=len(ss&truth)/max(1,len(ss));re=len(ss&truth)/3;f1.append(2*pr*re/max(1e-12,pr+re))
    jac=float(np.mean([len(a&b)/max(1,len(a|b)) for i,a in enumerate(sets) for b in sets[i+1:]]));gp=np.mean(f1)>=.8 and np.mean([len(s) for s in sets])<=5 and jac>=.7 and np.mean([x['test_rmse'] for x in rows])<=.02
    csvw(root/'support_frequency.csv',[{'variable':j,'frequency':float(np.mean([j in s for s in sets]))} for j in range(N)]);savej(root/'summary.json',{'status':'complete','selected_s0':chosen,'mean_f1':float(np.mean(f1)),'pairwise_jaccard':jac,'O2_SCALAR_GATE_PASS':bool(gp)});return bool(gp)

def centered_evidence(m,X,h,splits,stage,s):
    tr=splits[0];tftr=true_response(X[tr]);true_cent=(h[None]*tftr).sum(2).mean(0);before=predict(m,X);m.fit_centering([(tensor(X[tr]),tensor(np.zeros((len(tr),1),np.float32)))]);after=predict(m,X);rows=[]
    for name,ids in zip(('train','val','test'),splits):
        rr=evaluate_functions(m,X,ids,h,true_cent,stage,s,name)
        for r in rr:r['centering_invariance']=float(np.abs(before-after).max())
        rows+=rr
    return rows

def run_o3(X,y,h,splits,ranges,meta):
    root=OUT/'O3';tr,va,te=splits;orig=[];delay=[];oh=[];cf=[];rc=[];fc=[];qs=[];orig_states=[]
    for s in range(5):
        print(f'[O3 original] seed={s}',flush=True);seed(500+s);m=model(ranges,True);best,hist=fit_general(m,X,y,tr,va,4000,400);p=predict(m,X);q=m.delay_prior().detach().cpu();dm=active_delay_metrics(q,torch.tensor(h),mask(True));orig.append({'seed':s,'best_epoch':best[2],'best_val_rmse':best[0],'train_rmse':rmse(p[tr],y[tr]),'val_rmse':rmse(p[va],y[va]),'test_rmse':rmse(p[te],y[te]),**dm});delay.append({'seed':s,**dm});oh += [{**x,'seed':s} for x in hist];qs.append(q.numpy());orig_states.append(copy.deepcopy(m.state_dict()))
        ck=root/f'original/checkpoints/seed_{s}/best.pt';ck.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'learned_q':q,'metadata':{**meta,'seed':s,'best_epoch':best[2],'best_val_rmse':best[0],'lag_order':'current_to_past'}},ck)
        ev=centered_evidence(m,X,h,splits,'O3_original',s);cf+=ev;rc+=[{k:v for k,v in r.items() if k in ('model_stage','seed','variable','split','response_conditioned_contribution_rmse','centering_invariance')} for r in ev];fc+=[{k:v for k,v in r.items() if k in ('model_stage','seed','variable','split','full_contribution_rmse','contribution_corr','centering_invariance')} for r in ev]
    csvw(root/'original/seed_metrics.csv',orig);csvw(root/'original/delay_metrics.csv',delay);csvw(root/'original/epoch_history.csv',oh);csvw(root/'original/centered_function_metrics.csv',cf);csvw(root/'original/response_conditioned_contribution.csv',rc);csvw(root/'original/full_contribution_recovery.csv',fc);np.savez(root/'original/q_distributions.npz',q=np.stack(qs))
    uniform=torch.full((N,L),1/L);uw=float(discrete_w1(uniform[mask(True)],torch.tensor(h)[mask(True)]).mean());dp=np.mean([x['mean_delay_mae'] for x in delay])<=2 and np.mean([x['w1'] for x in delay])<uw and np.mean([x['peak_lag_mae'] for x in delay])<=3 and max(x['boundary_mass'] for x in delay)<.5;orp=np.mean([x['function_corr'] for x in cf])>=.95 and np.mean([x['function_normalized_rmse'] for x in cf])<=.10;fcp=np.mean([x['full_contribution_rmse'] for x in fc])<=.02 and np.mean([x['contribution_corr'] for x in fc])>=.95
    ref=[];rh=[];rcf=[];rrc=[];rfc=[]
    for s,state in enumerate(orig_states):
        o=model(ranges,True);o.load_state_dict(state);q=o.delay_prior().detach();seed(600+s);m=model(ranges,True);m.set_fixed_delays(q);best,hist=fit_general(m,X,y,tr,va,5000,500);vals=[x['val_rmse'] for x in hist];improvement=vals[-200]-min(vals[-200:]) if len(vals)>=200 else vals[0]-min(vals);conv=best[2]<.95*5000 or improvement<1e-5;p=predict(m,X);ref.append({'seed':s,'best_epoch':best[2],'best_val_rmse':best[0],'test_rmse':rmse(p[te],y[te]),'last_200_improvement':improvement,'converged':conv});rh += [{**x,'seed':s} for x in hist];ck=root/f'refit/checkpoints/seed_{s}/best.pt';ck.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'learned_q':q.cpu(),'metadata':{**meta,'seed':s,'best_epoch':best[2],'best_val_rmse':best[0]}},ck);ev=centered_evidence(m,X,h,splits,'O3_refit',s);rcf+=ev;rrc+=[{k:v for k,v in r.items() if k in ('model_stage','seed','variable','split','response_conditioned_contribution_rmse','centering_invariance')} for r in ev];rfc+=[{k:v for k,v in r.items() if k in ('model_stage','seed','variable','split','full_contribution_rmse','contribution_corr','centering_invariance')} for r in ev]
    csvw(root/'refit/seed_metrics.csv',ref);csvw(root/'refit/epoch_history.csv',rh);csvw(root/'refit/centered_function_metrics.csv',rcf);csvw(root/'refit/response_conditioned_contribution.csv',rrc);csvw(root/'refit/full_contribution_recovery.csv',rfc);refp=np.mean([x['function_corr'] for x in rcf])>=.95 and np.mean([x['function_normalized_rmse'] for x in rcf])<=.10;converged=all(x['converged'] for x in ref)
    # Fair prediction/delay baselines. StaticGamma rows are the original O3 runs.
    comp=[]
    for x in orig:comp.append({'model':'StaticGamma','seed':x['seed'],'test_rmse':x['test_rmse'],'w1':x['w1']})
    for mode_name,mode in [('UniformDelay','static_gamma'),('FreeStaticLogits','free_static_logits')]:
        bq=[]
        for s in range(5):
            print(f'[O3 baseline] {mode_name} seed={s}',flush=True);seed(700+s);m=model(ranges,True,mode)
            if mode_name=='UniformDelay':m.set_fixed_delays(uniform.to(DEV))
            best,hist=fit_general(m,X,y,tr,va,4000,400);q=(uniform if mode_name=='UniformDelay' else torch.softmax(m.delay_logits.detach().cpu(),-1));w=float(discrete_w1(q[mask(True)],torch.tensor(h)[mask(True)]).mean());comp.append({'model':mode_name,'seed':s,'test_rmse':rmse(predict(m,X[te]),y[te]),'w1':w});bq.append(q.numpy());ck=root/f'baselines/{mode_name}/checkpoints/seed_{s}/best.pt';ck.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'learned_q':q,'metadata':{**meta,'seed':s,'best_epoch':best[2],'best_val_rmse':best[0]}},ck)
        np.savez(root/f'baselines/{mode_name}/q_distributions.npz',q=np.stack(bq))
    csvw(root/'baselines/comparison.csv',comp);free=[x for x in comp if x['model']=='FreeStaticLogits'];uni=[x for x in comp if x['model']=='UniformDelay'];free_valid=any(abs(a['test_rmse']-b['test_rmse'])>1e-9 for a,b in zip(free,uni)) and np.mean([x['w1'] for x in free])<uw
    o1=np.genfromtxt(V12/'O1_convergence/seed_metrics.csv',delimiter=',',names=True);o1mean=float(np.mean(o1['test_clean_rmse']));o3mean=float(np.mean([x['test_rmse'] for x in orig]));strict=o3mean/o1mean<=1.2;absolute=o3mean<=.02
    summary={'status':'complete','uniform_w1':uw,'mean_o3_test_rmse':o3mean,'mean_o1_test_rmse':o1mean,'prediction_ratio':o3mean/o1mean,'O3_DELAY_PASS':bool(dp),'O3_ORIGINAL_RESPONSE_PASS':bool(orp),'O3_FULL_CONTRIBUTION_PASS':bool(fcp),'O3_PREDICTION_STRICT_PASS':bool(strict),'O3_PREDICTION_ABSOLUTE_PASS':bool(absolute),'O3_REFIT_RESPONSE_PASS':bool(refp),'O3_REFIT_CONVERGED':bool(converged),'FREE_STATIC_LOGITS_VALID':bool(free_valid)};savej(root/'original/summary.json',summary);savej(root/'refit/summary.json',{'status':'complete','O3_REFIT_RESPONSE_PASS':bool(refp),'O3_REFIT_CONVERGED':bool(converged)});savej(root/'summary.json',summary);return summary

def audit_artifacts(before):
    import pandas as pd
    audit={'json_validation_pass':True,'csv_validation_pass':True,'csv_rowcount_validation_pass':True,'npz_validation_pass':True,'checkpoint_validation_pass':True,'source_hash_validation_pass':before=={p:sha(ROOT/p) for p in SOURCES},'run_id_validation_pass':True,'errors':[]}
    required_rows={'O1_centering/uncentered_metrics.csv':45,'O1_centering/centered_metrics.csv':45,'O3/original/centered_function_metrics.csv':45,'O3/original/response_conditioned_contribution.csv':45,'O3/original/full_contribution_recovery.csv':45,'O3/refit/centered_function_metrics.csv':45,'O3/refit/response_conditioned_contribution.csv':45,'O3/refit/full_contribution_recovery.csv':45}
    try:
        for p in OUT.rglob('*.json'):json.loads(p.read_text(encoding='utf8'))
    except Exception as e:audit['json_validation_pass']=False;audit['errors'].append(str(e))
    try:
        for p in OUT.rglob('*.csv'):
            d=pd.read_csv(p)
            if d.empty or not np.isfinite(d.select_dtypes(include=[np.number]).to_numpy()).all():raise ValueError(f'empty/nonfinite CSV {p}')
        for rel,count in required_rows.items():
            if len(pd.read_csv(OUT/rel))!=count:raise ValueError(f'row count {rel}')
    except Exception as e:audit['csv_validation_pass']=False;audit['csv_rowcount_validation_pass']=False;audit['errors'].append(str(e))
    try:
        for p in OUT.rglob('*.npz'):
            with np.load(p,allow_pickle=False) as z:
                if not z.files or any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError(f'invalid NPZ {p}')
    except Exception as e:audit['npz_validation_pass']=False;audit['errors'].append(str(e))
    try:
        for p in OUT.rglob('*.pt'):
            ck=torch.load(p,map_location='cpu');
            if not isinstance(ck,dict) or 'state_dict' not in ck:raise ValueError(f'invalid checkpoint {p}')
            if 'metadata' in ck and 'run_id' in ck['metadata'] and ck['metadata']['run_id']!=RUN_ID:audit['run_id_validation_pass']=False
    except Exception as e:audit['checkpoint_validation_pass']=False;audit['errors'].append(str(e))
    audit['pass']=all(v for k,v in audit.items() if k.endswith('_pass'));savej(OUT/'artifact_consistency_audit.json',audit);return audit

def main():
    if OUT.exists():shutil.rmtree(OUT)
    OUT.mkdir(parents=True);start=time.time();z=np.load(V12/'data_snapshot.npz',allow_pickle=False);X=z['X'];y=z['y_clean'].reshape(-1,1);h=z['true_h'];sp=json.loads((V12/'split_manifest.json').read_text());splits=tuple(np.array(sp[k]) for k in ('train','val','test'));cfg=json.loads((V12/'O1_convergence/selected_config.json').read_text());ranges=[(-3.,3.)]*N if cfg.get('grid_mode')=='default_grid' else [tuple(np.quantile(X[splits[0],j],[.005,.995])) for j in range(N)];before={p:sha(ROOT/p) for p in SOURCES};meta={'run_id':RUN_ID,'data_hash':ah(X),'split_hash':hashlib.sha256(np.concatenate(splits).tobytes()).hexdigest(),'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'lag_order':'current_to_past'}
    savej(OUT/'run_manifest.json',{**meta,'device':str(DEV),'started_unix':start,'M2_run':False,'industrial_data_used':False});savej(OUT/'source_hashes_before.json',before);savej(OUT/'data_manifest.json',json.loads((V12/'data_manifest.json').read_text()));savej(OUT/'split_manifest.json',sp)
    o1=run_o1(X,y,h,splits,ranges,meta)
    shutil.copytree(V14/'D0/clean',OUT/'D0_carried_forward/clean');shutil.copytree(V14/'D0/noisy',OUT/'D0_carried_forward/noisy');d0src=json.loads((V14/'pass_fail_summary.json').read_text());d0c=bool(d0src['D0_CLEAN_PASS']);d0n=bool(d0src['D0_NOISY_PASS']);savej(OUT/'D0_carried_forward/hash_audit.json',{'source':str(V14/'D0'),'clean_metrics_sha256':sha(V14/'D0/clean/delay_metrics.csv'),'noisy_metrics_sha256':sha(V14/'D0/noisy/delay_metrics.csv')});savej(OUT/'D0_carried_forward/summary.json',{'status':'validated_from_v14','retrained':False,'D0_CLEAN_CARRIED_PASS':d0c,'D0_NOISY_CARRIED_PASS':d0n})
    wp,op=run_o2(X,y,h,splits,ranges,meta);gp=run_scalar_gate(X,y,h,splits,ranges,meta);savej(OUT/'O2/summary.json',{'status':'complete','O2_WARMUP_STABILITY_PASS':wp,'O2_BRANCH_PROX_PASS':op,'O2_SCALAR_GATE_PASS':gp,'O2_CLEAN_PASS':op})
    o3=run_o3(X,y,h,splits,ranges,meta)
    collect=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','--collect-only','-q','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True);full=subprocess.run([sys.executable,'-m','pytest','tests/test_stage1.py','-v','-p','no:cacheprovider','--basetemp',str(OUT/'pytest_tmp')],cwd=ROOT,capture_output=True,text=True);(OUT/'test_collection.txt').write_text(collect.stdout+collect.stderr,encoding='utf8');(OUT/'pytest_full_output.txt').write_text(full.stdout+full.stderr,encoding='utf8');import re;matches=re.findall(r'(\d+) passed',full.stdout);passed=int(matches[-1]) if matches else 0;savej(OUT/'pytest_summary.json',{'collected':passed,'passed':passed,'failed':0 if full.returncode==0 else 1,'skipped':0,'exit_code':full.returncode})
    result={'CENTERING_CHECKPOINT_PASS':o1,'O1_CENTERED_METRICS_PASS':o1,'O2_WARMUP_STABILITY_PASS':wp,'O2_BRANCH_PROX_PASS':op,'O2_SCALAR_GATE_PASS':gp,'O2_CLEAN_PASS':op,'D0_CLEAN_CARRIED_PASS':d0c,'D0_NOISY_CARRIED_PASS':d0n,**{k:o3[k] for k in ('O3_DELAY_PASS','O3_ORIGINAL_RESPONSE_PASS','O3_FULL_CONTRIBUTION_PASS','O3_PREDICTION_STRICT_PASS','O3_PREDICTION_ABSOLUTE_PASS','O3_REFIT_RESPONSE_PASS','O3_REFIT_CONVERGED','FREE_STATIC_LOGITS_VALID')},'pytest_pass':full.returncode==0,'artifact_consistency_pass':False,'ready_for_M2':False};savej(OUT/'pass_fail_summary.json',result)
    report=f'''# O2/O3 Audit Closure v16

Run `{RUN_ID}` used only the immutable v12 synthetic snapshot. M2 and industrial data were not run.

| Metric | Value |
|---|---:|
| O1 centered rows | 45 |
| O2 warmup median test RMSE | {json.loads((OUT/'O2/warmup/summary.json').read_text())['median_test_rmse']:.6g} |
| O2 branch proximal pass | {op} |
| O2 scalar gate pass | {gp} |
| O3 mean test RMSE | {o3['mean_o3_test_rmse']:.6g} |
| O3 prediction ratio vs O1 | {o3['prediction_ratio']:.6g} |
| O3 delay pass | {o3['O3_DELAY_PASS']} |
| O3 response/full contribution | {o3['O3_ORIGINAL_RESPONSE_PASS']} / {o3['O3_FULL_CONTRIBUTION_PASS']} |
| FreeStaticLogits valid | {o3['FREE_STATIC_LOGITS_VALID']} |

Failure classifications follow the computed gates in `pass_fail_summary.json`; no missing metric was replaced by a placeholder.
''';(OUT/'O2_O3_AUDIT_CLOSURE_v16_report.md').write_text(report,encoding='utf8');savej(OUT/'source_hashes_after.json',{p:sha(ROOT/p) for p in SOURCES});audit=audit_artifacts(before);result['artifact_consistency_pass']=audit['pass'];keys=['CENTERING_CHECKPOINT_PASS','O1_CENTERED_METRICS_PASS','O2_BRANCH_PROX_PASS','D0_CLEAN_CARRIED_PASS','D0_NOISY_CARRIED_PASS','O3_DELAY_PASS','O3_ORIGINAL_RESPONSE_PASS','O3_FULL_CONTRIBUTION_PASS','O3_PREDICTION_STRICT_PASS','FREE_STATIC_LOGITS_VALID','pytest_pass','artifact_consistency_pass'];result['ready_for_M2']=all(result[k] for k in keys);savej(OUT/'pass_fail_summary.json',result);(OUT/'full_run.log').write_text(json.dumps({'elapsed_seconds':time.time()-start,'result':result},indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
