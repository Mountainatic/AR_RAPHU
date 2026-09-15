"""One-shot formal inference from development-frozen contracts; no fitting."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('/root/autodl-tmp/PRISM_V211_STRICT_OOF_PUBLIC5_HYBRID_HW_20260912_R5_PENALIZED_OBJECTIVE_FULL_6X2_R1')
TEP=Path('/root/autodl-tmp/PRISM_V211_STRICT_OOF_TEP_REPAIR_73D0BD4_20260913_R2')
SHARED=Path('/root/autodl-tmp/PRISM_R5_SHARED_INPUTS')
PART=Path('/root/autodl-tmp/FORMAL_PARTITION_MATERIALIZATION_20260914')
VIEWS=[
('Debutanizer','DEB_C4__H5__W1','public3_shared','primary','record_time',True),
('SRU H2S','SRU_H2S_REP_H1__H1__W1','sru_shared','primary','record_time',True),
('SRU SO2','SRU_SO2_REP_H1__H1__W1','sru_shared','primary','record_time',True),
('PMSM proxy-excluded','PMSM_PM5__H600__W60','public3_shared','proxy_excluded','record_time',True),
('MetroPT P60','METRO_P60__H6__W1','public3_shared','proxy_excluded','record_time',True),
('MetroPT Oil20','METRO_OIL20__H120__W12','public3_shared','primary','record_time',True),
('TEP input-only','TEP_G_NOWCAST_H0__H0__W1','tep_shared','proxy_excluded','record_time',False),
('TEP record-time','TEP_G_NOWCAST_H0__H0__W1','tep_shared','proxy_excluded','record_time',True),
('TEP maturity-5','TEP_G_NOWCAST_H0__H0__W1','tep_shared','proxy_excluded','analyzer_maturity_5_steps',True)]
PART_NAMES={'debutanizer':'Debutanizer','sru':None,'pmsm':'PMSM','metropt':None,'tep':'TEP'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def ahash(ids,x):
 h=hashlib.sha256()
 for v in ids: h.update(str(v).encode()); h.update(b'\0')
 h.update(np.asarray(x,dtype='<f8').tobytes()); return h.hexdigest()
def metrics(y,p):
 e=np.asarray(p)-np.asarray(y); mse=float(np.mean(e*e,dtype=np.float64)); den=float(np.sum((y-np.mean(y))**2,dtype=np.float64))
 return {'rmse':float(np.sqrt(mse)),'mae':float(np.mean(np.abs(e),dtype=np.float64)),'delta_r2':float(1-np.sum(e*e)/den) if den else None}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--preflight',type=Path,required=True); a=ap.parse_args()
 sys.path.insert(0,str(a.project/'PRISM_INDUSTRIAL_BENCHMARK_V1/src'))
 from prism_benchmark.cpu_data import BaseAccessor,ViewSpec,load_heads
 from prism_benchmark.v211_support import load_native_samples,support_id_hash
 from prism_benchmark.v211_public_all_baselines import SupportRequirement,apply_common_requirements
 from prism_benchmark.v2_k import profile_values
 from prism_benchmark.v2_urysohn import predict_contract,basis_from_metadata
 from prism_benchmark.v2_c import _ridge_predict
 from prism_benchmark.v211_w import predict_w_correction
 from prism_benchmark.v211_a import mature_residual_features,predict_mature_residual_ar
 from prism_benchmark.representative_prism_checkpoints import _predict_joint,_w_basis_from_contract
 from prism_benchmark.v2_config import load_frozen_config
 pre=read(a.preflight); allowed={'PASS','PASS_WITH_ONE_JOINT_NOT_APPLICABLE'}
 if pre['status'] not in allowed or pre['formal_target_values_read'] or pre['fit_calls'] or pre['selection_calls']: raise RuntimeError('preflight not clean')
 lockp=PART/'LOCK/FINAL_FORMAL_TEST_PROTOCOL_LOCK_v2.json'; lock=read(lockp)
 if not lock['protocol_frozen'] or lock['formal_test_opened']: raise RuntimeError('partition lock invalid')
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'PREDICTIONS').mkdir(exist_ok=True)
 execution={'status':'FROZEN','created_utc':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.project,text=True).strip(),'partition_lock_sha256':sha(lockp),'preflight_sha256':sha(a.preflight),'final_fit_policy':'NO_REFIT_REUSE_DEVELOPMENT_FINAL_SELECTED_CONTRACTS','fit_calls_allowed':False,'selection_calls_allowed':False,'formal_views':9,'joint_exception':'TEP maturity-5: NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT','formal_test_opened':False}
 execution_path=a.output/'EXECUTION_LOCK.json'; opening_path=a.output/'FORMAL_TEST_OPENING.json'
 if execution_path.exists() and opening_path.exists():
  prior_execution=read(execution_path); opening=read(opening_path)
  if prior_execution['partition_lock_sha256']!=execution['partition_lock_sha256'] or prior_execution['preflight_sha256']!=execution['preflight_sha256']: raise RuntimeError('resume lock mismatch')
  execution_sha=sha(execution_path); opened=opening['timestamp_utc']
  failure_path=a.output/'RUN_ATTEMPT_1_FAILURE.json'
  if not failure_path.exists(): write(failure_path,{'status':'FAILED_BEFORE_PREDICTION_OR_METRIC','opened_utc':opened,'error_type':'EMPTY_MEMBERSHIP_FROM_SOURCE_VIEW_KEY_FORMAT','formal_target_values_read':True,'formal_predictions_generated':False,'formal_metrics_computed':False,'model_or_protocol_changed':False})
 else:
  write(execution_path,execution); execution_sha=sha(execution_path); opened=datetime.now(timezone.utc).isoformat(); opening={'FORMAL_TEST_OPENED':True,'timestamp_utc':opened,'commit':execution['commit'],'partition_lock_sha256':execution['partition_lock_sha256'],'execution_lock_sha256':execution_sha,'first_target_access_follows_this_event':True}; write(opening_path,opening)
 config=load_frozen_config(a.project/'PRISM_INDUSTRIAL_BENCHMARK_V1')
 table_path=a.output/'TABLE_STAGEWISE.csv'; replay_path=a.output/'FORMAL_EXACT_ZERO_REPLAY_AUDIT.csv'
 rows=pd.read_csv(table_path).to_dict('records') if table_path.exists() else []
 replay=pd.read_csv(replay_path).to_dict('records') if replay_path.exists() else []
 completed=set()
 for prior_task,_,_,_,_,prior_dynamic in VIEWS:
  expected={'K','KC','KCW','KCWA'} | ({'J'} if prior_dynamic and prior_task!='TEP maturity-5' else set())
  present={str(row['model']) for row in rows if row['task']==prior_task}
  prediction_path=a.output/'PREDICTIONS'/(prior_task.replace(' ','_')+'.parquet')
  if prediction_path.exists() and expected <= present:
   completed.add(prior_task)
 if opening_path.exists() and completed and not (a.output/'RUN_ATTEMPT_2_FAILURE.json').exists():
  write(a.output/'RUN_ATTEMPT_2_FAILURE.json',{
   'status':'FAILED_AFTER_PARTIAL_PREDICTION_AND_METRICS',
   'observed_utc':datetime.now(timezone.utc).isoformat(),
   'error_type':'FROZEN_COMMON_SUPPORT_NOT_APPLIED_BEFORE_LONG_HISTORY_FEATURE_ACCESS',
   'error_message':'ValueError: block outside entity support: 2',
   'completed_tasks_preserved':sorted(completed),
   'formal_test_opened':True,
   'formal_target_values_read':True,
   'model_or_protocol_changed':False,
   'remediation':'MECHANICAL_FROZEN_CONTRACT_COMMON_SUPPORT_FILTER_AND_RESUME'
  })
 e1=pd.read_csv('/root/autodl-tmp/PRISM_STRICT_OOF_E1_E6_REVALIDATION_20260914_R14_FINAL_PACKAGE/E1_STAGEWISE/table_stagewise.csv').set_index('task')
 for task,hid,sname,proxy,availability,dynamic in VIEWS:
  if task in completed:
   print(task,'SKIP_COMPLETED_PRESERVED',flush=True)
   continue
  started=time.time(); shared=SHARED/sname; h=next(x for x in load_heads(shared,False) if x.head_id==hid); view=ViewSpec(h,'dynamic' if dynamic else 'input_only',availability,proxy); dr=TEP if h.dataset=='tep' else BASE
  C=read(BASE/f'results/DEVELOPMENT/C/{hid}/{proxy}/RESULT.json'); W=read(BASE/f'results/DEVELOPMENT/W/{hid}/{proxy}/RESULT.json'); channel=C['best_active_k_channel']; K=read(BASE/f'results/DEVELOPMENT/K/{hid}/{proxy}/{channel}/RESULT.json')
  A=read(dr/f'results/DEVELOPMENT/A/{hid}/{availability}/{proxy}/RESULT.json') if dynamic else None; J=read(dr/f'results/DEVELOPMENT/JOINT/{hid}/{availability}/{proxy}/RESULT.json') if dynamic else None
  samples=load_native_samples(shared,view,'test')
  pname='SRU-H2S' if h.target=='y1' else 'SRU-SO2' if h.target=='y2' else 'MetroPT-P60' if h.target=='Reservoirs' else 'MetroPT-Oil20' if h.target=='Oil_temperature' else PART_NAMES[h.dataset]
  source_view=f'{view.information_set}/{view.availability_scenario}/{view.proxy_policy}'
  members=pd.read_parquet(PART/f'PARTITIONS/{pname}/formal_test.parquet',columns=['base_origin_id','source_view'],filters=[('source_view','==',source_view)])
  samples=members[['base_origin_id']].merge(samples,on='base_origin_id',validate='one_to_one').sort_values(['entity_id','origin']).reset_index(drop=True)
  if len(samples)!=len(members): raise RuntimeError(f'membership mismatch {task}')
  formal_membership_rows=len(samples)
  frozen_input_histories=[int(x) for x in C.get('active_selected_k_histories',{}).values()]
  frozen_input_histories.extend(int(item['profile'][1]) for item in C['channel_contracts'])
  frozen_input_histories.append(int(K['selected_profile'][1]))
  requirements=[SupportRequirement(input_history_steps=max(frozen_input_histories or [1]))]
  if dynamic and A['selection']['routing_status']=='ACTIVE':
   apf=A['final_selected_contract']['profile']; requirements.append(SupportRequirement(target_delta_steps=int(apf[0]),target_history_steps=int(apf[1])))
  if dynamic and task!='TEP maturity-5' and J.get('ar_profile'):
   jpf=J['ar_profile']; requirements.append(SupportRequirement(target_delta_steps=int(jpf[0]),target_history_steps=int(jpf[1])))
  samples=apply_common_requirements(samples,requirements).reset_index(drop=True)
  if samples.empty: raise RuntimeError(f'empty frozen common support {task}')
  accessor=BaseAccessor(shared,h.dataset,'test',[*C['active_channels'],channel,h.target])
  compressed=[]; joint=[]
  for item in C['channel_contracts']:
   values,_=profile_values(accessor,samples,item['channel'],tuple(item['profile']),item['m_tau']); contract=item['k_contract']; compressed.append(predict_contract(values,contract)); raw=basis_from_metadata(contract['basis']).transform(values).reshape(len(values),-1); joint.append(raw[:,np.asarray(item['joint_columns'],dtype=np.int64)])
  cm=np.column_stack(compressed); jm=np.concatenate(joint,axis=1); gc=np.asarray(C.get('global_joint_columns',[]),dtype=np.int64); jm=jm[:,gc] if len(gc) else jm
  values,_=profile_values(accessor,samples,channel,tuple(K['selected_profile']),K['selected_m_tau']); pk=predict_contract(values,K['final_selected_contract'])
  cc=C['final_selected_contract']; family=cc['family']; pc=cm[:,C['active_channels'].index(cc['channel'])] if family in {'BEST_ACTIVE_K','BEST_ACTIVE_K_CHANNEL'} else _ridge_predict(jm if family=='ADDITIVE_JOINT_BASIS' else cm,cc)
  pw=pc+predict_w_correction(pc,W['final_selected_contract'])
  routes={'C':C['family_selection']['routing_status'],'W':W['selection']['routing_status'],'A':A['selection']['routing_status'] if A else 'ZERO_IDENTITY'}
  pkc=pk.copy() if routes['C']=='ZERO_IDENTITY' else pc
  pkw=pkc.copy() if routes['W']=='ZERO_IDENTITY' else pw
  if dynamic and routes['A']=='ACTIVE':
   contract=A['final_selected_contract']; oof=pd.read_parquet(BASE/'results'/W['oof_path']); train=load_native_samples(shared,view,'train'); oof=train[['base_origin_id']].merge(oof,on='base_origin_id',validate='one_to_one'); oof['residual']=oof['y_true']-oof['physical_w_oof']; devw=pd.read_parquet(BASE/'results'/W['prediction_path']); validation=load_native_samples(shared,view,'validation')[['base_origin_id']].merge(devw,on='base_origin_id',validate='one_to_one'); validation['residual']=validation['y_true']-validation['y_pred']; testres=samples[['entity_id','origin','y_true']].copy(); testres['residual']=testres['y_true']-pw; source=pd.concat([oof[['entity_id','origin','residual']],validation[['entity_id','origin','residual']],testres[['entity_id','origin','residual']]],ignore_index=True); rmean=float(oof['residual'].mean()); af,coverage,causal=mature_residual_features(samples,source,h_steps=h.h_steps,w_steps=h.w_steps,delta=int(contract['profile'][0]),history=int(contract['profile'][1]),maximum_lags=int(config['A_module']['state_profile']['maximum_lags']),residual_mean=rmean); pa=pw+predict_mature_residual_ar(af,contract)
  else: pa=pkw.copy(); causal={'past_test_truth_allowed':False}
  preds={'K':pk,'KC':pkc,'KCW':pkw,'KCWA':pa}
  joint_status='NOT_APPLICABLE_INPUT_ONLY'
  if dynamic and task!='TEP maturity-5':
   jc=J['final_selected_contract']; blocks={'K':cm if jc['k_representation']=='CHANNEL_COMPRESSED' else jm,'W':_w_basis_from_contract(pc,J['joint_w_basis_contract']),'A':accessor.target_state(samples,h.target,*J['ar_profile'])}; preds['J']=_predict_joint(blocks,jc); joint_status='PASS'
  elif dynamic: joint_status='NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT'
  current=accessor.block_means(samples,h.target,[(0,int(h.w0_steps))]).reshape(-1); y=samples['y_true'].to_numpy(dtype=np.float64); future=current+y; persist=metrics(future,current)
  out=samples[['base_origin_id','entity_id','origin']].copy(); out['y_true']=y
  for name,pred in preds.items():
   out[name]=pred; m=metrics(y,pred); lm=metrics(future,current+pred); m.update(level_rmse=lm['rmse'],level_mae=lm['mae'],level_r2=lm['delta_r2'],persistence_rmse=persist['rmse'],persistence_r2=persist['delta_r2'],persistence_skill=1-(m['rmse']**2)/(persist['rmse']**2) if persist['rmse'] else None); er=e1.loc[task]; params=int(er[f'{name}_parameter_count']) if f'{name}_parameter_count' in er else int(J['final_selected_contract']['parameter_count']); rows.append({'task':task,'view':f'{view.information_set}/{availability}','model':name,'formal_membership_rows':formal_membership_rows,'rows':len(samples),'support_hash':support_id_hash(samples),'prediction_hash':ahash(samples['base_origin_id'],pred),'parameter_count':params,'joint_status':joint_status,**routes,**m})
  for child,parent,stage in [('KC','K','C'),('KCW','KC','W'),('KCWA','KCW','A')]:
   if routes[stage]=='ZERO_IDENTITY': replay.append({'task':task,'stage':stage,'max_abs_difference':float(np.max(np.abs(preds[child]-preds[parent]),initial=0)),'prediction_hash_equal':ahash(samples['base_origin_id'],preds[child])==ahash(samples['base_origin_id'],preds[parent]),'status':'PASS'})
  dest=a.output/'PREDICTIONS'/(task.replace(' ','_')+'.parquet'); out.to_parquet(dest,index=False,compression='zstd'); print(task,len(samples),joint_status,round(time.time()-started,1),flush=True); del accessor,cm,jm,out,samples
  pd.DataFrame(rows).to_csv(table_path,index=False); pd.DataFrame(replay).to_csv(replay_path,index=False)
  write(a.output/'RUN_STATUS.json',{'status':'RUNNING','opened_utc':opened,'completed_tasks':task})
 write(a.output/'RUN_STATUS.json',{'status':'COMPLETED','opened_utc':opened,'completed_utc':datetime.now(timezone.utc).isoformat(),'completed_tasks':[x[0] for x in VIEWS],'fit_calls':0,'selection_calls':0,'joint_not_applicable':['TEP maturity-5']})
 print('FORMAL COMPLETED',flush=True)
if __name__=='__main__': main()
