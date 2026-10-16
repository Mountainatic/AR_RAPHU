"""Fresh L128 neural selection followed by frozen-epoch, ten-seed source fits."""
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from prism_benchmark import neural3 as n
from prism_benchmark.cpu_data import input_columns,sha256_file
from prism_benchmark.v211_support import load_native_samples,support_id_hash
from .accessor import FoldPCAAccessor
from .run_cpu import ROOT,TASK,view,dump,verify_release

def run():
    release_sha=verify_release()
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=ROOT/'baselines/neural';output.mkdir(parents=True,exist_ok=False)
    jobs=[j for j in json.loads((ROOT/'prepared/BASELINE_JOBS.json').read_text()) if j['family']=='NEURAL_BASELINE']
    records=[]
    for job in jobs:
        destination=output/job['direction']/job['information_set']/job['model'];destination.mkdir(parents=True)
        item=dict(job,status='RUNNING',test_accessed=False)
        dump(output/'RUN_STATUS.json',dict(status='RUNNING',current=item,records=records))
        try:
            shared=ROOT/'C1'/job['direction'];v=view(job['information_set']);dynamic=v.information_set=='dynamic'
            train=load_native_samples(shared,v,'train');validation=load_native_samples(shared,v,'validation')
            columns=input_columns(shared,TASK,'primary')
            accessor=FoldPCAAccessor(shared,v.head.dataset,'validation',[*columns,v.head.target]).for_fit(train)
            n._set_target_column(accessor,v.head.target)
            scaler=n.fit_scaler(accessor,train,columns,v.head.target,128)
            candidates=[n.Candidate(job['model'],0,128,capacity,lr,'L128') for capacity in n.CAPACITIES for lr in n.LEARNING_RATES]
            screening=[]
            for candidate in candidates:
                result=n._fit_one(candidate,accessor,train,validation,columns,v.head.target,scaler,
                    dynamic=dynamic,seed=n.SCREENING_SEED,device=device,max_epochs=100,patience=12,anchor_steps=1)
                screening.append(result);dump(destination/'SCREENING.json',screening)
            best=min(r['validation_mse'] for r in screening)
            selected=min((r for r in screening if r['validation_mse']<=best*1.01),
                key=lambda r:(r['parameter_count'],0 if r['capacity']=='SMALL' else 1,r['learning_rate']))
            if selected['parameter_count']>n.PARAMETER_BUDGET:raise RuntimeError('Parameter budget exceeded')
            candidate=next(c for c in candidates if c.candidate_id==selected['candidate_id'])
            contract=dict(task=TASK,history_steps=128,selected=selected,scaler=scaler.to_json(),joint_lift_pca=accessor.pca,
                final_fit='SOURCE_TRAIN_ONLY',source_fit_support_hash=support_id_hash(train),test_used_for_selection=False,
                frozen_epochs=selected['best_epoch'],final_seeds=job['final_seeds'],columns=columns)
            dump(destination/'SELECTION_FREEZE.json',contract)
            seeds=[]
            for seed in job['final_seeds']:
                if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('NEW_TASK_LOW_DISK')
                model,fit=n._fit_fixed_epochs(candidate,accessor,train,columns,scaler,dynamic=dynamic,
                    seed=seed,epochs=selected['best_epoch'],device=device)
                expected=n._predict(model,accessor,validation,columns,128,scaler,dynamic=dynamic,device=device)
                checkpoint=destination/f'seed_{seed}.pt'
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
                loaded=n.build_model(candidate.model,len(columns)+int(dynamic),candidate.capacity).to(device)
                loaded.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
                observed=n._predict(loaded,accessor,validation,columns,128,scaler,dynamic=dynamic,device=device)
                error=float(np.max(np.abs(observed-expected),initial=0.))
                if error>1e-6:raise RuntimeError(f'Neural reload mismatch {error}')
                frame=validation.copy();frame['y_pred']=observed
                frame['current_level']=accessor.block_means(validation,v.head.target,[(0,1)]).reshape(-1)
                frame['future_level_true']=frame.current_level+frame.y_true;frame['future_level_pred']=frame.current_level+observed
                prediction=destination/f'validation_seed_{seed}.parquet';frame.to_parquet(prediction,index=False,compression='zstd')
                seeds.append(dict(seed=seed,fit=fit,checkpoint=str(checkpoint),sha256=sha256_file(checkpoint),
                    validation=str(prediction),validation_sha256=sha256_file(prediction),reload_max_abs_error=error))
                dump(destination/'SEED_PROGRESS.json',seeds)
                del model,loaded;torch.cuda.empty_cache()
            item.update(status='FROZEN',selection_path=str(destination/'SELECTION_FREEZE.json'),seeds=seeds)
            dump(destination/'DEPLOYMENT.json',item)
        except Exception as e:item.update(status='FAILED_RETAINED',error=str(e),traceback=traceback.format_exc())
        records.append(item);dump(output/'RUN_STATUS.json',dict(status='RUNNING',records=records))
        print(job['direction'],job['information_set'],job['model'],item['status'],flush=True)
    status='ALL_NEURAL_JOBS_FROZEN' if all(r['status']=='FROZEN' for r in records) else 'FAILED_RETAINED'
    dump(output/'NEURAL_FREEZE_MANIFEST.json',dict(status=status,records=records,test_used_for_selection=False,baseline_release_sha256=release_sha))
    dump(output/'RUN_STATUS.json',dict(status=status,records=records))

if __name__=='__main__':run()
