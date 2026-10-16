"""New-E1 same-rod whitening replay, using sealed complete Urysohn K/C/W/A."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import sys,json,time,hashlib,traceback,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MAIN=Path('/root/autodl-tmp/CZ_CROSSROD_L128_STAGEWISE_V3_20261012')
MAIN_PIN='a1109dc5d5e3b6bc0f00ed018b4399ad87b2bf3691f6790c5d881e1b48b3f507'
sys.path[:0]=[str(MAIN),str(MAIN/'code'),str(MAIN/'code/src')]
from finalize_crossrod_fast import main_runner
from fast_inference import FastInferencePipeline
from canonical.cz_crossrod_l128_20261008.deployment import verify_deployment
from canonical.cz_crossrod_l128_20261008.data import load_inference,SAMPLE_RELATIVE
from canonical.cz_crossrod_l128_20261008.invariants import file_sha,digest
from canonical.cz_crossrod_l128_20261008.pointwise import paired_pointwise,export_pointwise
from canonical.cz_crossrod_l128_20261008.discrete_observation import DiscreteObservationSpec
from canonical.cz_crossrod_l128_20261008.discrete_audit import audit_target_contract,CZHeadAuditConfig
from canonical.cz_crossrod_l128_20261008.discrete_diagnostics import segmented_diagnostics,grid_candidate_evidence
import whitening_statistics as ws

CASES={'Rod1_Common4':('Rod_1','Rod_1_to_Rod_2'),'Rod2_Common4':('Rod_2','Rod_2_to_Rod_1')}
START=time.perf_counter()
def dump(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def log(message):print(f'[{time.perf_counter()-START:.1f}s] {message}',flush=True)
def ordered_states(state):
    nodes=[]
    while state is not None:nodes.append(state);state=state.get('parent')
    return list(reversed(nodes))

def preregister():
    if (ROOT/'EXPERIMENT_PREREGISTRATION.json').exists():raise RuntimeError('Refuse overwrite or reinterpret an existing run')
    assert json.loads((MAIN/'release/RELEASE_PIN.json').read_text())['release_sha256']==MAIN_PIN
    membership=pd.read_parquet(ROOT/'LEGACY_FROZEN_MEMBERSHIP.parquet')
    original=pd.read_parquet(ROOT/'LEGACY_ORIGINAL_SOURCE_MEMBERSHIP.parquet')
    bound={};cases={};contexts={}
    for arm,(rod,case) in CASES.items():
        runner=main_runner(case);context=runner._boot();dep=MAIN/'formal_actual'/case/'DEPLOYMENT_SEAL.json'
        state,_=verify_deployment(dep,file_sha(dep),context['registration'],MAIN_PIN)
        bound[str(dep)]=file_sha(dep)
        stages=ordered_states(state);assert [n['stage'] for n in stages]==['K','C','W','A']
        train=pd.read_parquet(MAIN/'C1'/case/SAMPLE_RELATIVE/'train.parquet')
        wanted=membership[(membership.rod==rod)&(membership.split=='train')]
        assert set(zip(train.entity_id,train.origin))==set(zip(wanted.entity,wanted.origin))
        wanted_original=original[(original.rod==rod)&(original.original_split=='train')]
        assert set(zip(train.entity_id,train.origin))==set(zip(wanted_original.entity,wanted_original.origin))
        assert len(membership[(membership.rod==rod)&(membership.split=='purged')])==257
        for node in stages:
            for proof in node['outer_realizations']:
                path=Path(proof['pointwise_path']).with_name('FROZEN_OOF_'+str(proof['fold'])+'.parquet')
                bound[str(path)]=file_sha(path)
        context.update(deployment_checkpoint_verified=True,inference_operation='source_validation')
        contexts[arm]=(runner,context,state)
        cases[arm]=dict(training_rod=rod,evaluation_rod=rod,inherited_case_id=case,train_rows=len(train),
            validation_rows=int(((membership.rod==rod)&(membership.split=='validation')).sum()),
            descriptive_holdout_rows=int(((membership.rod==rod)&(membership.split=='test')).sum()),
            routes={n['stage']:n['routing_status'] for n in stages},A_profile=state.get('profile'),
            deployment_sha256=file_sha(dep),full_new_E1_model_used=True)
    for path in [MAIN/'INFERENCE_EXTENSION_SEAL.json',MAIN/'INFERENCE_EXTENSION_ACCEPTANCE.json',MAIN/'release/RELEASE_SEAL.json']:
        bound[str(path)]=file_sha(path)
    sources={p.name:file_sha(p) for p in ROOT.glob('*.py')}
    protocol=dict(schema='CZ_NEW_E1_SAME_ROD_WHITENING_V1',classification='POST_HOC_DESCRIPTIVE_WHITENING_DIAGNOSTIC',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),main_root=str(MAIN),main_release_sha256=MAIN_PIN,
        model='COMPLETE_NEW_STAGEWISE_URYSHON_K_C_W_A',input_history_samples=128,input_history_seconds=256,
        membership_sha256=file_sha(ROOT/'LEGACY_FROZEN_MEMBERSHIP.parquet'),cases=cases,
        split_rule='EXACT_PRIOR_SAME_ROD_MEMBERSHIP: original source train; validation first half; purge257; descriptive tail',
        main_model_training='EXACT_VERIFIED_REUSE_OF_NEW_E1_SOURCE_TRAIN_CHECKPOINTS',
        main_route_or_recipe_retuning=False,main_model_fit_allowed=False,
        A_recipe='EXACT_NEW_E1_SOURCE_DEVELOPMENT_SELECTED_AND_FROZEN',
        scoring='EMPTY_RESIDUAL_STATE_PER_SPLIT_AND_ENTITY; ONLY_MATURE_PREVIOUS_DEPLOYED_W_RESIDUALS',
        probe=dict(H=32,alpha=1.,fit='SEALED_NEW_E1_OUTER_OOF_GLOBAL_DEPLOYED_RESIDUALS',fold_boundary_reset=True,
                   probe_fit_before_validation_and_holdout_scoring=True,used_for_selection=False),
        PSD=dict(fs_Hz=.5,nperseg=1024,noverlap=512,band_Hz=[.125,.155],detrend='constant',method='WELCH_SAME_AS_PRIOR_WHITENING'),
        ACF=dict(method='GLOBAL_CENTERED_BIASED_SEGMENTED_FFT',E_lags=[160,900]),
        v06_diagnostics='CURRENT_RELEASE_DISCRETE_OBSERVATION_TARGET_AUDIT_SEGMENTED_PSD_ACF_PACF_READ_ONLY',
        white_noise_gate=False,instrument_unit_verified=False,cadence_evidence='REGISTERED_2S_NOT_INDEPENDENT_TIMESTAMPS',
        all_legal_arm='NOT_IN_CURRENT_NEW_E1_REGISTERED_CHANNEL_UNIVERSE; NOT_EXECUTED_OR_SUBSTITUTED_WITH_RIDGE',
        historical_test_seen=True,independent_confirmatory_claim=False,
        bound_artifact_sha256=bound,adapter_source_sha256=sources)
    dump('EXPERIMENT_PREREGISTRATION.json',protocol)
    return protocol,membership,contexts

def fit_fixed_probes(contexts):
    probes={};exports=[]
    for arm,(_,context,state) in contexts.items():
        for node in ordered_states(state):
            pieces=[]
            for proof in node['outer_realizations']:
                path=Path(proof['pointwise_path']).with_name('FROZEN_OOF_'+str(proof['fold'])+'.parquet')
                f=pd.read_parquet(path)
                assert set(f.split)=={'train'}
                assert (f.global_routing_status==node['routing_status']).all()
                f=f[['entity_id','origin','fold','y_true','global_deployed_prediction','global_residual']].copy()
                assert np.allclose(f.y_true-f.global_deployed_prediction,f.global_residual,atol=1e-14,rtol=0)
                pieces.append(f)
            cf=pd.concat(pieces,ignore_index=True)
            assert not cf[['entity_id','origin']].duplicated().any()
            model=ws.fit_probe(cf,'global_residual');stage=node['stage'];probes[arm,stage]=model
            path=ROOT/'probe_training'/(arm+'_'+stage+'_CF.parquet');path.parent.mkdir(exist_ok=True)
            cf.to_parquet(path,index=False);exports.append(path)
            destination=ROOT/'probes'/(arm+'_'+stage+'.npz');destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination,**model);exports.append(destination)
    dump('GLOBAL_MODEL_AND_PROBE_FREEZE.json',dict(status='FROZEN_BEFORE_NEW_SAME_ROD_SCORING',
        main_release_sha256=MAIN_PIN,preregistration_sha256=file_sha(ROOT/'EXPERIMENT_PREREGISTRATION.json'),
        artifact_sha256={str(p):file_sha(p) for p in exports},diagnostic_probe_fit_calls=len(probes),main_model_fit_calls=0))
    return probes

def raw_audit(context,frame,arm,split):
    source=Path(context['C1_manifest']['source_shared'])/'base_data/cz_czochralski/train.parquet'
    raw=pd.read_parquet(source,columns=['entity_id','row_in_entity','crystal_diameter'])
    raw=raw.rename(columns={'row_in_entity':'raw_index','crystal_diameter':'recorded_level'})
    raw['segment_id']=raw.entity_id;raw['split_id']='SOURCE_ROD_OBSERVATION_CONTEXT'
    spec=DiscreteObservationSpec(sample_period_seconds=2.,time_origin_policy='EXCLUSIVE_CZ_ORIGIN',
        timestamp_field=None,output_unit='recorded_diameter_unit_unverified',head_contract_id='CZ_NEW_E1_L128_H1_SAME_ROD')
    samples=frame.copy();samples['segment_id']=samples.entity_id;samples['split_id']='SOURCE_ROD_OBSERVATION_CONTEXT'
    audit=audit_target_contract(raw,CZHeadAuditConfig('CZ_L128_H1_D_t_MINUS_D_t_MINUS_1',1,128),spec,samples=samples)
    assert audit.legal_support.all(),audit.loc[~audit.legal_support,'invalid_reason'].value_counts().to_dict()
    audit['experiment_split']=split
    path=ROOT/'target_alignment'/(arm+'_'+split+'.csv');path.parent.mkdir(exist_ok=True);audit.to_csv(path,index=False)
    level=raw.set_index(['entity_id','raw_index']).recorded_level.reindex(pd.MultiIndex.from_arrays([frame.entity_id,frame.origin])).to_numpy(float)
    return spec,level,audit

def score(contexts,membership,probes):
    metrics=[];predictability=[];verification=[];points=[];v06summary=[]
    for arm,(runner,context,state) in contexts.items():
        rod,_=CASES[arm]
        # This is source-rod validation data under the same verified inference barrier.
        ev,_,shared,accessor_split=load_inference(dict(context,mode='inference_only'),'source_validation')
        prior=membership[(membership.rod==rod)&(membership.original_split=='source_validation')]
        assert set(zip(ev.entity_id,ev.origin))==set(zip(prior.entity,prior.origin))
        for split,legacy_split in [('validation','validation'),('descriptive_holdout','test')]:
            subset=membership[(membership.rod==rod)&(membership.split==legacy_split)]
            wanted=set(zip(subset.entity,subset.origin))
            frame=ev.loc[[pair in wanted for pair in zip(ev.entity_id,ev.origin)]].copy().reset_index(drop=True)
            assert len(frame)==len(subset)>900 and set(zip(frame.entity_id,frame.origin))==wanted
            output=ROOT/'predictions'/arm/split
            pipe=runner._pipeline(context,output,'inference_only');pipe.bind_deployment(state)
            pipe.accessor=pipe._make_accessor(shared,accessor_split)
            assert isinstance(pipe,FastInferencePipeline)
            spec,level,audit=raw_audit(context,frame,arm,split)
            for node in ordered_states(state):
                stage=node['stage'];parent=np.zeros(len(frame)) if stage=='K' else pipe.emit(node['parent'],frame.drop(columns=['y_true']))
                if stage=='A':child,native,proof=pipe.online_a(node,frame)
                else:child,native,_=pipe.predict(node,frame.drop(columns=['y_true']));proof={}
                prediction=child if node['routing_status']=='ADMITTED' else parent
                record,binding=paired_pointwise(frame,stage=stage,kind='source_validation',parent=parent,
                    model_child=child,deployed=prediction,row_domain=native,routing_status=node['routing_status'],
                    policy_id=pipe.policy_id,partition_id=pipe.partition_id(frame),parameters=node,state_provenance=proof)
                binding.update(experiment_role='SAME_ROD_'+split.upper(),experiment_preregistration_sha256=file_sha(ROOT/'EXPERIMENT_PREREGISTRATION.json'))
                export_pointwise(output/(stage+'_pointwise.parquet'),record,binding)
                measured=record.copy();measured['recorded_level']=level;measured['raw_index']=measured.origin
                measured['segment_id']=measured.entity_id;measured['split_id']=split
                measured['registered_target']=measured.y_true
                stats,acf,pacf,psd=ws.diagnose(measured,'residual',arm=arm,split=split,stage=stage,route=node['routing_status'])
                metrics.append(stats);dest=ROOT/'diagnostics'/arm/split;dest.mkdir(parents=True,exist_ok=True)
                acf.to_csv(dest/(stage+'_ACF.csv'),index=False);pacf.to_csv(dest/(stage+'_PACF.csv'),index=False);psd.to_csv(dest/(stage+'_PSD.csv'),index=False)
                predictability.extend(ws.probe_metrics(measured,'residual',probes[arm,stage],arm=arm,split=split,stage=stage))
                record['experiment_arm']=arm;record['experiment_split']=split;points.append(record)
                q='post-'+stage+(' innovation' if stage=='A' else ' residual');measured[q]=measured.residual
                v06=segmented_diagnostics(measured,spec,value_columns=('recorded_level','registered_target',q))
                for j,(summary,spectrum,correlation) in enumerate(v06):
                    summary.update(arm=arm,split=split,stage=stage,fit_called=False)
                    v06summary.append(summary)
                    dest_v06=ROOT/'v06_diagnostics'/arm/split/stage;dest_v06.mkdir(parents=True,exist_ok=True)
                    if spectrum is not None:spectrum.to_csv(dest_v06/(str(j)+'_PSD.csv'),index=False);correlation.to_csv(dest_v06/(str(j)+'_ACF_PACF.csv'),index=False)
                centered=record.residual.to_numpy(float)-record.residual.mean()
                direct=np.correlate(centered,centered,'full')[len(centered)-1:len(centered)+900]/np.dot(centered,centered)
                assert np.max(abs(direct-acf.ACF.to_numpy()))<1e-12
                if stage=='A':
                    # Legal maturity, N/A identity, and no current/future target use.
                    assert np.array_equal(child[~native],parent[~native])
                    k=min(160,len(frame)//3);changed=frame.copy();changed.loc[k:,'y_true']+=1e6
                    revised,mask2,_=pipe.online_a(node,changed)
                    assert np.array_equal(native,mask2) and np.array_equal(child[:k+1],revised[:k+1])
                    reverse=frame.iloc[100:].copy().reset_index(drop=True)
                    _,reset_native,_=pipe.online_a(node,reverse)
                    offsets=__import__('canonical.cz_crossrod_l128_20261008.a_state',fromlist=['offsets']).offsets(node['profile'],pipe.A_max)
                    required=int(max(offsets))+1
                    assert not reset_native[:required].any()
                    assert not native[:required].any()
                # Frozen W/K predictors must be independent of scored targets.
                if stage!='A':
                    changed=frame.copy();changed.y_true+=1e6
                    replay,_native,_=pipe.predict(node,changed)
                    assert np.array_equal(child,replay)
                verification.append(dict(arm=arm,split=split,stage=stage,rows=len(frame),
                    model_fit_calls=pipe.fit_calls,FFT_direct_ACF_max_error=float(np.max(abs(direct-acf.ACF.to_numpy()))),
                    scored_target_not_static_predictor=True,A_future_maturity_check=stage=='A',
                    cold_start_rows=int((~native).sum()) if stage=='A' else None,legal_target_rows=int(audit.legal_support.sum())))
            pipe.assert_inference_only();assert pipe.fit_calls==0
            log(arm+' '+split+' K/C/W/A complete')
    result=pd.DataFrame(metrics);result.to_csv(ROOT/'WHITENING_METRICS.csv',index=False)
    pd.DataFrame(predictability).to_csv(ROOT/'RESIDUAL_PREDICTABILITY.csv',index=False)
    pd.concat(points,ignore_index=True).to_parquet(ROOT/'ALL_STAGE_POINTWISE.parquet',index=False)
    dump('v06_diagnostics/SUMMARY.json',dict(status='COMPLETED',segments=v06summary,model_release_sha256=MAIN_PIN,white_gate=False))
    dump('VERIFICATION.json',dict(status='PASS',checks=verification,main_model_fit_calls=0,
        exact_new_E1_model_chain_used=True,legacy_split_membership_preserved=True,new_model_source_and_freezes_verified=True))
    return result

def make_report(metrics,protocol):
    probe=pd.read_csv(ROOT/'RESIDUAL_PREDICTABILITY.csv')
    comparisons=[]
    lines=['# 新版 E1 同棒白化实验重跑','',
        '本次实际使用最新逐层冻结的完整 Urysohn K/C/W/A，输入历史统一 128 点（256 秒）。复用刚完成的新版源棒训练并重新预测：Rod1 模型评分 Rod1，Rod2 模型评分 Rod2；模型、配方、route 不根据白化或尾段改动。', '',
        '训练源码与 release 绑定为 `'+MAIN_PIN+'`。原轻量白化实验的同棒划分保留：源棒验证前半段用于描述性验证，中间 257 行隔离，尾段为 descriptive holdout。因为配方沿用新版 E1 的源训练选择，这次前半验证也不重新选择 H/alpha。A 残差历史分别为 Rod1 96 点、Rod2 32 点。','',
        '| 同棒模型 | 原训练行数 | 本次验证 | 隔离 | 本次尾段 |','|---|---:|---:|---:|---:|']
    for arm,c in protocol['cases'].items():lines.append(f'| {arm} | {c["train_rows"]} | {c["validation_rows"]} | 257 | {c["descriptive_holdout_rows"]} |')
    lines += ['', '## 各阶段结果','', '| arm | split | stage | route | RMSE | E160 | E900 | 约7秒频带能量 | lag7 | lag14 |', '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for q in metrics.itertuples():lines.append(f'| {q.arm} | {q.split} | {q.stage} | {q.route} | {q.RMSE:.8f} | {q.E160:.4f} | {q.E900:.4f} | {q.seven_second_band_energy:.4g} | {q.lag7:.4f} | {q.lag14:.4f} |')
    lines += ['', '## A 相对 W 的变化','', '| arm | split | RMSE降幅 | E160降幅 | E900降幅 | 频带能量降幅 | 最终峰周期 |', '|---|---|---:|---:|---:|---:|---:|']
    for arm in CASES:
        for split in ['validation','descriptive_holdout']:
            b=metrics[(metrics.arm==arm)&(metrics.split==split)&(metrics.stage=='W')].iloc[0]
            a=metrics[(metrics.arm==arm)&(metrics.split==split)&(metrics.stage=='A')].iloc[0]
            q=dict(arm=arm,split=split)
            for name in ['RMSE','E160','E900','seven_second_band_energy']:q[name+'_reduction_percent']=100*(1-a[name]/b[name])
            comparisons.append(q)
            lines.append(f'| {arm} | {split} | {q["RMSE_reduction_percent"]:.2f}% | {q["E160_reduction_percent"]:.2f}% | {q["E900_reduction_percent"]:.2f}% | {q["seven_second_band_energy_reduction_percent"]:.2f}% | {a.dominant_period_seconds:.3f}s |')
    pd.DataFrame(comparisons).to_csv(ROOT/'A_VS_W_COMPARISON.csv',index=False)
    lines += ['', '正降幅表示指标减少，负值表示恶化。E160/E900 是 lags 1–160/900 的 ACF 平方和，不是误差能量或统计显著性。lag7=14秒，lag14=28秒，不等同于约7秒谱峰。','',
        '## 固定 residual probe','', '固定 H=32、alpha=1；分别仅拟合新版各阶段封存 outer-OOF 的残差。每折/实体/缺行均清空特征历史，未用部署模型的训练内误差，未用验证或尾段标签拟合 probe；该 probe 是诊断模型，其拟合与主模型零重训分别计数。','',
        '| arm | split | stage | 成熟历史 probe R² |','|---|---|---|---:|']
    for q in probe[probe.support=='mature_history'].itertuples():lines.append(f'| {q.arm} | {q.split} | {q.stage} | {q.R2:.4f} |')
    lines += ['', '## 解释与范围','',
        '这是新版 E1 的同棒白化重跑，与原 raw-history Ridge 的模型不同，不能把数值差异只归因于输入长度。原 Rod1_AllLegal 不在当前新版 E1 的注册四通道模型中，本次未用旧 Ridge 代替它，也未宣称该 arm 已完成。正式跨棒 E1 结果与协议保持独立封存。','',
        '本次另调用当前 release 的 v0.6 观测合同、目标时间索引审计和分段 PSD/ACF/PACF；只读，不改变标签、评分行、route、prediction rounding。对原始记录的单位/ADC 没有外部核实；2秒采样来自注册约定而非独立时间戳校验。','',
        '历史尾段已被查看，本次为回顾性诊断，不是独立确认性试验。Welch 按原实验固定 fs=0.5Hz、nperseg=1024、noverlap=512、频带0.125–0.155Hz。此频带与上一轮跨棒 v0.6 表的0.133–0.153Hz不同，不能直接横比能量占比。验证/尾段仅约两个 Welch 窗口；900阶相关有效对数有限。ACF图上的点态95%参考带只作描述，未用作验收或多重检验。指标下降不等于证明白噪声，也不识别7秒峰的物理来源。','',
        '## 图','']
    for arm in CASES:
        split='descriptive_holdout';fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for stage,color in [('K','#98a1a9'),('C','#67a9cf'),('W','#1767b2'),('A','#c65d28')]:
            d=ROOT/'diagnostics'/arm/split
            acf=pd.read_csv(d/(stage+'_ACF.csv'));psd=pd.read_csv(d/(stage+'_PSD.csv'))
            axes[0].plot(acf.lag_seconds,acf.ACF,label=stage,color=color,lw=1.2)
            axes[1].semilogy(psd.frequency_Hz,psd.PSD,label=stage,color=color,lw=1.2)
        axes[0].set(xlim=(0,320),xlabel='Lag (seconds)',ylabel='ACF')
        axes[1].set(xlim=(.1,.18),xlabel='Frequency (Hz)',ylabel='Welch PSD (recorded units²/Hz)')
        axes[1].axvspan(.125,.155,color='gray',alpha=.1)
        for ax in axes:ax.legend();ax.grid(alpha=.15)
        fig.suptitle(arm+' — new E1 same-rod descriptive holdout, L128')
        name=arm+'_HOLDOUT_ACF_PSD.png';fig.savefig(ROOT/name,dpi=180);plt.close(fig)
        lines += ['!['+arm+']('+name+')','']
    (ROOT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    try:
        protocol,membership,contexts=preregister();log('New E1 provenance and prior same-rod split registered')
        dump('RUN_STATUS.json',dict(status='RUNNING',main_release_sha256=MAIN_PIN))
        probes=fit_fixed_probes(contexts);log('Fixed diagnostic probes trained only on sealed source outer-OOF; all models frozen')
        metrics=score(contexts,membership,probes)
        for path,h in protocol['bound_artifact_sha256'].items():assert file_sha(path)==h
        for path,h in json.loads((ROOT/'GLOBAL_MODEL_AND_PROBE_FREEZE.json').read_text())['artifact_sha256'].items():assert file_sha(path)==h
        make_report(metrics,protocol)
        dump('RUN_STATUS.json',dict(status='COMPLETED',main_release_sha256=MAIN_PIN,main_model_fit_calls=0,
            diagnostic_probe_fit_calls=len(probes),new_same_rod_prediction_stage_files=16,
            independent_same_rod_whitening_rerun=True,full_new_E1_model_used=True,all_legal_executed=False,
            runtime_seconds=time.perf_counter()-START,verification='PASS',formal_whiteness_claim=False))
        files={str(p.relative_to(ROOT)):dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts and p.name!='DELIVERY_MANIFEST.json'}
        dump('DELIVERY_MANIFEST.json',dict(status='COMPLETE',files=files))
        with zipfile.ZipFile(ROOT/'NEW_E1_WHITENING_DELIVERY.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in ROOT.rglob('*'):
                if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
        log('COMPLETE; original E1 source/model/checkpoint hashes unchanged')
    except Exception as e:
        dump('RUN_STATUS.json',dict(status='FAILED_RETAINED',error=str(e),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
