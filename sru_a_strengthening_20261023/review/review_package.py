"""Independent review of V8/V9/V10. Never runs old-round packaging."""
from pathlib import Path
import json,hashlib,pickle,tarfile,datetime,sys,subprocess
import numpy as np,pandas as pd
BASE=Path('/root/autodl-tmp');ROOT=BASE/'PRISM_SRU_A_STRENGTHENING_REVIEW_20261023'
BRANCHES=[('PRISM_SRU_DYNAMIC_RESPONSE_A_20261023','sru_dynamic_response_a_20261023','V8 动态响应',12),('PRISM_SRU_CONDITIONAL_FEEDBACK_A_20261023','sru_conditional_feedback_a_20261023','V9 条件残差反馈',13),('PRISM_SRU_LATENT_INNOVATION_A_20261023','sru_latent_innovation_a_20261023','V10 潜在坐标创新',15)]
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rmse(p,col):return float(np.sqrt(np.mean((p.y-p[col])**2)))
def audit(obj,proofs):
    if isinstance(obj,list):
        for x in obj:audit(x,proofs)
    elif isinstance(obj,dict):
        for left,right in [('last_fit_label','first_score_dependency'),('initializer_last_label','first_score_recursive_dependency')]:
            if left in obj and right in obj:
                assert obj[left]<obj[right];proofs.add(hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest())
        if 'fit_rows' in obj and 'parameters' in obj:assert obj['fit_rows']>=max(200,20*obj['parameters'])
        for key,upper in [('kkt',1e-8),('relative_kkt',1e-8),('condition',1e14),('condition_number',1e14)]:
            if key in obj:assert 0<=obj[key]<=upper
        if all(k in obj for k in ['rows','parameters','kkt']):assert obj['rows']>=max(200,20*obj['parameters'])
        if all(k in obj for k in ['theta','P','fit_rows']):
            assert obj['fit_rows']>=max(200,20*len(obj['theta'])) and np.isfinite(obj['theta']).all() and np.linalg.norm(obj['theta'])<=1e6
            P=np.array(obj['P']);np.linalg.cholesky((P+P.T)*.5)
        for v in obj.values():audit(v,proofs)
def effect(y,a,b,skip=0):
    ea=(y[skip:]-a[skip:])**2;eb=(y[skip:]-b[skip:])**2;n=len(ea);rng=np.random.default_rng(20261023);r=[]
    for _ in range(2500):
        starts=rng.integers(0,n,size=(n+191)//192);ix=((starts[:,None]+np.arange(192))%n).ravel()[:n];r.append(float(100*(np.sqrt(eb[ix].mean()/ea[ix].mean())-1)))
    return dict(skip=skip,gain_percent=float(100*(np.sqrt(eb.mean()/ea.mean())-1)),CI95=np.quantile(r,[.025,.975]).tolist(),block=192)
def review():
    cases=[];proofs=set();counts=[];ledgers=[];total=0
    for remote,local,label,tests in BRANCHES:
        p=BASE/remote;reg=read(p/'REGISTRATION_BEFORE_FITS.json');queue=read(p/'QUEUE_COMPLETED.json')
        assert all(q['exit_code']==0 for v in queue.values() for q in v)
        assert f'{tests} passed' in (p/'CORE_TESTS_ATTEMPT_1.log').read_text()
        assert all(sha(p/k)==v for k,v in reg['code_sha256'].items()) and sha(p/'TRAINING_DATA.npz')==reg['training_sha256'] and sha(p/'DATA_VIEW_BEFORE_FITS.json')==reg['view_sha256']
        view=read(p/'DATA_VIEW_BEFORE_FITS.json')
        for v in view['records'].values():assert all(sha(k)==h for k,h in v['files'].items())
        for ref in view['existing_final_ledger_references']:assert sha(ref['path'])==ref['sha256'];ledgers.append(ref)
        assert sha(p/'rls_core.py')==sha(BASE/'PRISM_DELAYED_LABEL_SOFTSENSOR_20261023/rls_core.py')
        replay=ROOT/(local+'_CAUSAL_REPLAY.json')
        if not replay.exists():subprocess.run([sys.executable,str(ROOT/'causal_replay.py'),str(p),str(replay)],check=True)
        assert read(replay)['status']=='PASSED'
        calls=0;initial=len(proofs)
        for case in reg['cases']:
            dest=p/'cases'/case;r=read(dest/'DEVELOPMENT_RESULT.json');pred=pd.read_parquet(dest/'DEVELOPMENT_predictions.parquet');frozen=read(dest/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json')
            assert all(sha(dest/k)==v for k,v in frozen['files'].items()) and r['models_unchanged'] and r['no_offline_refit'] and not r['source_final_numeric_read'] and len(pred)==r['rows_scored']==2016
            for row in r['rows']:assert np.isclose(row['RMSE'],rmse(pred,row['variant']),rtol=1e-13,atol=1e-14)
            for k,v in r['baselines'].items():assert np.isclose(v,rmse(pred,k),rtol=1e-13,atol=1e-14)
            for e in r['effects']:
                check=effect(pred.y.to_numpy(),pred.full.to_numpy(),pred[e['control']].to_numpy(),e['skip'])
                assert np.isclose(check['gain_percent'],e['gain_percent'],rtol=1e-12,atol=1e-11) and np.allclose(check['CI95'],e['CI95'],rtol=1e-12,atol=1e-11)
            for name in reg['variants']:
                t=read(dest/(name+'_TRAINING.json'));audit(t,proofs);calls+=sum(x['A'] is not None for x in t['joint_candidates'])
                assert t['joint_candidates'][t['joint_selected_index']]['selection_MSE']==min(x['selection_MSE'] for x in t['joint_candidates'])
            b=pickle.loads((dest/'BASELINES_FROZEN.pkl').read_bytes());audit(b,proofs);calls+=sum(x['A'] is not None for x in b['ADAPTIVE_ARX']['joint_candidates'])
            if remote!=BRANCHES[0][0]:
                rec=read(dest/'BASELINE_REUSE_RECEIPT.json');assert sha(rec['source'])==rec['source_sha256'];assert len(rec['unchanged_comparators'])==15
                origin=pd.read_parquet(BASE/BRANCHES[0][0]/'cases'/case/'DEVELOPMENT_predictions.parquet')
                for k in rec['unchanged_comparators']+['Persistence']:np.testing.assert_array_equal(origin[k],pred[k])
            positive=all(e['gain_percent']>0 and e['CI95'][0]>0 for e in r['effects']);below=rmse(pred,'full')<min(r['baselines'].values());assert r['goal_achieved']==bool(positive and below)
            assert all(all(v for v in q.values() if isinstance(v,bool)) for q in r['QA'])
            supplemental={};prior_refs={};aligned={}
            for root,tag in [('PRISM_SRU_ISSUED_RESPONSE_A_20261023','V6')]+[(a,f'V{i+8}') for i,(a,*_) in enumerate(BRANCHES)]:
                path=BASE/root/'cases'/case/'DEVELOPMENT_predictions.parquet';q=pd.read_parquet(path)
                np.testing.assert_array_equal(q.origin,pred.origin);np.testing.assert_array_equal(q.y,pred.y)
                supplemental[tag+'_ADAPTIVE_ARX']=rmse(q,'ADAPTIVE_ARX');aligned[tag+'_ADAPTIVE_ARX']=q.ADAPTIVE_ARX.to_numpy()
                if root!=remote:prior_refs[tag+'_PRISM_full']=rmse(q,'full')
            combined={**r['baselines'],**supplemental};best=min(combined,key=combined.get);bestvec=pred[best].to_numpy() if best in pred else aligned[best]
            stronger=effect(pred.y.to_numpy(),pred.full.to_numpy(),bestvec)
            quarters=[]
            for qi,ix in enumerate(np.array_split(np.arange(len(pred)),4)):
                q=pred.iloc[ix];quarters.append(dict(quarter=qi+1,first_origin=int(q.origin.min()),last_origin=int(q.origin.max()),full_RMSE=rmse(q,'full'),no_A_RMSE=rmse(q,'no_A'),no_A_gain_percent=100*(rmse(q,'no_A')/rmse(q,'full')-1)))
            static=rmse(pred,'full_A_frozen_updates');full=pickle.loads((dest/'full_FROZEN.pkl').read_bytes())
            for e in r['frozen_update_control']:
                check=effect(pred.y.to_numpy(),pred.full.to_numpy(),pred.full_A_frozen_updates.to_numpy(),e['skip']);assert np.allclose(check['CI95'],e['CI95'],atol=1e-11)
            cases.append(dict(branch=local,label=label,full_A_selected=full['A']['selected'],full_A_admitted=full['A']['admitted'],full_KCW=full['upper']['selected'],A_coordinate=full['A'].get('coordinate'),frozen_A_update_RMSE=static,online_update_extra_RMSE_reduction_percent=100*(1-rmse(pred,'full')/static),report_A_RMSE=rmse(pred,'report_A'),fixed_four_quarter_diagnostic=quarters,supplemental_frozen_baselines=supplemental,prior_PRISM_reference_models=prior_refs,strongest_retained_baseline=best,strongest_retained_baseline_RMSE=combined[best],strongest_retained_baseline_comparison=stronger,full_RMSE_reduction_vs_strongest_percent=100*(1-rmse(pred,'full')/combined[best]),retained_baseline_goal_achieved=bool(positive and rmse(pred,'full')<min(combined.values())),**r))
        total+=calls;counts.append(dict(branch=local,settings=3,final_configurations=15,selected_main_A=12,selected_baseline_A=3,candidate_A_searches=calls,additional_distinct_support_objects=len(proofs)-initial))
    successes=[dict(branch=c['branch'],case=c['case']) for c in cases if c['retained_baseline_goal_achieved']]
    result=dict(status='COMPLETED',branches=counts,settings=9,final_configurations=45,selected_main_A=36,selected_baseline_A=9,candidate_A_searches=total,distinct_support_proof_objects=len(proofs),new_physical_sources=0,new_physical_records=0,source_final_numeric_read=False,cases=cases,joint_development_successes=successes,scope='Repeated adaptive SRU development, descriptive intervals not search-adjusted; no independent final confirmation.',causal_replay_chains=54,nonadaptive_baseline_reuse_checked_bitwise=True,previous_strong_baselines_retained=True)
    assert len(cases)==9 and len(successes)==2
    dump(ROOT/'COMBINED_REVIEW.json',result);dump(ROOT/'SHARED_FINAL_LEDGER_AUDIT.json',ledgers)
    report(result);print('A_STRENGTHENING_REVIEW_COMPLETED',total,successes,flush=True)
def report(r):
    lines=['# SRU：条件残差反馈增强 A','',
    'V9在H2S同期估计、标签延迟4和16步两个设置中，KC/W/A独立删除效应及全段、固定启动描述性区间均正，完整模型优于17项登记基线和此前已得到的更强冻结基线。这是反复使用SRU开发段后的探索结果，尚无独立保留段确认。本轮最终尾段未读取。','',
    f"本轮3分支、9设置、45条最终配置、36个最终主模型A政策、9个ADAPTIVE_ARX A政策，实际{r['candidate_A_searches']}次候选A搜索。无新增物理来源或记录；原54对支持修正及旧结果不变。",'',
    '任务是用当前x(t)、历史过程量和已成熟y估计y(t)，不是预测未来4/16步。4/8/16步是实验标签延迟，未核实为现场仪表时延。原10081点中前6048训练，6048–8063开发，8064–10080保留。本轮仅加载前8064点，旧源不能称新锁箱。','',
    'V8增加发行响应q(t)及其1/4/8/16/32步差分和q×差分。V9再加入一条成熟残差r(t-delay)，及r×[q,Δq4,Δq32]，使反馈系数随当前工况和响应变化。V10另测输出链接潜在坐标创新，训练风险仍按物理单位评分。均保留标准化ridge100初始化和成熟stored-phi RLS核心，属于明确结构改进，不是未修改v0.6，也不是阻尼正余弦+AR版本。','',
    '两个V9成功设置均选择QC32_RLS、最近512条合法初始化、forget=.999、gain=1。延迟4的KC为有限Urysohn过程响应，延迟16为线性过程响应；两者W为带符号平方根逆链接。不能据此宣称原非线性KC处处有效。','',
    '![开发候选与删除效应](SRU_A_strengthening.png)','',
    '|标签延迟|完整RMSE|保留最强基线RMSE|完整降低%|去KC增加%|去W增加%|去A增加%|','|---|---:|---:|---:|---:|---:|---:|']
    good=[c for c in r['cases'] if c['retained_baseline_goal_achieved']]
    for c in good:
        e={x['control']:x for x in c['effects'] if not x['skip']};lines.append(f"|{c['label_delay']}|{c['rows'][0]['RMSE']:.9f}|{c['strongest_retained_baseline_RMSE']:.9f}|{c['full_RMSE_reduction_vs_strongest_percent']:.3f}|"+'|'.join(f"{e[k]['gain_percent']:.3f}" for k in ['no_KC','no_W','no_A'])+'|')
    lines+=['','增加%=100×(删除后RMSE/完整RMSE−1)。删除后均独立重训及重新选择。','','|延迟|对照|全段95%描述性区间|固定启动95%描述性区间|','|---|---|---|---|']
    for c in good:
        for name in ['no_KC','no_W','no_A']:
            e=[x for x in c['effects'] if x['control']==name];lines.append(f"|{c['label_delay']}|{name}|{e[0]['CI95']}|{e[1]['CI95']}|")
    lines+=['','2500次配对循环块重采样，块长192；固定启动去掉32+delay−1点。开发段多轮复用，区间未校正搜索次数；不能当独立验证。对最强保留基线的配对区间另存JSON，均值领先不自动代表显著。','','|延迟|原报告A对照RMSE|冻结A更新RMSE|在线新A RMSE|在线更新额外降低%|','|---|---:|---:|---:|---:|']
    for c in good:lines.append(f"|{c['label_delay']}|{c['report_A_RMSE']:.9f}|{c['frozen_A_update_RMSE']:.9f}|{c['rows'][0]['RMSE']:.9f}|{c['online_update_extra_RMSE_reduction_percent']:.3f}|")
    lines+=['','冻结更新仍读取成熟残差特征，仅冻结RLS参数。去A收益混合初始化校准、条件反馈与参数适应，不全是在线更新。延迟16时，冻结更新相对在线完整模型的误差增加区间为[1.876%,34.831%]，固定启动区间也为正；延迟4此区间跨零，不能同样声称稳定的在线更新增益。','','|延迟|固定四等分去A增加%（1/2/3/4）|','|---|---|']
    for c in good:lines.append(f"|{c['label_delay']}|"+' / '.join(f"{q['no_A_gain_percent']:.3f}" for q in c['fixed_four_quarter_diagnostic'])+'|')
    lines+=['','四等分仅作固定诊断，不用来选窗或改变总评分。两个候选都四段正，不等于四次独立验证。最强基线相对完整模型的误差增加区间：延迟4为[-0.350%,24.154%]，延迟16为[-18.384%,23.918%]；两者跨零，所以只能称当前开发均值更优。','','|分支|延迟|完整RMSE|去A增加%|胜保留强基线且三块区间均正|','|---|---:|---:|---:|---|']
    for c in r['cases']:
        e=next(e for e in c['effects'] if e['control']=='no_A' and not e['skip']);lines.append(f"|{c['label']}|{c['label_delay']}|{c['rows'][0]['RMSE']:.9f}|{e['gain_percent']:.3f}|{c['retained_baseline_goal_achieved']}|")
    lines+=['','V8延迟16的A有正作用但落后ADAPTIVE_ARX；延迟4/8未改善。V10潜在坐标延迟4较差、延迟8/16回选物理坐标且等于V8，不算独立复现。V9延迟8也失败。全部阴性保留。','',
    '基线包含Persistence、OWN、ARX、逆链接ARX、HGB、SVR、在线ARX及128目标历史/40过程历史扩展，覆盖A的q32及上游7步跨度。ADAPTIVE_ARX获得相同新A家族、折和预算。V9/V10复用V8的16项非ADAPTIVE冻结基线，哈希和预测逐值一致，不能计为重新拟合。此前V6及本轮各分支ADAPTIVE_ARX额外保留：延迟4最强为V6，延迟16为V8。历史PRISM完整模型另外列参考。','',
    f"独立复核{r['distinct_support_proof_objects']}个不同支持证明对象、全保存预测误差及重采样区间、代码/源/冻结模型哈希和数值证书。54条主/消融/ADAPTIVE链逐段重放，在9个前缀长度下遮蔽所有未成熟y和未来X，仍重现保存预测；初始化、特征和更新均符合标签成熟限制，无额外拟合。分支检查12/13/15通过（含共用项，不相加作独立新增检查）。",'',
    '完整内外A支持隔离、至少200条且每系数20条、数值门槛未放宽。设计限制：noA保留阶段分区，重选上游但未把A校准标签追加给W拟合，阶段预算影响消融解释。需结合全训练预算强基线和冻结更新控制。当前结果只限SRU H2S和两个假定延迟。保留段在来源暴露审计和选择冻结前保持封存。','',
    '[完整数值与区间](COMBINED_REVIEW.json)；[来源账本快照](SHARED_FINAL_LEDGER_AUDIT.json)。']
    (ROOT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def package():
    r=read(ROOT/'COMBINED_REVIEW.json');archive=ROOT/'SRU_A_STRENGTHENING_20261023.tar.gz';assert not archive.exists();files=[]
    for folder,label in [(BASE/a,b) for a,b,*_ in BRANCHES]+[(ROOT,'sru_a_strengthening_review_20261023')]:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix=='.pyc' or p.name.endswith('.tar.gz') or p.name in ['DEVELOPMENT_PACKAGE_MANIFEST.json','SERVER_PACKAGE_VERIFIED.json']:continue
            assert not p.is_symlink();files.append((p,label+'/'+p.relative_to(folder).as_posix()))
    manifest={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,n in files};mp=ROOT/'DEVELOPMENT_PACKAGE_MANIFEST.json';dump(mp,manifest);mn='sru_a_strengthening_review_20261023/'+mp.name
    with tarfile.open(archive,'w:gz') as t:
        for p,n in files+[(mp,mn)]:t.add(p,arcname=n,recursive=False)
    with tarfile.open(archive,'r:gz') as t:
        assert set(t.getnames())==set(manifest)|{mn}
        for n,v in manifest.items():
            b=t.extractfile(n).read();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
    receipt=dict(status='VERIFIED',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(mp),members=len(manifest),all_members_sha256_verified=True,final_configurations=45,candidate_A_searches=r['candidate_A_searches'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_final_numeric_read=False,server='connect.westb.seetacloud.com:50905');dump(ROOT/'SERVER_PACKAGE_VERIFIED.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':review() if sys.argv[1]=='review' else package()
