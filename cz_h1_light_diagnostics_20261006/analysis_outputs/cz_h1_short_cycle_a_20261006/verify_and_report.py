"""Independent verification and concise complete report; no selection or refitting base."""
import json, time, zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import run_experiment as r

ROOT=r.ROOT
def csv(name):return pd.read_csv(ROOT/f'{name}.csv')
def table(columns,rows):
    return '| '+' | '.join(columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)+'\n'

def verify():
    freeze=json.loads((ROOT/'GLOBAL_SELECTION_FREEZE.json').read_text(encoding='utf-8'))
    for name,h in freeze['source_hashes'].items():assert r.sha(r.PREV/name)==h
    for name,h in freeze['model_hashes'].items():assert r.sha(ROOT/name)==h
    out=pd.read_parquet(ROOT/'SHORT_CYCLE_POINTWISE.parquet')
    membership=pd.read_parquet(r.PREV/'FROZEN_MEMBERSHIP.parquet')
    checks=[]
    for (arm,split),g in out.groupby(['arm','split']):
        prior=pd.read_parquet(r.PREV/('POINTWISE_VALIDATION.parquet' if split=='validation' else 'POINTWISE_TEST.parquet'))
        prior=prior[prior.arm==arm].reset_index(drop=True)
        H=freeze['recipes'][arm]['H'];keys=[]
        for kind in ['A0','A1','A2']:
            frame=g[g.A_model==kind].reset_index(drop=True);keys.append(list(zip(frame.entity,frame.origin)))
            assert keys[-1]==list(zip(prior.entity,prior.origin))
            assert np.array_equal(frame.base_prediction.to_numpy(),prior.base_prediction.to_numpy())
            assert np.array_equal(frame.base_residual.to_numpy(),prior.base_residual.to_numpy())
            assert np.max(abs(frame.final_prediction-frame.base_prediction-frame.A_prediction))<1e-12
            assert (frame.A_prediction.iloc[:H]==0).all() and not frame.A_state_available.iloc[:H].any()
            assert frame.A_state_available.iloc[H:].all()
            model=dict(np.load(ROOT/f'{arm}_{kind}.npz'))
            replay=r.infer(prior,kind,H,model)
            assert np.max(abs(replay.A_prediction-frame.A_prediction))<1e-12
            if kind=='A0':assert np.max(abs(frame.final_residual-prior.final_residual))<1e-12
            k=H+37;changed=prior.copy();changed.loc[k:,'base_residual']+=9.
            revised=r.infer(changed,kind,H,model)
            assert np.max(abs(revised.A_prediction.iloc[:k+1]-frame.A_prediction.iloc[:k+1]))==0
            assert np.max(abs(revised.A_prediction.iloc[k+1:]-frame.A_prediction.iloc[k+1:]))>1e-6
            # Current/future target cannot enter correction except through later mature base residual.
            changed_y=prior.copy();changed_y.loc[k:,'y_true']+=7.
            target_revised=r.infer(changed_y,kind,H,model)
            assert np.max(abs(target_revised.A_prediction-frame.A_prediction))==0
            a,_=r.acf(frame);z=frame.final_residual.to_numpy()-frame.final_residual.mean()
            direct=np.correlate(z,z,'full')[len(z)-1:len(z)+900]/np.dot(z,z)
            assert np.max(abs(a-direct))<1e-12
        assert keys[0]==keys[1]==keys[2]
        checks.append(dict(arm=arm,split=split,n=len(prior),common_scoring_rows=True,frozen_base_exact=True,A0_prior_output_exact=True,cold_start_rows=H,causal_residual_and_target_perturbations='PASS',direct_ACF='PASS'))
    # A1 has exactly 28 enhancements from within existing history; harmonic features add no new span.
    rng=np.random.default_rng(7);x=rng.normal(size=(500,96));X1=r.transform(x,'A1');X2=r.transform(x,'A2')
    assert X1.shape==(500,124) and X2.shape==(500,126)
    k=r.SHORT[0]
    assert np.array_equal(X1[:,96+14],x[:,k-1]-x[:,k])
    assert np.array_equal(X1[:,96+21],(np.sign(x[:,k-1])!=np.sign(x[:,k])).astype(float))
    lag=np.arange(1,97);weights=np.column_stack([np.cos(2*np.pi*.14*lag*2),np.sin(2*np.pi*.14*lag*2)])
    assert np.max(abs(X2[:,-2:]-x@weights))<1e-12
    y=rng.normal(size=500);z=r.transform(rng.normal(size=(90,96)),'A2')
    scaler=StandardScaler().fit(X2)
    grid=r.RidgeGrid(X2,y);solver_errors={}
    for alpha in r.ALPHAS:
        p1=r.predict(grid.state(alpha),z)
        p2=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(X2),y).predict(scaler.transform(z))
        solver_errors[str(alpha)]=float(np.max(abs(p1-p2)))
    assert max(solver_errors.values())<1e-6
    probe_stream=pd.read_parquet(ROOT/'PROBE_TRAIN_PREQUENTIAL.parquet')
    assert (probe_stream.global_origin>probe_stream.probe_A_fit_end).all()
    selected=csv('SHORT_CYCLE_MODEL_SELECTION')
    assert len(selected)==36*len(freeze['included_arms'])
    assert int(selected.selected_primary.sum())==3*len(freeze['included_arms'])
    r.write('VERIFICATION.json',dict(status='PASS',pointwise_checks=checks,prior_files_and_base_models_unchanged=True,source_row_ids_exactly_reused=True,all_model_and_probe_hashes_unchanged_since_freeze=True,independent_sklearn_Ridge_max_errors=solver_errors,short_block_feature_formulas_checked=True,harmonics_exact_linear_combinations_of_raw_lags=True,probe_training_prequential_only=True,base_refit_or_search=False,raw_controls_or_workbook_read=False))

def report():
    m=csv('SHORT_CYCLE_WHITENING_METRICS');selection=csv('SHORT_CYCLE_MODEL_SELECTION');p=csv('SHORT_CYCLE_RESIDUAL_PREDICTABILITY')
    freeze=json.loads((ROOT/'GLOBAL_SELECTION_FREEZE.json').read_text(encoding='utf-8'))
    decisions=json.loads((ROOT/'DECISIONS.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'RUN_SUMMARY.json').read_text(encoding='utf-8'))
    def row(arm,split,kind):return m[(m.arm==arm)&(m.split==split)&(m.A_model==kind)].iloc[0].to_dict()
    text='# CZ H1 7 秒短周期残差修正实验\n\n'
    text+='完成日期：2026-10-06（Asia/Shanghai）。分类：POST_HOC_DESCRIPTIVE_DIAGNOSTIC。旧test和周期峰已被查看，本轮继续使用原descriptive holdout，不能称作独立确认性检验。\n\n'
    text+='## 结论\n\n**NO_ADDITIONAL_SHORT_CYCLE_GAIN；推荐继续使用 A0。**\n\n'
    text+='该标签指未达到预注册的“明显压峰”成功标准，**不是观察到零增益**。Rod2 A1有小幅、可迁移的改善：validation/holdout注册频带能量下降约3.78%/3.70%，RMSE改善0.24%/0.32%，E160/E900均略降。但最大残余谱峰仍存在，部分lag绝对相关反而增加，bias也更大。A2相对A1几乎重合，没有额外主收益。\n\n'
    text+='按“Rod2成功后才复制”的条件，Rod2未达到validation复制门槛，**没有运行Rod1_Common4或Rod1_AllLegal**。未增加候选、改变频带或继续复杂调参。\n\n'
    text+='## 冻结复用与实验协议\n\n'
    text+='完整复用 `cz_h1_light_whitening_20261006` 的冻结membership、base history/alpha/scaler/model、train prequential OOF residual和validation/holdout base pointwise。没有打开原工作簿，没有读取控制变量，没有重训base或运行base history search，更没有K/C/W、Urysohn、recursive nested-OOF。所有源文件及模型hash在结束时保持一致。\n\n'
    text+='Rod2主比较固定 **H=96 samples=192s**；A0为旧alpha=100的原模型和原scaler，直接加载，逐行复现旧final prediction/residual。A1/A2的alpha只能从 `{1e-4,1e-2,1,100}` 依据tuning-validation冷启动RMSE选择，本轮均选100。H `{32,64,96}` 仅作小范围validation敏感性检查，三个模型的最佳敏感性H也均为96；这些结果没有用来替换主H，其他H没有打开holdout评分。\n\n'
    text+='A1表示在本轮训练前固定为：完整96个raw residual lags + 固定短周期lags `{1,2,3,4,5,7,8}` 上的四类变换（sign、absolute magnitude、first difference、zero crossing），共124维。固定短block已有raw列，不重复添加。Diff/zero-crossing需要k+1≤9，仍在相同96个成熟残差内；没有使用H+1或current residual。这一实现是“full raw history + very-small transformed short block”，没有额外尝试给全部96个lag添加非线性变换；结果只适用于此次冻结表示。\n\n'
    text+='A2=A1+两个预注册harmonic columns，126维；f0固定0.14Hz、dt=2s。`c=sum r(t-k)cos(2πf0kdt)`、`s=sum r(t-k)sin(2πf0kdt)`。它只作diagnostic，未追踪或拟合holdout峰频。\n\n'
    text+='**A2的两个feature是已存在raw lags的线性组合**，没有增加新的观测信息或线性函数空间；在标准化Ridge中它们通过冗余列改变正则化几何，才可能带来窄带先验收益。因此“harmonic未收益”不等于排除周期，也不等于0.14Hz没有可预测结构。A1中的diff也属于线性组合；sign/magnitude/zero-crossing才提供不同的非线性表示。\n\n'
    text+='A1/A2拟合只使用已有train OOF base residual，丢弃前H行不完整history，fit scaler仅在这些trainOOF特征上拟合。Validation/holdout分别清空state，首96行correction=0，特征先构造、当前base residual成熟后才加入state；实体或origin中断则再次清空。三个模型均评分相同原始rows：validation1883、holdout1626，主结果包含相同冷启动96行。\n\n'
    text+='固定probe仍是H32、alpha1。A0复用上一轮saved final-residual probe；A1/A2 probes只拟合另一次普通prequential fixed-recipe A prediction得到的训练final residual，没有用A in-sample误差。Probe的scaler同样train-only。没有重选probe H/alpha，也不按probe结果选择A。\n\n'
    text+='## 成功与复制门槛\n\n'
    text+='用户规定band明显下降、E160/E900不恶化、RMSE恶化不超过0.5%；本轮在训练前将“明显”操作化为**频带能量至少下降10%**。10%是本轮自行预注册的诊断阈值，并非用户指定数值；它不代表统计显著性。完整成功要求validation与holdout都满足三项。复制门槛只看Rod2 validation：A1或A2相对A0满足上述条件才依原样协议发展Rod1，避免让holdout控制候选/模型选择。\n\n'
    text+='A1在validation上band降幅仅3.78%，因此未过门槛；A2仅3.71%，同样未过。该判断在读取本轮holdout pointwise前已冻结。`GLOBAL_SELECTION_FREEZE.json` 保存三个主模型和probe hashes；随后仅打开一次旧POINTWISE_TEST.parquet用于descriptive scoring。\n\n'
    text+='## 预测与白化指标\n\n'
    rows=[]
    for split in ['validation','descriptive_holdout']:
        for kind in ['A0','A1','A2']:
            a=row('Rod2_Common4',split,kind)
            rows.append([split,kind,f"{a['RMSE']:.9f}",f"{a['MAE']:.9f}",f"{a['bias']:.9f}",f"{a['E160']:.6f}",f"{a['E900']:.6f}",f"{a['band_power']:.9g}",f"{a['nearest_0140_PSD']:.9g}"])
    text+=table(['split','model','RMSE','MAE','bias','E160','E900','band power','nearest0.14 PSD'],rows)
    text+='相对A0，正值表示指标下降/改善：\n\n'
    changes=[]
    for split in ['validation','descriptive_holdout']:
        base=row('Rod2_Common4',split,'A0')
        for kind in ['A1','A2']:
            child=row('Rod2_Common4',split,kind);_,ch=r.improvement(child,base)
            changes.append(dict(arm='Rod2_Common4',split=split,comparison=f'{kind}_vs_A0',**ch))
    rows=[[a['split'],a['comparison'],*[f"{a[key]:.3f}%" for key in ['RMSE','E160','E900','band_power','nearest_0140_PSD','dominant_PSD']]] for a in changes]
    text+=table(['split','comparison','RMSE下降','E160下降','E900下降','band下降','nearest-bin PSD下降','dominant PSD下降'],rows)
    text+='三模型dominant peak位置相同：validation0.137695Hz（7.262s），holdout0.138184Hz（7.237s）。Nearest-bin-to-0.14是0.140137Hz，不是上述dominant bin；holdout nearest-bin降低约13%不能替代整体band仅降3.7%的事实。Welch参数与旧报告相同：fs0.5、nperseg1024、noverlap512。尾段只有两个Welch averages，谱估计噪声和E900长lag不确定性依然存在。\n\n'
    text+='A1绝对bias由约1e-5/1.8e-5升至约2.25e-4/2.29e-4。RMSE仍略有改善，但增加的bias也是谨慎保留A0的理由；没有按此结果修改intercept或增加校准步骤。\n\n'
    text+='## lag3/4/7/14\n\n'
    rows=[]
    for split in ['validation','descriptive_holdout']:
        for kind in ['A0','A1','A2']:
            a=row('Rod2_Common4',split,kind);rows.append([split,kind,*[f"{a[f'rho_lag{k}']:.6f}" for k in [3,4,7,14]]])
    text+=table(['split','model','rho3','rho4','rho7','rho14'],rows)
    text+='Validation A1在四个指定lag的绝对相关均下降；holdout仅lag4从0.02965降到0.02636，lag3从0.00548升到0.00894、lag7从-0.01108变为-0.01905、lag14从-0.03640变为-0.03702，后面三项的绝对值增加。E160/E900整体仍下降，说明局部峰改善并不一致。Lag3/4对应6/8s，lag7/14对应14/28s，不等于约0.14Hz频谱周期。\n\n'
    text+='## 固定residual-predictability probe\n\n'
    rows=[]
    for _,a in p[p.support=='mature_history'].iterrows():rows.append([a['split'],a.A_model,int(a.n),f'{a.R2:.6f}',f'{a.probe_RMSE:.9f}',f'{a.RMSE_improvement_vs_zero_percent:.3f}%',f'{a.RMSE_improvement_vs_train_mean_percent:.3f}%'])
    text+=table(['split','model','mature rows','R²','probe RMSE','improvement vs zero','improvement vs train mean'],rows)
    text+='各probe在held-out均为负R²，表示固定AR probe没有可迁移的预测收益；A1/A2更负，不能机械解释为“更白”。分布偏移、bias及probe fit差异也影响R²。All-row cold-start结果见CSV，未用probe判断重选模型或证明innovation。\n\n'
    text+='## A2及history敏感性\n\n'
    rows=[]
    for split in ['validation','descriptive_holdout']:
        _,ch=r.improvement(row('Rod2_Common4',split,'A2'),row('Rod2_Common4',split,'A1'))
        changes.append(dict(arm='Rod2_Common4',split=split,comparison='A2_vs_A1',**ch))
        rows.append([split,*[f'{ch[k]:.4f}%' for k in ['RMSE','E160','E900','band_power','nearest_0140_PSD']]])
    text+=table(['split','A2 RMSE下降vsA1','E160下降','E900下降','band下降','nearest PSD下降'],rows)
    text+='A2比A1的RMSE略差约0.0038%，band energy略高约0.07–0.08%，没有额外有效压峰。Nearest0.14 bin虽然再小降约0.25–0.30%，但全band与综合指标没有收益；不裁定NARROWBAND_PRIOR_DIAGNOSTIC_ONLY。\n\n'
    selected=selection[selection.selected_alpha_for_history]
    text+=table(['model','H','selected alpha','validation RMSE','sensitivity best H'],[[a.A_model,int(a.H),f'{a.alpha_A:g}',f'{a.validation_RMSE:.9f}',bool(a.sensitivity_best_H)] for _,a in selected.iterrows()])
    text+='Sensitivity共3H×3models×4alphas=36个候选；不扩展history/alpha，不增加频率或feature搜索。\n\n'
    text+='## 八个问题的明确回答\n\n'
    text+='1. **A1小幅优于A0，但未达本轮明显改善门槛**；不是零增益，也没有主模型替换证据。\n2. **0.125–0.155Hz band power下降3.780%（validation）、3.696%（holdout）**。\n3. **E160/E900分别下降6.519%/4.672%（validation），2.897%/1.648%（holdout）**。\n4. **指定lag不全改善**：validation四项绝对值均降；holdout仅lag4降，其余三项绝对值略增。\n5. **RMSE没有恶化**：分别改善0.242%/0.317%，但bias增加。\n6. **A2无额外综合收益**；两个harmonic为raw lags的线性组合，在本Ridge设定下仅改变正则化权重。\n7. **小幅band/E/RMSE改善可从validation保持到holdout**，但幅度远小于初次A0对base的白化收益，局部lag和probe不支持强追加结论。\n8. **推荐A0；A1保存作此次小幅表示敏感性结果，A2仅作diagnostic**。没有运行Rod1，停止追加调参。\n\n'
    text+='## 解释边界、验证与交付\n\n'
    text+='可以描述为：**remaining short-period residual dependence is predictable from mature residual history**，且本轮小规模非线性short block还能提取少量增益。不能宣称已确认传感器、晶转或真实物理状态。前次机制诊断裁定仍是UNRESOLVED_RESIDUAL_STATE，0.01记录格点明确但binary event timing证据不足；本轮没有改变该裁定。\n\n'
    text+='独立验证通过：所有源文件/base/scaler hashes不变；A0逐行复现旧结果；三模型scoring row IDs、base prediction/residual完全相同；前H行均无correction；扰动当前/未来residual不影响当前/过去预测，当前/未来target不进入features；变换公式和harmonic线性关系正确；显式correlation与FFT ACF一致；独立sklearn SVD Ridge解与批量Gram解一致；probe训练块的fit_end严格早于scored origins。见 `VERIFICATION.json`。\n\n'
    elapsed=time.time()-r.ATTACH.stat().st_ctime
    text+=f"主实验runner wall time **{summary['runner_wall_seconds']:.2f}s**；从请求附件创建到报告生成约 **{elapsed/60:.2f}分钟**（起点为请求到达的近似代理），包含协议冻结、实现、核对与交付，未超过30分钟预算。\n\n"
    text+='必需6个结果文件和Rod2 ACF/PSD图齐全；另提供validation图、probe CSV、训练probe prequential pointwise、冻结模型NPZ、preregistration、hashes、决策JSON、代码与日志。复现时在analysis_outputs下新的同级空目录放置两份py；执行run_experiment.py和verify_and_report.py。旧目录有freeze时拒绝覆写。\n\n'
    text+='![Rod2 holdout ACF](Rod2_A0_A1_A2_ACF.png)\n\n![Rod2 holdout PSD](Rod2_A0_A1_A2_PSD.png)\n\n![Rod2 validation ACF](Rod2_A0_A1_A2_ACF_validation.png)\n\n![Rod2 validation PSD](Rod2_A0_A1_A2_PSD_validation.png)\n'
    (ROOT/'SHORT_CYCLE_A_REPORT_CN.md').write_text(text,encoding='utf-8')
    pd.DataFrame(changes).to_csv(ROOT/'SHORT_CYCLE_COMPARISONS.csv',index=False)
    r.write('DELIVERY_TIMING.json',dict(request_to_report_wall_seconds=elapsed,runner_wall_seconds=summary['runner_wall_seconds'],timezone='Asia/Shanghai'))
    r.write('DELIVERY_MANIFEST.json',dict(status='COMPLETE',files={p.name:dict(bytes=p.stat().st_size,sha256=r.sha(p)) for p in ROOT.iterdir() if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','CZ_H1_SHORT_CYCLE_A_DELIVERY.zip']}))
    with zipfile.ZipFile(ROOT/'CZ_H1_SHORT_CYCLE_A_DELIVERY.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix!='.zip':z.write(p,arcname=p.name)
    print('PASS: verified and packaged')

if __name__=='__main__':
    with threadpool_limits(limits=2):verify();report()
