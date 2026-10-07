"""Verify frozen artifacts, independently replay inference, then report (no tuning)."""
import json, time, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import audit_and_data as d
import run_experiment as r

ROOT=d.ROOT

def verify():
    freeze=json.loads((ROOT/'GLOBAL_SELECTION_FREEZE.json').read_text(encoding='utf-8'))
    for name,h in freeze['artifacts'].items(): assert r.digest(ROOT/name)==h
    assert r.digest(ROOT/'ROD1_ALL_LEGAL_VARIABLES.json')==freeze['variable_list_sha256']
    assert r.digest(ROOT/'FROZEN_MEMBERSHIP.parquet')==freeze['membership_sha256']
    mem=pd.read_parquet(ROOT/'FROZEN_MEMBERSHIP.parquet')
    original=pd.read_parquet(ROOT/'ORIGINAL_SOURCE_MEMBERSHIP.parquet')
    assert set(zip(mem[mem.split=='train'].entity,mem[mem.split=='train'].origin))==set(zip(original[original.original_split=='train'].entity,original[original.original_split=='train'].origin))
    for rod,g in mem.groupby('rod'):
        assert len(g[g.split=='purged'])==257
        val=g[g.split=='validation'];test=g[g.split=='test']
        assert set(val.entity)==set(test.entity)
        assert test.origin.min()-val.origin.max()==258
        assert len(g[['entity','origin']].drop_duplicates())==len(g)
    oof=pd.read_parquet(ROOT/'BASE_OOF_POINTWISE.parquet')
    assert (oof.global_origin>oof.fit_end).all()
    assert np.max(abs(oof.y_true-oof.y_base_pred-oof.r_base))<1e-14
    _,segs=d.load()
    tests=[]
    for split in ['validation','test']:
        frame=pd.read_parquet(ROOT/f'POINTWISE_{split.upper()}.parquet')
        for arm,g0 in frame.groupby('arm',sort=False):
            g=g0.reset_index(drop=True);c=freeze['arms'][arm];H=c['H'];L=c['L']
            model=dict(np.load(ROOT/f'{arm}_A.npz'))
            correction,available=r.online(g,model,H)
            assert (~g.A_state_available.iloc[:H]).all() and g.A_state_available.iloc[H:].all()
            assert np.array_equal(available,g.A_state_available)
            assert np.max(abs(g.final_prediction-g.base_prediction-correction))<1e-14
            assert np.max(abs(g.y_true-g.final_prediction-g.final_residual))<1e-14
            # Future/current residual perturbation must not affect predictions through origin k.
            changed=g.copy();k=H+30
            changed.loc[k:,'base_residual']+=10.
            revised,_=r.online(changed,model,H)
            assert np.max(abs(correction[:k+1]-revised[:k+1]))==0
            assert np.max(abs(correction[k+1:]-revised[k+1:]))>1e-5
            a,_=r.acf_segmented(g,'final_residual')
            z=g.final_residual.to_numpy()-g.final_residual.mean()
            independently=np.correlate(z,z,'full')[len(z)-1:len(z)+900]/np.dot(z,z)
            assert np.max(abs(a-independently))<1e-12
            # Replay saved base model against raw controls; mutate a current control row.
            pca=json.loads((ROOT/f'{arm}_PCA.json').read_text(encoding='utf-8'))
            ss=d._apply_joint_lift(segs[c['rod']],pca)
            by={s.segment_id:s.frame[c['variables']].to_numpy(float) for s in ss}
            x=np.vstack([by[e][int(t)-L:int(t)][::-1].T.reshape(-1) for e,t in zip(g.entity,g.origin)])
            bmodel=dict(np.load(ROOT/f'{arm}_BASE.npz'))
            pred=r.predict(bmodel,x)
            assert np.max(abs(pred-g.base_prediction))<1e-12
            target_t=int(g.origin.iloc[k]);entity=g.entity.iloc[k]
            by[entity][target_t,0]+=100.
            x2=np.vstack([by[e][int(t)-L:int(t)][::-1].T.reshape(-1) for e,t in zip(g.entity,g.origin)])
            pred2=r.predict(bmodel,x2)
            assert np.max(abs(pred[:k+1]-pred2[:k+1]))==0
            assert abs(pred2[k+1]-pred[k+1])>1e-8
            tests.append(dict(arm=arm,split=split,pointwise_rows=len(g),cold_start_rows=H,model_replay_max_error=float(np.max(abs(pred-g.base_prediction))),ACF_direct_comparison_error=float(np.max(abs(a-independently))),future_control_perturbation_pass=True,current_and_future_residual_perturbation_pass=True))
    # Independent solver check includes singular and constant features.
    rng=np.random.default_rng(3);x=rng.normal(size=(500,30));x[:,2]=1.;x[:,4]=x[:,3];y=rng.normal(size=500)
    z=rng.normal(size=(80,30));z[:,2]=1.;z[:,4]=z[:,3]
    grid=r.RidgeGrid(x,y);scaler=StandardScaler().fit(x)
    errs={str(a):float(np.max(abs(r.predict(grid.state(a),z)-Ridge(alpha=a,solver='svd').fit(scaler.transform(x),y).predict(scaler.transform(z))))) for a in r.ALPHAS}
    assert max(errs.values())<1e-6
    assert len(pd.read_csv(ROOT/'BASE_MODEL_SELECTION.csv'))==90
    assert len(pd.read_csv(ROOT/'A_MODEL_SELECTION.csv'))==90
    d.write_json('VERIFICATION.json',dict(status='PASS',checks=tests,independent_sklearn_Ridge_errors=errs,train_membership_unchanged=True,OOF_fit_end_before_score=True,artifacts_unchanged_since_freeze=True,original_workbook_sha256=r.digest(d.RAW)))

def table(cols,rows):
    return '| '+' | '.join(cols)+' |\n|'+ '|'.join(['---']*len(cols))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'

def report():
    summary=json.loads((ROOT/'RUN_SUMMARY.json').read_text(encoding='utf-8'))
    recipes=summary['recipes']
    m=pd.read_csv(ROOT/'METRICS_ALL_ARMS.csv');w=pd.read_csv(ROOT/'WHITENING_METRICS.csv');s=pd.read_csv(ROOT/'SPECTRAL_SUMMARY.csv');p=pd.read_csv(ROOT/'RESIDUAL_PREDICTABILITY.csv');peaks=pd.read_csv(ROOT/'TOP_PEAKS.csv')
    def mm(arm,split,stage):return m[(m.arm==arm)&(m.split==split)&(m.stage==stage)].iloc[0]
    def ww(arm,split):return w[(w.arm==arm)&(w.split==split)].iloc[0]
    def ss(arm,split,stage):return s[(s.arm==arm)&(s.split==split)&(s.stage==stage)].iloc[0]
    def pp(arm,split,stage):return p[(p.arm==arm)&(p.split==split)&(p.stage==stage)&(p.support=='mature_history')].iloc[0]
    text='# CZ H1 轻量残差白化实验交付报告\n\n'
    text+='分类：**POST_HOC_RESIDUAL_WHITENING_DIAGNOSTIC**。完成日期：2026-10-06（Asia/Shanghai）。两棒既往 test 已被观察；本轮时间尾段也是同一历史数据，不能称为新的确认性证据。\n\n'
    text+='## 主要结论\n\n'
    text+='普通 raw-history Ridge 后，三个 arm 都仍存在短周期残差相关。仅用成熟的过去残差作 correction，三者 RMSE、E160、E900、主要重复峰和固定 AR probe predictability 均改善。Rod2 改善更稳定；Rod1 validation 上的强降幅到 test 尾段有所回落，仍保留约 7 秒频谱结构。\n\n'
    text+='Rod1 的额外合法变量**没有一致改善 base prediction**。AllLegal 在 validation/test 的 base RMSE 分别增加约 1.51%/1.96%；其 final test E160/E900 较 Common4 更低，但 validation 两项反而略高。这不足以认定遗漏 measured controls 是 Rod1 残余相关性的主因。结论更接近 **the remaining Rod1 dependence is unlikely to be explained by the additional measured controls under this simple history-window representation.** Regime/state dependence 是相容解释，未做机制识别，不能宣称已被证明。\n\n'
    text+='## 数据、变量和划分\n\n'
    text+=f'原始文件：`实验数据1-张.xlsx`，SHA-256 `{r.digest(d.RAW)}`。Rod1 20103 行，Rod2 20627 行；采样周期沿用正式代码的 2 s，无时间戳独立校验。目标严格为 `D[t]-D[t-1]`，预测输入最晚 `t-1`。原始单位未提供，RMSE 使用原数值单位。分段沿用 Rod1 334/688/734，Rod2 745，不跨段建窗口。\n\n'
    text+='原有正式 H1 是 cross-rod transfer，没有同棒独立 test；用户在本轮明确授权确定性修订：原 source-train membership 全部保留，原 source-validation 前半用于选参，中间 257 行隔离，余下尾段 descriptive test。该修订在训练前保存 `FROZEN_MEMBERSHIP.parquet` 和 `EXPERIMENT_PREREGISTRATION.json`，没有随机划分。旧 cross-rod test 不作为本轮 held-out 评分集。\n\n'
    mem=pd.read_parquet(ROOT/'FROZEN_MEMBERSHIP.parquet')
    text+=table(['rod','train','validation','purged','test'],[[rod,*[int(((mem.rod==rod)&(mem.split==split)).sum()) for split in ['train','validation','purged','test']]] for rod in ['Rod_1','Rod_2']])
    text+='Rod1-only 原始字段为氩气流量设定、晶体长度、炉压。氩气流量设定为常数，虽保留于冻结清单，实际提供零方差信息。加热元件温度是两棒都有、Common4 未含的测量 context。AllLegal 冻结为 10 个通道：Common4 + 原晶升/埚升速度 + 温度 + 三个 Rod1-only 字段。\n\n'
    text+='字段合法性沿用项目 record_time 约定：只使用已经记录的过去行。原工作簿没有独立延迟/离线回填元数据；特别是晶体长度按当时已记录的工艺状态处理，此可用性是假设而非外部认证。没有从直径构造任何 predictor。审计文件标明这一点。\n\n'
    text+='joint_lift 原样提取正式 `cz_l256_nowcast.py` 的数据函数：仅在当前 fit 的 L 步历史行去重集合上标准化两种升速、取 PC1、固定晶升 loading 非负；每个 train-CV / OOF fit 重新拟合。候选 PCA 的 history extent 随 L 变化，复用 composite 定义，不复用全数据拟合状态。\n\n'
    text+='## 选择、OOF 和在线使用\n\n'
    text+='base 每 arm 6 histories × 5 alpha；每 L 通过 4 个 expanding prequential score blocks 选 alpha（前 1/5 为初始 fit，其后四段依次评分）。再用 validation RMSE 选 L；距最小值 0.1% 内取较短 L。每个 score block 的 target 不进入自己的 fit，索引有严格前后顺序。预处理仅拟合相应 training rows。\n\n'
    text+='确定 recipe 后，同一 4-block prequential 过程只 refit 参数，得到 base OOF residual。A 每 arm 6 H × 5 alpha，全部仅拟合完整过去 H 个 OOF residual，用在线冷启动 validation RMSE 选择 H/alpha；0.1% 近似并列优先短 H，再较强 alpha。A 不读取 raw controls，不使用 in-sample base residual，不填不存在的 residual。\n\n'
    text+='validation/test 每个连续段 residual state 清空，前 H 行 correction=0。每个 t 先预测，再在 y_t 成熟后加入 base residual，下一 t 才可用。缺行也清空。base/A 的 test 均加载冻结的 source-train / trainOOF 模型，未用 validation refit；两棒 test target 构造各一次，三 arm 配置及 probes 全冻结后才评分。Schema audit 按要求观察全表字段统计，但 test target/metrics 未用于选择。\n\n'
    text+='这是普通 prequential OOF，base recipe 已经条件于开发期选择；没有 recursive nested-OOF，也不声称该条件选择等于完整嵌套验证。\n\n'
    text+=table(['arm','L samples / seconds','base alpha','A H samples / seconds','A alpha'],[[a,f"{c['L']} / {c['L']*2}",c['alpha_base'],f"{c['H']} / {c['H']*2}",c['alpha_A']] for a,c in recipes.items()])
    text+='Rod1_Common4 最小 validation RMSE 的 L 是 128，但 L64 的损失在预注册 0.1% 并列范围内，故最终 L64。\n\n'
    text+='## 预测与白化结果\n\n以下为所有 scored rows，包含 A 冷启动。\n\n'
    rows=[]
    for split in ['validation','test']:
        for arm in recipes:
            b=mm(arm,split,'base');f=mm(arm,split,'final');q=ww(arm,split)
            rows.append([arm,split,f'{b.RMSE:.8f} → {f.RMSE:.8f}',f'{100*(1-f.RMSE/b.RMSE):.2f}%',f'{q.E160_base:.4f} → {q.E160_final:.4f}',f'{q.E160_reduction_percent:.2f}%',f'{q.E900_base:.4f} → {q.E900_final:.4f}',f'{q.E900_reduction_percent:.2f}%'])
    text+=table(['arm','split','RMSE base → final','RMSE降幅','E160 base → final','E160降幅','E900 base → final','E900降幅'],rows)
    text+='MAE、bias 及 raw target 层见 `METRICS_ALL_ARMS.csv`。Raw 层的 RMSE/MAE 是相对零变化 baseline，bias 是目标均值。最终三 arm 在 validation/test 的 MAE 与绝对 bias 均比各自 base 更低。\n\n'
    text+='Rod1 AllLegal 相对 Common4 的变化（正数为指标下降、负数为指标上升）：\n\n'
    comp=[]
    for split in ['validation','test']:
        for stage in ['base','final']:
            q1=ww('Rod1_Common4',split);q2=ww('Rod1_AllLegal',split)
            comp.append([split,stage,f"{100*(1-mm('Rod1_AllLegal',split,stage).RMSE/mm('Rod1_Common4',split,stage).RMSE):.2f}%",f"{100*(1-q2['E160_'+stage]/q1['E160_'+stage]):.2f}%",f"{100*(1-q2['E900_'+stage]/q1['E900_'+stage]):.2f}%"])
    text+=table(['split','stage','RMSE improvement','E160 improvement','E900 improvement'],comp)
    text+='## 重复峰和频谱\n\nACF lag7 是 14 s，lag14 是 28 s；约 0.14 Hz 的 Welch 峰代表约 7 s，二者没有混用。\n\n'
    rows=[]
    for split in ['validation','test']:
        for arm in recipes:
            q=ww(arm,split)
            rows.append([arm,split,*[f"{q[f'lag{k}_base']:.4f} → {q[f'lag{k}_final']:.4f}" for k in [1,7,14]]])
    text+=table(['arm','split','lag1 base → final','lag7 base → final','lag14 base → final'],rows)
    rows=[]
    for split in ['validation','test']:
        for arm in recipes:
            b=ss(arm,split,'base');f=ss(arm,split,'final')
            rows.append([arm,split,f'{b.dominant_frequency_Hz:.6f} → {f.dominant_frequency_Hz:.6f}',f'{b.dominant_period_seconds:.3f} → {f.dominant_period_seconds:.3f}',f'{100*(1-f.dominant_PSD/b.dominant_PSD):.1f}%',f'{100*(1-f.seven_second_band_energy/b.seven_second_band_energy):.1f}%',f'{b.low_frequency_energy:.3g} → {f.low_frequency_energy:.3g}'])
    text+=table(['arm','split','dominant Hz','period s','最大峰 PSD 降幅（峰位置可变）','0.125–0.155Hz能量降幅','低频(0,0.02]Hz能量'],rows)
    text+='~0.14 Hz 附近结构减弱，但 final 的最强峰仍在该频带。Welch 固定 fs=0.5Hz、nperseg=1024、noverlap=512，Nyquist=0.25Hz。各段仅有两个 Welch 窗口，频谱峰位置和幅度的不确定性不可忽视；没有根据 test 改变窗宽或频带。\n\n'
    text+='最强 ACF local peaks（按绝对值排序，test，展示前三个；更多及 PSD peaks 见 `TOP_PEAKS.csv`）：\n\n'
    text+=table(['arm','stage','rank','lag samples','lag seconds','ACF'],[[x.arm,x.stage,int(x['rank']),int(x.lag_samples),int(x.lag_seconds),f'{x.value:.4f}'] for _,x in peaks[(peaks.split=='test')&(peaks.peak_type=='ACF')&(peaks['rank']<=3)].iterrows()])
    text+='## Held-out residual predictability\n\n'
    text+='预注册固定 Ridge AR probe：H=32、alpha=1。Base probe 只拟合 base OOF；final probe 只拟合第二次普通 prequential A prediction 得到的 final residual，避免使用 A 拟合内误差。第二次 prequential 每个 score block 也清空 state；recipe 固定，不再搜索。Probe evaluation 同样为空 state，输入只为该 residual 的过去。下表为 full probe-history 行，all-row 冷启动结果也在 CSV。R² 使用 held-out target centered SST；mean baseline 使用训练残差均值。负 R² 不表示证明独立，只表示该固定 probe 没有可迁移增益。\n\n'
    rows=[]
    for split in ['validation','test']:
        for arm in recipes:
            b=pp(arm,split,'base');f=pp(arm,split,'final')
            rows.append([arm,split,f'{b.R2:.4f} → {f.R2:.4f}',f'{b.RMSE_improvement_vs_zero_percent:.2f}% → {f.RMSE_improvement_vs_zero_percent:.2f}%',f'{b.RMSE_improvement_vs_train_mean_percent:.2f}% → {f.RMSE_improvement_vs_train_mean_percent:.2f}%'])
    text+=table(['arm','split','probe R² base → final','RMSE改善vs zero base → final','RMSE改善vs train mean base → final'],rows)
    text+='## 16 个问题的明确回答\n\n'
    text+='1. **Rod1_Common4 最终 L=64 samples=128 s**（0.1% 近似并列优先短 history）。\n2. **Rod1_AllLegal L=128 samples=256 s**。\n3. **Rod2_Common4 L=128 samples=256 s**。\n4. **确实多出三字段**：氩气流量设定、晶体长度、炉压；首项为常数，三者均按 record_time past-only 使用，availability 条件见审计。\n5. **未改善 base prediction**：validation/test RMSE 分别恶化约 1.51%/1.96%。\n6. **E160 没有一致改善**：AllLegal 相对 Common4，base validation 降低 3.34%、test 升高 10.35%；final validation 升高 2.50%、test 降低 21.98%。\n7. **E900 没有一致改善**：base validation 降低 5.47%、test 升高 4.73%；final validation 升高 1.26%、test 降低 21.03%。\n'
    for number,arm in [(8,'Rod1_Common4'),(9,'Rod1_AllLegal'),(10,'Rod2_Common4')]:
        v=ww(arm,'validation');t=ww(arm,'test')
        text+=f"{number}. **{arm} correction**：validation E160/E900 下降 {v.E160_reduction_percent:.2f}%/{v.E900_reduction_percent:.2f}%；test 下降 {t.E160_reduction_percent:.2f}%/{t.E900_reduction_percent:.2f}%。\n"
    text+='11. **lag7/lag14 均明显减弱**；Rod1 test 仍保留约 0.09–0.12 的相关，Rod2 test 变为 -0.011/-0.036。\n12. **~0.14Hz 结构减弱但未消失**：test 频带能量下降约 26.2%/33.9%/50.9%（Common4/AllLegal/Rod2；以频谱表精确值为准），最大峰仍在约 7.2s 附近。\n13. **固定 probe 的 predictability 下降**：三 arm test R² 从约 0.149/0.145/0.112 降至 0.034/0.019/-0.059。\n14. **本轮没有 arm 出现 correction 使 RMSE 改善而 E160/E900 恶化**。AllLegal 对 Common4 的验证期比较则存在 final RMSE 近似不变、白化略差，但不是稳定的 prediction gain。\n15. **更像尚未捕获的 regime/state dependence，而非已测控制遗漏的主要解释**；这是有限表示下的诊断判断，不能从两棒单次尾段比较确认机制。AllLegal test 白化有所改善，但没有 validation/base 上一致证据支持“遗漏 observed-controls 是主因”。\n'
    attach=Path('C:/Users/27437/.codex/attachments/8dfce39e-bde0-47fa-b47e-86d66143ef02/已粘贴的文本.txt')
    end=time.time();elapsed=end-attach.stat().st_ctime
    text+=f"16. **主实验 runner wall time {summary['runner_wall_seconds']:.2f}s**（包含冻结、拟合、test评分、诊断及12张图）。从请求附件创建至报告生成的任务交付 wall time 约 **{elapsed/60:.2f}分钟**，包含数据/协议核对、依赖准备、实现、验证和报告；此起点是请求到达的近似代理。审计输出目录创建至模型完成 {summary['audit_to_completion_wall_seconds']:.2f}s。\n\n"
    text+='## 判断、解释边界与交付\n\n'
    text+=table(['arm','判定','依据'],[['Rod1_Common4','MODERATE_WHITENING_IMPROVEMENT','test E160/E900约下降51%/51%，RMSE降低8%，probe下降；validation→test退化及残余峰明显'],['Rod1_AllLegal','MODERATE_WHITENING_IMPROVEMENT','test E160/E900约下降65%/63%，RMSE降低10%，probe下降；额外变量对Common4的优势仅见test，validation无一致优势'],['Rod2_Common4','STRONG_WHITENING_IMPROVEMENT','validation/test E160约下降79%/80%、E900约下降72%/76%，RMSE改善，lag7/14和probe显著减弱']])
    text+='这些类别是描述性证据判断，不是显著性门槛或白噪声认证。E900 的最高 lag900=1800s，而 test 仅约51/54分钟，最高 lag 仅644/726对；长 lag 不确定性较大。使用 centered biased ACF（分母全段 lag0能量）而非逐 lag 无偏估计，统计口径在训练前固定。Pointwise 95% band 仅图示，无多重比较控制；没有要求所有 lags 入区间，也没有用 Ljung-Box 作为通过标准。两个 rod 的 test 处于不同生长尾段，不能把棒间差异全部归因于控制信息。\n\n'
    text+='独立验证通过：冻结文件与模型哈希一致；原 train membership 完全一致；OOF fit_end 早于 score；saved-model prediction 完整重放；扰动当前/未来控制不影响当前/过去预测，扰动当前/未来 residual 不影响当前/过去 A 输出；全部冷启动行无 correction；直接相关计算与 FFT ACF 一致；独立 sklearn SVD Ridge（含常数和共线性）与批量 Gram 解差异约 1e-15。详见 `VERIFICATION.json`。\n\n'
    text+='所有必需 CSV/JSON/parquet、训练前冻结、源代码、日志、模型 NPZ 和额外频谱摘要均保存在本目录。没有 import 或运行 recursive K/C/W/A runner；没有修改原实验、原工作簿或历史模型。复现命令：在新的空输出目录中放置三份 `.py`，先运行 `audit_and_data.py`，再 `run_experiment.py`，最后 `verify_and_report.py`；原目录有 freeze 文件时 runner 拒绝覆写。\n\n'
    for arm in recipes:
        text+=f'### {arm} descriptive test 图\n\n![{arm} test ACF]({arm}_test_ACF.png)\n\n![{arm} test PSD]({arm}_test_PSD.png)\n\n'
    # Use computed rather than rounded manually transcribed band reductions.
    exact=[100*(1-ss(a,'test','final').seven_second_band_energy/ss(a,'test','base').seven_second_band_energy) for a in recipes]
    text=text.replace('26.2%/33.9%/50.9%', '/'.join(f'{x:.1f}%' for x in exact))
    (ROOT/'DELIVERY_REPORT_CN.md').write_text(text,encoding='utf-8')
    d.write_json('DELIVERY_TIMING.json',dict(request_attachment_creation_unix=attach.stat().st_ctime,report_completion_unix=end,request_to_report_wall_seconds=elapsed,runner_wall_seconds=summary['runner_wall_seconds'],time_basis='request attachment creation is approximate receipt time',timezone='Asia/Shanghai'))
    manifest={f.name:dict(bytes=f.stat().st_size,sha256=r.digest(f)) for f in ROOT.iterdir() if f.is_file() and f.name not in ['DELIVERY_MANIFEST.json','CZ_H1_LIGHT_WHITENING_DELIVERY.zip']}
    d.write_json('DELIVERY_MANIFEST.json',dict(status='COMPLETE',files=manifest))
    with zipfile.ZipFile(ROOT/'CZ_H1_LIGHT_WHITENING_DELIVERY.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in ROOT.iterdir():
            if f.is_file() and f.suffix!='.zip':z.write(f,arcname=f.name)
    print('PASS: verified and packaged',flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        verify();report()
