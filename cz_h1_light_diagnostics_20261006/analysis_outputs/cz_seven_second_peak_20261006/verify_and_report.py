"""Check alignment and diagnostic formulas; create Chinese mechanism report."""
import json, time, zipfile
import numpy as np
import pandas as pd
from scipy.signal import welch, coherence
import run_diagnostic as d

ROOT=d.ROOT

def table(columns,rows):
    return '| '+' | '.join(columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'
def read(name):return pd.read_csv(ROOT/f'{name}.csv')

def verify():
    pre=json.loads((ROOT/'DIAGNOSTIC_PREREGISTRATION.json').read_text(encoding='utf-8'))
    assert d.sha(d.RAW)==pre['raw_sha256']==d.EXPECTED
    for name,h in pre['input_artifact_hashes'].items():assert d.sha(d.PREV/name)==h
    alignment=read('POINTWISE_ALIGNMENT_AUDIT')
    assert len(alignment)==6 and (alignment.status=='PASS').all() and alignment.raw_target_alignment_max_error.max()<1e-12
    hist=read('STEP_INTERVAL_HISTOGRAM');lattice=read('LATTICE_AND_INTERVAL_SUMMARY')
    for rod in ['Rod_1','Rod_2']:
        pooled=lattice[lattice.entity=='WITHIN_SEGMENT_STATIC_COUNT_POOL'];n=int(pooled[pooled.rod==rod].step_intervals.iloc[0])
        h=hist[(hist.rod==rod)&(hist.scope=='rod')]
        assert h['count'].sum()==n
        assert abs(h.fraction.sum()-1)<1e-12
        segment_count=hist[(hist.rod==rod)&(hist.scope=='segment')].groupby('interval_samples')['count'].sum()
        assert np.array_equal(h.set_index('interval_samples')['count'].to_numpy(),segment_count.to_numpy())
    local=read('LOCAL_PEAK_TRACKING')
    assert len(local)==18
    for entity,g in local.groupby('entity'):
        assert (g.global_stop_exclusive-g.global_start>=1536).all()
        assert np.all(g.global_start.to_numpy()[1:]>=g.global_stop_exclusive.to_numpy()[:-1])
    assert local.peak_frequency_0125_0155.between(.125,.155).all()
    ev=read('LOCAL_EVENT_PEAKS');assert len(ev)==54
    assert not ev[['rod','entity','global_start','signal']].duplicated().any()
    phase=read('ROTATION_PHASE_HELDOUT')
    assert (phase.status=='NOT_PERFORMED_ROTATION_UNITS_UNCONFIRMED').all()
    assert phase.heldout_RMSE_change.isna().all()
    # Independent direct window implementation reproduces scipy Welch and coherence.
    raw=pd.read_excel(d.RAW,sheet_name='Sheet1').iloc[734:].reset_index(drop=True)
    yy=np.diff(raw['晶体直径'].to_numpy());xx=raw['晶升速度'].to_numpy()[1:]
    from scipy.signal.windows import hann
    win=hann(1024,sym=False);normalizer=d.FS*np.sum(win**2)
    fx=[];fy=[]
    for start in range(0,len(yy)-1024+1,512):
        x=xx[start:start+1024];y=yy[start:start+1024]
        fx.append(np.fft.rfft((x-x.mean())*win));fy.append(np.fft.rfft((y-y.mean())*win))
    fx=np.asarray(fx);fy=np.asarray(fy)
    pxx=np.mean(abs(fx)**2,axis=0)/normalizer;pyy=np.mean(abs(fy)**2,axis=0)/normalizer
    pyy[1:-1]*=2
    f,expected=welch(yy,fs=.5,nperseg=1024,noverlap=512)
    numerical_psd=float(np.max(abs(pyy-expected)))
    assert numerical_psd<1e-14
    direct_msc=abs(np.mean(np.conj(fx)*fy,axis=0))**2/(np.mean(abs(fx)**2,axis=0)*np.mean(abs(fy)**2,axis=0))
    _,expected_msc=coherence(xx,yy,fs=.5,nperseg=1024,noverlap=512)
    numerical_msc=float(np.max(abs(direct_msc-expected_msc)))
    assert numerical_msc<1e-11
    # Compare stored pointwise PSD to previous report: no new residual model was fitted.
    q=read('QUANTIZATION_DIAGNOSTIC');previous=pd.read_csv(d.PREV/'SPECTRAL_SUMMARY.csv')
    for arm in previous.arm.unique():
        for split in ['validation','test']:
            for stage,signal in [('base','base_residual'),('final','final_residual')]:
                a=q[(q.arm==arm)&(q.scope==split)&(q.signal==signal)].iloc[0]
                b=previous[(previous.arm==arm)&(previous.split==split)&(previous.stage==stage)].iloc[0]
                assert abs(a.band_power-b.seven_second_band_energy)<1e-15
    d.js('VERIFICATION.json',dict(status='PASS',source_hashes_unchanged=True,strict_raw_residual_alignment=True,within_segment_intervals_only=True,local_windows_nonoverlap=True,local_event_rows_unique=True,phase_not_tested_honestly_recorded=True,independent_direct_Welch_max_error=numerical_psd,independent_direct_MSC_max_error=numerical_msc,prior_residual_PSD_reproduced=True,no_model_training=True))

def report():
    q=read('QUANTIZATION_DIAGNOSTIC');l=read('LATTICE_AND_INTERVAL_SUMMARY');h=read('STEP_INTERVAL_HISTOGRAM');steps=read('STEP_SIZE_COUNTS');c=read('CONTROL_OUTPUT_COHERENCE');cp=read('CONTROL_SPECTRAL_SUMMARY');sp=read('ROTATION_SPEED_DISTRIBUTIONS');t=read('LOCAL_PEAK_TRACKING');corr=read('LOCAL_TRACKING_CORRELATIONS');signed=read('SIGNED_UNIT_STEP_DIAGNOSTIC');events=read('LOCAL_EVENT_PEAKS')
    previous_w=pd.read_csv(d.PREV/'WHITENING_METRICS.csv');previous_s=pd.read_csv(d.PREV/'SPECTRAL_SUMMARY.csv')
    long=q[(q.scope=='segment')&(q.n>=1536)]
    text='# CZ H1 约 7 秒短周期峰来源诊断报告\n\n'
    text+='日期：2026-10-06（Asia/Shanghai）。纯诊断，**没有训练任何 PRISM、K/C/W/A、Urysohn 或 residual predictor**。既有模型只使用已保存 pointwise residual，不加载或重训 checkpoint。所有结论均为 post-hoc descriptive。\n\n'
    text+='## 最终裁定\n\n**D. UNRESOLVED_RESIDUAL_STATE**。\n\n'
    text+='该标签表示“目前来源未解决”，不是证明隐藏状态，也不是排除旋转。H_Q 的关键 timing 条件未满足；H_R 有值得继续核对的双棒漂移线索，但单位无法确认，最关键的 rotation-phase held-out test **未执行**，不能当作负结果。既有 residual-history correction 能削弱峰，在当前 observables 下仍由 A 处理最合理。\n\n'
    text+='**最有信息量的新发现：约 0.14 Hz 结构主要在带方向的直径变化序列中，而不是是否跨格的 binary event train。局部峰频率又与晶转速度明显同向漂移。** 因而“存在 0.01 记录格点”不足以解释该峰，机械/测量相位效应仍是合理候选。\n\n'
    text+='## 数据、对齐和固定诊断参数\n\n'
    text+=f'原始工作簿 `实验数据1-张.xlsx` 的 SHA-256 为 `{d.EXPECTED}`，与轻量实验一致。沿用 dt=2s、fs=0.5Hz、Nyquist=0.25Hz。Rod1 分段边界 334/688/734，Rod2 为745，segment 内独立差分、Welch、coherence 和 inter-arrival；从未拼接两段动态序列。静态计数允许汇总来自各段的已计算值，不生成跨边界间隔。\n\n'
    text+='固定重点频带 **0.125–0.155Hz**，参考 bin 为最接近 0.138Hz。Welch nperseg=min(1024,n)、noverlap=floor(nperseg/2)，constant detrend，density scaling。长窗口分辨率 0.00048828125Hz。短段仅有一次平均时仍报告 PSD，coherence 留空并标记不可识别，因为单次 periodogram 的 MSC 会退化到1。\n\n'
    text+='残差对齐使用已保存 entity/origin/global_origin：原始绝对行等于 segment_start+origin，逐行确认 y_true=D[origin]-D[origin-1]、base_residual=y_true-base_prediction，六组 validation/test 全部通过，误差 <1e-12。Coherence 主表使用同一行 t 的原始输入，另报 t-1 对齐的 base-residual coherence；前者是同期观测关联，不是预测因果证据。\n\n'
    text+='局部 tracking 使用每段内不重叠2048 raw-row windows；剩余不足1536行的尾块不纳入。两棒各9块，导数每块2047点、3个 Welch 平均。另行报告既有 descriptive test 的尾段结果，未移动 band 或选择更好看的窗口。\n\n'
    text+='## Diagnostic A：量化格点与跨格 timing\n\n'
    pooled=l[l.entity=='WITHIN_SEGMENT_STATIC_COUNT_POOL']
    rows=[]
    for _,a in pooled.iterrows():
        rows.append([a.rod,int(a.D_unique_exact),int(a.dD_unique_exact),int(a.dD_unique_hundredth_grid),f'{a.dD_zero_ratio*100:.3f}%',f'{a.max_lattice_distance:.8g}',f'{a.interval_median_samples:g} / {a.interval_median_samples*2:g}',f'{a.interval_mode_samples:g} / {a.interval_mode_samples*2:g}',f'{a.interval_3_4_fraction*100:.2f}%',f'{a.independent_geometric_3_4_fraction*100:.2f}%'])
    text+=table(['rod','D unique','dD unique(raw float)','dD grid unique','dD=0','最大距0.01格点','interval median sample/s','mode sample/s','6–8s比例','同event-rate几何参考'],rows)
    text+='所有 D 值距最近0.01格点都小于5e-5，最大偏差约2.69e-5；这是近似0.01 lattice，不能把二进制存储尾差当成21/20种物理 step sizes。部分数值与 float32 编码吻合，但不是所有值完全符合“round(D,2)再存float32”，因此不宣称已识别记录器编码流程。round 到0.01格点后 dD 有6/7种值。\n\n'
    text+='非零 step sizes（各段静态计数汇总，主要三类）：\n\n'
    rows=[]
    for rod,g in steps.groupby('rod'):
        counts=g.groupby('step_size_grid').nonzero_count.sum().sort_values(ascending=False)
        for value,count in counts.head(3).items():rows.append([rod,f'{value:.2f}',int(count),f'{100*count/counts.sum():.2f}%'])
    text+=table(['rod','|dD| grid step','count','占非零step比例'],rows)
    text+='非零 event intervals 主要集中于1–2 samples（2–4s），呈快速递减分布。3–4 samples（6–8s）占约19.3%/21.5%，与对应事件发生率下的独立几何参考约19.2%/20.7%相近；没有在3–4 sample附近形成主峰。几何参考只作直观计数比较，未假设事件真的独立。完整 histogram 包含稀有长间隔，图只展示前12个sample间隔。\n\n'
    text+='长段五种信号的频谱（D含明显生长趋势，因此global dominant是低频；完整各segment结果见CSV）：\n\n'
    rows=[]
    for _,a in long[long.signal.isin(['D','dD','nonzero','absstep','signstep'])].iterrows():
        rows.append([a.rod,a.signal,f'{a.dominant_frequency_Hz:.6f}',f'{a.dominant_period_seconds:.3f}',f'{a.band_power:.5g}',f'{a.band_peak_frequency_Hz:.6f}',f'{a.near_0138_peak_to_background:.2f}',f'{a.band_max_permutation_p:.3f}'])
    text+=table(['rod','signal','dominant Hz','period s','registered band power','band peak Hz','near0.138 peak/background','band-max permutation p'],rows)
    text+='“局部峰”指固定0.135–0.142Hz范围内的local maximum；峰/背景比用0.10–0.18Hz中预注册band以外的median PSD作为背景，≥3标记prominent。显著性辅助检验为长段内部199次随机排列、固定seed，比较整个注册频带的最大PSD（p最小0.005）。它检验temporal ordering相对exchangeability，不检验物理机制；未对所有信号做全局多重比较校正，D的强趋势也使这种全序列排列null不适合判断其高频谱峰缺失。\n\n'
    text+='dD/signstep 的近0.14Hz峰约为邻域背景的22–31倍；nonzero/absstep在参考邻域约1.5–1.6倍。Rod1 absstep的全band-max permutation p=0.010对应其他较弱局部结构，不等于两棒稳定的0.138Hz step-event峰。两棒 nonzero 的全band p=0.080/0.085，最强全频峰反而约0.061Hz（~16.3s）。\n\n'
    text+='A1. **没有看到 binary nonzero-step event train 本身稳定携带与 dD 同频的强约7s峰**。可以有弱local maxima，但位置和主峰不一致；不能把“存在任意local maximum”称为H_Q成立。\n\n'
    text+='A2. **跨格没有以3–4 samples为主要间隔节律重复**；mode为1sample，median为2samples。\n\n'
    text+='A3. **可变step amplitude不是主导；“无方向event timing”也不足以解释。** 固定成0.01×sign(dD)后保留了符号与发生时刻，仍携带同频强峰；这是方向序列的重要性，不能偷换成binary quantization timing成立。长段定幅signed-unit与dD相关约0.976/0.982，保留band能量约85.1%/89.5%。去掉step方向后峰明显弱化。此观察仍不能分辨：方向节律是实际物理运动、测量相位效应，还是记录过程与潜在输出状态的组合。\n\n'
    rows=[]
    for (rod,signal),g in events.groupby(['rod','signal']):
        peak=t[t.rod==rod].peak_frequency_0125_0155.to_numpy()
        match=int(np.sum(abs(g.band_peak_frequency_Hz.to_numpy()-peak)<=2*.5/1024))
        rows.append([rod,signal,f'{match}/9'])
    text+=table(['rod','signal','local band peak与dD相距≤2 bins'],rows)
    text+='该定位容差仅为固定分辨率下的描述；nonzero/absstep多块不同频，signstep两棒9/9同频。\n\n'
    text+='## Diagnostic B：输入PSD与coherence\n\n长段 measured controls，coherence目标为同段dD；原始单位未知，PSD绝对数值不能直接跨变量比较：\n\n'
    rows=[]
    rawc=c[(c.scope=='raw_segment')&(c.Welch_averages>=5)]
    for _,a in rawc.iterrows():
        ps=cp[(cp.entity==a.entity)&(cp.channel==a.channel)].iloc[0]
        rows.append([a.rod,a.channel,f'{ps.band_power:.5g}',f'{a.coherence_nearest_0138:.4f}',f'{a.band_mean_coherence:.4f}',f'{a.band_max_coherence:.4f}',f'{a.band_max_frequency_Hz:.6f}',int(a.Welch_averages)])
    text+=table(['rod','channel','PSD band power','MSC nearest0.138','band mean MSC','band max MSC','max Hz','Welch averages'],rows)
    text+='晶转速度在nearest0.138的MSC约0.033/0.012，没有稳定的双棒强线性speed→dD coherence。升速有部分频带关联；输入与输出均处于闭环、生长阶段漂移条件，不能按单个maximum做因果归属。窄带结构频率随时间变化也会压低整段coherence。**低speed coherence不能排除角相位驱动的测量效应**，恒定speed仍可持续旋转。\n\n'
    text+='严格对齐的base-residual test coherence（同期t；完整validation及t-1在CSV）：\n\n'
    rows=[]
    for _,a in c[(c.scope=='test')&(c.target=='base_residual')&(c.input_alignment=='t')].iterrows():
        rows.append([a.arm,a.channel,f'{a.coherence_nearest_0138:.3f}',f'{a.band_mean_coherence:.3f}',f'{a.band_max_coherence:.3f}',f'{a.band_max_frequency_Hz:.6f}',int(a.Welch_averages)])
    text+=table(['arm','channel','nearest0.138 MSC','band mean','band max','max Hz','Welch averages'],rows)
    text+='尾段只有2个重叠Welch averages，MSC估计的噪声偏置很大。独立average假设下M=2的零coherence点95%阈值就为0.95，重叠/时间相关又使该公式不精确；更不能用对整个频带取最大值来声称显著。表中高值只能视为低精度描述，未用于机制裁定或模型选择。\n\n'
    text+='## Diagnostic C/D：旋转单位与phase\n\n'
    text+='**两种rotation单位UNCONFIRMED**；详见 `ROTATION_UNIT_AUDIT.md`。原始表头、comments、格式和CZ metadata无rpm/rev/s/rad/s证据。没有按常识假定rpm，没有做rpm/60、harmonic差值、angle积分或phase训练。`ROTATION_PHASE_HELDOUT.csv` 全部明确标记 NOT_PERFORMED_ROTATION_UNITS_UNCONFIRMED，指标留空；没有把未执行写成“没有held-out解释力”。因此没有生成phase-removal图。\n\n'
    rows=[]
    for _,a in sp[sp.channel.isin(['crystal_rotation','crucible_rotation'])].iterrows():rows.append([a.rod,a.entity,a.channel,f'{a.p10:.5f}',f'{a["median"]:.5f}',f'{a.p90:.5f}'])
    text+=table(['rod','segment','channel（原始单位）','p10','median','p90'],rows)
    text+='短段可提供速度分布，但其频谱分辨率/平均数量不足以给出可靠的峰-speed物理匹配；长段和window跟踪提供较可靠频率描述。没有用全局平均speed代替局部漂移。\n\n'
    text+='## Peak-frequency tracking\n\n'
    rows=[]
    for _,a in corr.iterrows():rows.append([a.rod,a.variable,int(a.n),f'{a.Pearson_r:.3f}',f'{a.Spearman_r:.3f}',f'{a.first_difference_Pearson_r:.3f}',f'{a.linear_time_partial_Pearson_r:.3f}'])
    text+=table(['rod','tracked variable','windows','Pearson r','Spearman r','first-difference r','linear-time partial r'],rows)
    text+='峰并非整根棒绝对固定：Rod1窗口band峰由0.144531下降到0.139648Hz；Rod2早段0.144531–0.145996，尾段降到0.137695Hz。轻量实验尾段的0.138–0.140Hz是这条漂移轨迹的后段；早段相应周期约6.85–6.92s，后段约7.16–7.26s。\n\n'
    text+='两棒峰频与晶转原始speed的r约0.974/0.971，值得优先核对单位和角相位。控制线性time趋势后关联分别为0.327/0.751，first-difference关联为0.154/0.752；Rod2线索更强，Rod1的整体高相关很大程度与共同时间趋势相容。各棒只有9块，peak-bin量化、连续生长regime和非线性时间趋势仍未消除，以上都是descriptive correlation。\n\n'
    text+='坩埚speed范围很窄，关联约0.565/0.586，控制time后约-0.327/-0.093，不能据此判定坩埚旋转phase。Step-event rate与峰的关联约-0.277/-0.329，未呈同向数值对应；event发生率约0.22–0.26Hz，亦不等于0.14Hz。这些比较没有确认旋转frequency/harmonic。\n\n'
    text+='## H_S与既有residual correction\n\n直接复用轻量实验已保存结果，没有训练新A：\n\n'
    rows=[]
    for _,a in previous_w[previous_w.split=='test'].iterrows():
        b=previous_s[(previous_s.arm==a.arm)&(previous_s.split=='test')&(previous_s.stage=='base')].iloc[0]
        f=previous_s[(previous_s.arm==a.arm)&(previous_s.split=='test')&(previous_s.stage=='final')].iloc[0]
        rows.append([a.arm,f'{a.E160_reduction_percent:.1f}%',f'{a.E900_reduction_percent:.1f}%',f'{100*(1-f.seven_second_band_energy/b.seven_second_band_energy):.1f}%',f'{100*(1-f.dominant_PSD/b.dominant_PSD):.1f}%'])
    text+=table(['arm','E160降幅','E900降幅','注册band能量降幅','dominant peak PSD降幅（峰位置可变）'],rows)
    text+='Residual history的observable predictability及峰削弱是已有证据。本次重算的band power逐组复现旧报告，未根据本次诊断重选H/alpha。它支持该结构可以被output/residual memory利用，**不等于证明其物理起因是隐藏状态**。可观测输入phase造成的周期也可能被A间接吸收。\n\n'
    text+='## 四种允许裁定的证据核对\n\n'
    text+=table(['假设/标签','本轮证据','裁定限制'],[['H_Q / A','近0.01 lattice明确；但binary event强峰、6–8s interval集中度和双棒同频稳定性不满足','不裁定QUANTIZATION_TIMING_SUPPORTED；lattice与signed-step谱峰不足以识别timing机制'],['H_R / B','双棒crystal-speed/peak漂移关联强，尤其Rod2调整time后仍较强','单位缺失，rotation-frequency/harmonic及phase held-out测试未执行；不能裁定ROTATION_PHASE_SUPPORTED，也不能排除'],['C','没有同时满足量化timing与rotation-phase两项的稳定证据','不裁定MIXED_MECHANISM_PLAUSIBLE'],['H_S / D','目前mechanism未解决；既有A显著削弱相关与峰','选择UNRESOLVED_RESIDUAL_STATE，物理来源保持unresolved']])
    text+='若随后获得有来源的旋转单位、实际速度与指令速度说明、采样延迟/记录同步信息，可按本任务固定的一阶/二阶phase representation执行前段fit后段score；在此之前，speed积分不是可靠的角相位证据。本轮到此结束，没有额外复杂模型或调参。\n\n'
    text+='## PRISM stage角色\n\n'
    text+='当前observable routing：**A = observable residual predictable memory**。来源仍unresolved，不能把这个工程处理归属写成物理归属证明。如果以后单位和phase held-out证据成立，理论上应优先归K的measured-input-driven dynamic response，并需要phase-aware/integrated-angle representation；旧speed-history K未消除峰不等于输入完全无信息。C只组合frozen K representations，W是current-q static closure，本周期峰不应让C/W承担新的temporal state。详见 `MODEL_ROLE_ASSESSMENT.md`。\n\n'
    text+='## 交付、验证和耗时\n\n'
    summary=json.loads((ROOT/'RUN_SUMMARY.json').read_text(encoding='utf-8'))
    elapsed=time.time()-d.ATTACH.stat().st_ctime
    text+=f"分析脚本 wall time **{summary['analysis_wall_seconds']:.2f}s**；含单位核对、实现、验证和报告的请求到交付耗时约 **{elapsed/60:.2f}分钟**（请求附件创建时间为到达时间近似代理），低于30分钟预算。没有重训；追加tracking去趋势描述与修复CSV幂等写入后执行过两次同参数诊断，累计分析脚本wall约19s。\n\n"
    text+='验证通过：输入文件与既有交付hash不变，逐行原始目标/残差对齐，event intervals计数可由各段静态汇总复现，18块local windows无跨段重叠，54条local event记录唯一，独立显式Hann FFT计算重现Welch PSD与MSC，既有residual band power重现。详见 `VERIFICATION.json`。\n\n'
    text+='必需文件齐全：`QUANTIZATION_DIAGNOSTIC.csv`（五信号和signed unit step的PSD摘要，含原段和已交付尾段）、`STEP_INTERVAL_HISTOGRAM.csv`、`CONTROL_PSD.csv`、`CONTROL_OUTPUT_COHERENCE.csv`、`ROTATION_UNIT_AUDIT.md`、`ROTATION_PHASE_HELDOUT.csv`、`LOCAL_PEAK_TRACKING.csv`、`MODEL_ROLE_ASSESSMENT.md` 及4张要求的非phase图。额外提供 lattice/step-size、局部相关、全谱、输入频谱摘要、对齐audit和单位证据。运行脚本位于本目录，不依赖训练runner。\n\n'
    for rod in ['Rod1','Rod2']:
        text+=f'### {rod} 诊断图\n\n![{rod} step PSD]({rod}_dD_step_event_PSD.png)\n\n![{rod} rotation vs peak]({rod}_rotation_vs_peak.png)\n\n'
    (ROOT/'SEVEN_SECOND_PEAK_DIAGNOSTIC_REPORT_CN.md').write_text(text,encoding='utf-8')
    roles='# PRISM stage角色评估\n\n本轮唯一机制裁定：**UNRESOLVED_RESIDUAL_STATE**。这是来源未解决的诊断标签。\n\n当前可观测信息下，A处理该残差结构合理：A = observable residual predictable memory。已有held-out residual correction降低E160/E900、注册频带能量和峰幅，但不能据此证明隐藏物理state。\n\n旋转仍有线索：两棒局部peak/crystal-speed同向漂移，r约0.97；Rod2去线性time后的描述关联仍约0.75。单位缺失，phase未检验，因此不把结构永久归给A。若未来确认单位且phase model在held-out稳定压低峰，理论归属优先K，需要phase-aware/integrated-angle representation；旧speed-history K没有吃掉峰不等于measured input没信息。\n\n近0.01 lattice和signed-step峰均已观察到，但binary nonzero timing没有稳定同频强峰，event intervals也没有6–8s主峰，未裁定QUANTIZATION_TIMING_SUPPORTED。若后来有更强记录timing证据，该可由output history预测的measurement/recording-state component在observable routing中由A处理也合理，不是plant input-response K。\n\nC只组合frozen K representations；W为current-q static closure。不因短周期峰引入C/W的新temporal state。本轮没有改变或重新训练任何stage。\n'
    (ROOT/'MODEL_ROLE_ASSESSMENT.md').write_text(roles,encoding='utf-8')
    d.js('DELIVERY_TIMING.json',dict(request_to_report_wall_seconds=elapsed,analysis_wall_seconds=summary['analysis_wall_seconds'],timezone='Asia/Shanghai',mechanism_verdict='UNRESOLVED_RESIDUAL_STATE',phase_test_status='NOT_PERFORMED_UNCONFIRMED_UNITS'))
    d.js('DELIVERY_MANIFEST.json',dict(status='COMPLETE',files={p.name:dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in ROOT.iterdir() if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','CZ_SEVEN_SECOND_PEAK_DIAGNOSTIC_DELIVERY.zip']}))
    with zipfile.ZipFile(ROOT/'CZ_SEVEN_SECOND_PEAK_DIAGNOSTIC_DELIVERY.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix!='.zip':archive.write(p,arcname=p.name)
    print('PASS: diagnostic verified and packaged')

if __name__=='__main__':verify();report()
