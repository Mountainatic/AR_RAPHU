"""Independent metric/point checks and matched-window report; no fitting."""
from pathlib import Path
import os,json,hashlib,pickle
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';WINDOWS={4:8,8:16,16:16}
ORDER=['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer','PRISM_no_A','PRISM']
NEURAL=['LSTM','Transformer','iTransformer','TimeXer']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def verify():
    reg=read(ROOT/'REGISTRATION.json');gate=read(ROOT/'ALL_FITS_FROZEN.json')
    assert gate['all_fits_frozen_before_evaluation'] and read(ROOT/'MIGRATION_CHECK.json')['status']=='COMPLETED'
    for group in [reg['code_sha256'],reg['reference_sha256'],gate['receipts']]:assert all(sha(ROOT/k)==v for k,v in group.items())
    assert sha(ROOT/'data/train.npz')==reg['train_sha256'] and sha(ROOT/'data/development.npz')==reg['development_file_sha256']
    raw=np.vstack([np.load(ROOT/'data/train.npz')['values'],np.load(ROOT/'data/development.npz')['values']]);assert raw.shape==(8064,6)
    m=pd.read_csv(OUT/'metrics.csv');long=pd.read_parquet(OUT/'point_predictions_long.parquet');assert len(m)==27 and len(long)==54432
    runtime=[];checks=[]
    for p in (ROOT/'runs').rglob('FROZEN.json'):
        r=read(p);d=int(p.parent.parent.name[1:]);assert not r['development_read_during_fit'] and r['latest_fit_target']<=6048-d
        assert all(sha(p.parent/k)==v for k,v in r['files'].items())
        if p.parent.name in NEURAL:assert r['selected']['config']['L']==WINDOWS[d]
        runtime.append(dict(delay=d,model=p.parent.name,seconds=r['runtime_seconds'],selected=json.dumps(r['selected'],sort_keys=True)))
    assert len(runtime)==15;pd.DataFrame(runtime).to_csv(OUT/'runtime_and_selection.csv',index=False)
    for d,L in WINDOWS.items():
        f=pd.read_parquet(OUT/f'point_predictions_D{d}.parquet');csv=pd.read_csv(OUT/f'point_predictions_D{d}.csv')
        np.testing.assert_allclose(csv,f,rtol=1e-12,atol=1e-12);o=np.arange(6048,8064)
        np.testing.assert_array_equal(f.origin,o);np.testing.assert_array_equal(f.y_true,raw[o,0]);y=f.y_true.to_numpy()
        linear=pickle.loads((ROOT/'runs'/f'D{d}'/'linear/linear.pkl').read_bytes())
        for name,obj in linear.items():
            c=obj['selected']['config'];assert c['Ly']==L
            own=raw[o[:,None]-d-np.arange(L),0]
            if name=='ARX':assert c['Xlags']==list(range(L))
            x=own if name=='AR' else np.c_[own,raw[o[:,None]-np.arange(L),1:].reshape(len(o),-1)]
            p=obj['ym']+obj['ys']*obj['model'].predict(x)+(raw[o-d,0] if obj['selected']['target']=='delta' else 0.)
            np.testing.assert_allclose(p,f[name],rtol=1e-12,atol=1e-12)
            assert obj['regularization_certificate']['relative_kkt']<1e-8
        for name in ORDER:
            p=f[name].to_numpy();e=y-p;v=np.abs(y)>1e-8
            expected=dict(R2=1-np.dot(e,e)/np.dot(y-y.mean(),y-y.mean()),RMSE=np.sqrt(np.dot(e,e)/len(e)),MAE=np.mean(np.abs(e)),
                MAPE_pct=100*np.mean(np.abs(e[v])/np.abs(y[v])),sMAPE_pct=200*np.mean(np.abs(e)/np.maximum(np.abs(y)+np.abs(p),1e-8)),
                WAPE_pct=100*np.abs(e).sum()/np.abs(y).sum(),bias=np.mean(p-y))
            row=m[(m.label_delay==d)&(m.model==name)].iloc[0]
            assert row.direct_window==(1 if name=='Persistence' else L)
            for k,value in expected.items():np.testing.assert_allclose(value,row[k],rtol=1e-10,atol=1e-12)
            block=long[(long.label_delay==d)&(long.model==name)];np.testing.assert_allclose(p,block.y_pred,rtol=1e-12,atol=1e-12)
            if name in NEURAL:
                predictions=[pd.read_parquet(OUT/f'D{d}_{name}_seed{s}.parquet').y_pred.to_numpy() for s in [0,1,2]]
                np.testing.assert_allclose(np.mean(predictions,axis=0),p,rtol=1e-12,atol=1e-12)
            if name.startswith('PRISM'):np.testing.assert_array_equal(p,pd.read_parquet(ROOT/'data'/f'prism_D{d}.parquet')[name])
            if name=='PRISM':np.testing.assert_allclose(p,pd.read_parquet(ROOT/'previous'/f'prism_D{d}.parquet').PRISM,rtol=1e-12,atol=1e-12)
            checks.append(dict(delay=d,model=name,window=int(row.direct_window),metrics_recomputed=True,point_alignment=True))
        assert read(ROOT/'audits'/f'PRISM_D{d}_CAUSAL_REPLAY.json')['status']=='COMPLETED'
    assert read(OUT/'CAUSAL_AUDIT.json')['status']=='COMPLETED' and len(read(OUT/'CAUSAL_AUDIT.json')['checks'])==36
    seeds=pd.read_csv(OUT/'metrics_by_seed.csv');stats=seeds.groupby(['label_delay','model'])[['R2','RMSE','MAE','MAPE_pct']].agg(['mean','std']).reset_index()
    stats.columns=['_'.join(c).rstrip('_') for c in stats.columns];stats.to_csv(OUT/'metrics_seed_mean_std.csv',index=False)
    old=pd.read_csv(ROOT/'previous/fixed128_metrics.csv');old['model']=old.model.replace({'PRISM_L128':'PRISM','PRISM_L128_no_A':'PRISM_no_A'})
    comparison=m.merge(old,on=['label_delay','model'],suffixes=('_matched','_fixed128'));comparison['RMSE_change_pct']=100*(comparison.RMSE_matched/comparison.RMSE_fixed128-1)
    comparison.to_csv(OUT/'comparison_with_fixed128.csv',index=False)
    selected=pd.read_csv(ROOT/'previous/selected_prism_metrics.csv')
    preserved=m[m.model=='PRISM'].merge(selected[selected.model=='PRISM'],on=['label_delay','model'],suffixes=('_new','_previous'))
    for k in ['R2','RMSE','MAE','MAPE_pct']:np.testing.assert_allclose(preserved[k+'_new'],preserved[k+'_previous'],rtol=1e-12,atol=1e-12)
    dump(OUT/'INDEPENDENT_VERIFICATION.json',dict(status='COMPLETED',checks=checks,new_neural_checkpoints=36,new_linear_models=6,
        existing_PRISM_checkpoints_replayed=6,metric_rows=27,point_rows=54432,full_PRISM_predictions_preserved=True,
        all_frozen_hashes_verified=True,reserved_tail_read=False))
    first=pd.read_csv(ROOT/'previous/first_matched_metrics.csv')
    repeated=m.merge(first,on=['label_delay','model'],suffixes=('_repeat','_first'))
    repeated['RMSE_change_pct']=100*(repeated.RMSE_repeat/repeated.RMSE_first-1)
    repeated.to_csv(OUT/'comparison_with_first_matched.csv',index=False)
    return m
def figures(m):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={'PRISM':'#bd3c36','PRISM_no_A':'#80579d'}
    for d,L in WINDOWS.items():
        f=pd.read_parquet(OUT/f'point_predictions_D{d}.parquet');fig,axes=plt.subplots(3,3,figsize=(18,10),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,ORDER):
            r=m[(m.label_delay==d)&(m.model==name)].iloc[0];ax.plot(f.origin,f.y_true,c='#39434b',lw=.8,label='Observed',alpha=.65)
            ax.plot(f.origin,f[name],c=colors.get(name,'#247ba0'),lw=.8,label='Predicted');ax.grid(alpha=.15)
            ax.set_title(f'{name} | R2={r.R2:.4f} RMSE={r.RMSE:.5f}',loc='left',fontsize=9)
        for ax in axes[-1]:ax.set_xlabel('Original row index')
        for ax in axes[:,0]:ax.set_ylabel('H2S (source scale)')
        axes[0,0].legend(fontsize=8);fig.suptitle(f'SRU H2S | label delay {d} | matched direct window {L}\nFull PRISM retains online A; neural predictions average3 seeds',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/f'point_predictions_D{d}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(18,11))
    for j,(d,L) in enumerate(WINDOWS.items()):
        f=m[m.label_delay==d].set_index('model').loc[ORDER]
        for i,key in enumerate(['R2','RMSE','MAE','MAPE_pct']):
            ax=axes[j,i];ax.barh(range(9),f[key],color=[colors.get(n,'#247ba0') for n in ORDER]);ax.invert_yaxis()
            ax.set_yticks(range(9),ORDER if i==0 else ['']*9);ax.set_title(f'Delay {d} | L={L} | {key}');ax.grid(axis='x',alpha=.2)
    fig.suptitle('SRU matched-window benchmark | same2016 development targets',fontsize=14);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'metrics_comparison.png',dpi=150);plt.close(fig)
def report(m):
    table='|标签延迟|历史窗口|模型|R²|RMSE|MAE|MAPE (%)|\n|---:|---:|---|---:|---:|---:|---:|\n'
    for d,L in WINDOWS.items():
        for name in ORDER:
            r=m[(m.label_delay==d)&(m.model==name)].iloc[0];table+=f'|{d}|{int(r.direct_window)}|{name}|{r.R2:.6f}|{r.RMSE:.8f}|{r.MAE:.8f}|{r.MAPE_pct:.4f}|\n'
    text='''# SRU H2S：基线采用PRISM选定的相同历史窗口

状态：COMPLETED。本轮为新服务器上的再次完整重训。依照用户要求，标签延迟4步的AR、ARX、LSTM、Transformer、iTransformer、TimeXer统一8步历史，延迟8和16步统一16步历史。窗口来自此前完整PRISM训练内选择，未按本轮基线开发结果挑窗口。主表保留完整PRISM在线A，另列相同窗口的no_A与Persistence。旧实验均保留。

## 协议

相同期软测量y(t)，可用X至t、y至t-delay。直接X为x(t-L+1)..x(t)，成熟输出为y(t-delay-L+1)..y(t-delay)。AR用完整L个成熟输出，ARX用完整L个输出及5L个过程输入；神经模型用相同6通道L步序列。Persistence只用最新成熟输出。

训练原始区间[0,6048)，可拟合目标192..6048-delay（含端点）；评估为相同2016个开发目标[6048,8064)。保留尾段[8064,10081)未读取。4/8/16是实验性标签成熟延迟，不是预测未来步长或已确认的分析仪延迟。

只调整基线直接窗口，保留上一轮的训练规则：AR/ARX正则0.01/1/100/1000/10000与level/delta；神经模型宽度32/64、学习率0.001/0.0003两对配置×level/delta，共4项。训练内前向折70%-85%及85%-100%，继续使用保守隔离128+delay步，保持训练/评分目标日期不随本轮短窗口改变。最多80epoch、patience12、batch256；最终epoch为选中两折epoch的中位数。训练内选择种子20261023，最终种子0/1/2；主表为三种子逐点均值，单种子指标与标准差另存。固定512推理批量，尾块补零后丢弃填充结果。

完整PRISM直接重放此前冻结的L8/L16/L16模型，完整预测与上一轮一致；没有重新训练或重新选A。相同窗口no_A使用原候选的冻结模型，因此延迟4步no_A由此前独立选中的L16改为匹配完整PRISM的L8。其余no_A不变。每个PRISM过程响应仍由其模型结构压缩，不把压缩特征称为L个独立系数。

本轮统一的是直接历史跨度和标签可用性。PRISM仍保留分阶段训练、已有窗口搜索与完整在线A；成熟残差、发行预测及RLS状态可间接依赖更早数据。其他模型开发段权重固定，不能将本表称为完全相同总记忆、训练预算或在线更新预算。

## 环境与校验

按用户再次明确要求，在connect.westb.seetacloud.com:13762的新RTX4090实例独立重新拟合全部基线，前一轮RTX5090结果原样保留。本轮沿用已审计的严格FP32后端：关闭TF32、MHA fastpath和非math的SDPA路径；模型、优化候选、训练内折、随机种子及数值容限均保持不变。训练前用旧L128/seed0冻结权重与CPU FP64参考核对四类神经网络，通过原rtol3e-5/atol2e-6容限，见MIGRATION_CHECK.json。

本轮没有沿用已有基线权重，而是重新做训练内选择及三个种子的最终拟合。使用相同种子是计算复跑的定义，不构成新的独立随机种子样本；数据仍是同一个开发区间。仅保留完整PRISM及同窗口no_A的既有冻结权重。与上一轮相同窗口结果的逐模型比较见results/comparison_with_first_matched.csv；硬件变化和数值舍入可使神经训练轨迹及选中配置变化。

全部15组新基线拟合冻结后才生成本轮新模型开发预测。PRISM未来X/未成熟y扰动及7个前缀通过1e-12检查，36条神经种子链通过原容限与批量前缀检查；独立重算指标并核对CSV/Parquet、种子均值、线性模型和完整PRISM旧预测。

开发段此前已多轮使用，本结果不是独立最终测试或普遍优越性证明。指标在源数据原尺度计算，无裁剪。MAPE百分数，仅排除abs(y)<=1e-8并计数，另有sMAPE、WAPE和bias。

## 完整指标

'''+table+'''

## 产物

- [完整指标](results/metrics.csv)、[逐种子指标](results/metrics_by_seed.csv)、[种子均值/标准差](results/metrics_seed_mean_std.csv)。
- [54432行逐点预测](results/point_predictions_long.csv)、[Parquet长表](results/point_predictions_long.parquet)。
- [延迟4逐点](results/point_predictions_D4.csv)、[延迟8逐点](results/point_predictions_D8.csv)、[延迟16逐点](results/point_predictions_D16.csv)。
- [与旧固定128步结果的描述性比较](results/comparison_with_fixed128.csv)、[训练时间与配置](results/runtime_and_selection.csv)。
- [预登记](REGISTRATION.json)、[全部冻结门槛](ALL_FITS_FROZEN.json)、[独立复核](results/INDEPENDENT_VERIFICATION.json)、[迁移数值检查](MIGRATION_CHECK.json)。runs/包含全部新权重、内层候选分数及轨迹，prism_reference/包含原模型和完整候选收据；没有重新拟合PRISM。

## 复现

原科学Python3.10环境用于prism_replay.py，Torch/Python3.12环境用于其他脚本。包内不附完整环境，精确版本保存在REGISTRATION.json及原参考登记。恢复原字节后，读取已有冻结模型可执行prism_replay.py4/8/16及experiment.py evaluate；不需要重训。

新目录重训需保留全部源码、data/、previous/及prism_reference/，先experiment.py prepare和smoke，并完成环境迁移核对，再launch.py。launch用两个GPU工作进程，每进程3个CPU线程并使用MPS，待全部拟合冻结后重放PRISM和评价。report.py只独立复核及绘图。模型实现沿用作者官方TimeXer仓库固定提交76011909357972bd55a27adba2e1be994d81b327，两篇论文均为2024年，不宣称截至2026年全部最新SOTA。

![指标](results/metrics_comparison.png)

![延迟4](results/point_predictions_D4.png)

![延迟8](results/point_predictions_D8.png)

![延迟16](results/point_predictions_D16.png)
'''
    (ROOT/'REPORT_CN.md').write_text(text,encoding='utf-8');(ROOT/'README.md').write_text('# SRU matched-window baselines\n\n[Report](REPORT_CN.md) | [Metrics](results/metrics.csv)\n',encoding='utf-8')
if __name__=='__main__':
    m=verify();figures(m);report(m);dump(OUT/'REPORT_COMPLETED.json',dict(status='COMPLETED',metric_rows=27,point_rows=54432,reserved_tail_read=False));print('REPORT_COMPLETED',flush=True)
