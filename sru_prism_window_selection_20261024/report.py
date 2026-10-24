"""Combine frozen128 baselines with selected PRISM; recompute metrics and audit."""
from pathlib import Path
import os,json,hashlib,shutil,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
BASE=['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer']
NEURAL=['LSTM','Transformer','iTransformer','TimeXer']
ORDER=BASE+['PRISM_no_A','PRISM']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def metrics(y,p):
    e=y-p;valid=np.abs(y)>1e-8
    return dict(n=len(y),R2=1-np.dot(e,e)/np.dot(y-y.mean(),y-y.mean()),RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(np.abs(e)),
        MAPE_pct=100*np.mean(np.abs(e[valid])/np.abs(y[valid])),MAPE_excluded_nearzero=int((~valid).sum()),
        sMAPE_pct=200*np.mean(np.abs(e)/np.maximum(np.abs(y)+np.abs(p),1e-8)),WAPE_pct=100*np.abs(e).sum()/np.abs(y).sum(),bias=np.mean(p-y))
def combine():
    OUT.mkdir(exist_ok=True);reg=read(ROOT/'REGISTRATION.json');gate=read(ROOT/'ALL_SELECTIONS_FROZEN.json')
    assert gate['all_fits_frozen_before_evaluation']
    for group in [reg['code_sha256'],reg['baseline_files_sha256'],gate['receipts']]:assert all(sha(ROOT/k)==v for k,v in group.items())
    assert sha(ROOT/'data/train.npz')==reg['train_sha256'] and sha(ROOT/'data/development.npz')==reg['development_file_sha256']
    assert sha(ROOT/'WINDOW_CANDIDATES.csv')==gate['ledger_sha256']
    raw=np.vstack([np.load(ROOT/'data/train.npz')['values'],np.load(ROOT/'data/development.npz')['values']]);assert raw.shape==(8064,6)
    rows=[];long=[];anchors=[];windows=[]
    for d in [4,8,16]:
        prior=pd.read_parquet(ROOT/'baseline_reference'/f'point_predictions_D{d}.parquet')
        f=prior[['origin','label_delay','latest_available_y_index','y_true']+BASE].copy()
        new=pd.read_parquet(ROOT/'data'/f'prism_D{d}.parquet')
        np.testing.assert_array_equal(new.origin,f.origin);np.testing.assert_array_equal(new.y_true,f.y_true)
        np.testing.assert_array_equal(f.origin,np.arange(6048,8064));np.testing.assert_array_equal(f.y_true,raw[f.origin,0])
        assert read(ROOT/'audits'/f'D{d}_CAUSAL_REPLAY.json')['status']=='COMPLETED'
        for name,variant in [('PRISM','full'),('PRISM_no_A','no_A')]:
            f[name]=new[name].to_numpy();window=gate['choices'][str(d)][variant]['window']
            frozen=read(ROOT/'runs'/f'D{d}'/f'L{window}'/'FROZEN.json')
            windows.append(dict(label_delay=d,model=name,direct_window=window,training_selection_MSE=gate['choices'][str(d)][variant]['selection_MSE'],
                KCW_config=json.dumps(frozen['selected'][variant]['KCW']['config'],sort_keys=True),W=frozen['selected'][variant]['KCW']['mode'],
                A=json.dumps(frozen['selected'][variant]['A'],sort_keys=True)))
        for name in ORDER:
            y=f.y_true.to_numpy();p=f[name].to_numpy();e=y-p;valid=np.abs(y)>1e-8
            direct=gate['choices'][str(d)]['full' if name=='PRISM' else 'no_A']['window'] if name.startswith('PRISM') else (1 if name=='Persistence' else 128)
            rows.append(dict(label_delay=d,model=name,prediction='mean_of_seeds_0_1_2' if name in NEURAL else 'deterministic',direct_window=direct,**metrics(y,p)))
            long.append(pd.DataFrame(dict(origin=f.origin,label_delay=d,model=name,direct_window=direct,y_true=y,y_pred=p,residual=e,
                absolute_error=np.abs(e),squared_error=e**2,absolute_percentage_error_pct=np.where(valid,100*np.abs(e)/np.maximum(np.abs(y),1e-8),np.nan))))
            if name in BASE:np.testing.assert_array_equal(p,prior[name])
        f.to_csv(OUT/f'point_predictions_D{d}.csv',index=False);f.to_parquet(OUT/f'point_predictions_D{d}.parquet',index=False)
        # Reproducibility anchor: same L128 code/geometry/seed-free fits as preceding run.
        oldroot=ROOT/'baseline_reference'/'L128_ANCHORS'
        for variant in ['full','no_A']:
            now=read(ROOT/'runs'/f'D{d}'/'L128'/'FROZEN.json')['selected'][variant]
            old=read(oldroot/f'D{d}_FROZEN.json')['selected'][variant]
            assert now['KCW']['config']==old['KCW']['config'] and now['KCW']['mode']==old['KCW']['mode']
            if variant=='full':
                np.testing.assert_allclose(now['A']['parent_risk'],old['A']['parent_risk'],rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(now['A']['child_risk'],old['A']['child_risk'],rtol=1e-12,atol=1e-14)
            anchors.append(dict(delay=d,variant=variant,selected_KCW_equal=True,full_A_risks_equal=variant=='full'))
    m=pd.DataFrame(rows);m.to_csv(OUT/'metrics.csv',index=False);l=pd.concat(long,ignore_index=True)
    l.to_csv(OUT/'point_predictions_long.csv',index=False);l.to_parquet(OUT/'point_predictions_long.parquet',index=False)
    pd.DataFrame(windows).to_csv(OUT/'selected_windows.csv',index=False)
    for name in ['metrics_by_seed.csv','metrics_seed_mean_std.csv']:shutil.copy2(ROOT/'baseline_reference'/name,OUT/name)
    old=pd.read_csv(ROOT/'baseline_reference'/'metrics.csv');old=old[old.model.isin(['PRISM_L128','PRISM_L128_no_A'])].copy()
    old['model']=old.model.replace({'PRISM_L128':'PRISM','PRISM_L128_no_A':'PRISM_no_A'})
    comparison=m.merge(old,on=['label_delay','model'],suffixes=('_selected','_fixed128'))
    comparison['RMSE_change_pct']=100*(comparison.RMSE_selected/comparison.RMSE_fixed128-1)
    comparison.to_csv(OUT/'prism_vs_fixed128.csv',index=False)
    return m,l,anchors
def verify(m,l,anchors):
    # Separate pass through serialized point tables, independent metric arithmetic.
    count=0
    for d in [4,8,16]:
        f=pd.read_parquet(OUT/f'point_predictions_D{d}.parquet');csv=pd.read_csv(OUT/f'point_predictions_D{d}.csv')
        np.testing.assert_allclose(csv,f,rtol=1e-12,atol=1e-12)
        for name in ORDER:
            y=f.y_true.to_numpy();p=f[name].to_numpy();e=y-p;v=np.abs(y)>1e-8
            expected=dict(R2=1-sum(e*e)/sum((y-np.mean(y))**2),RMSE=(sum(e*e)/len(e))**.5,MAE=sum(np.abs(e))/len(e),MAPE_pct=100*sum(np.abs(e[v]/y[v]))/sum(v))
            row=m[(m.label_delay==d)&(m.model==name)].iloc[0]
            for k,value in expected.items():np.testing.assert_allclose(value,row[k],rtol=1e-12,atol=1e-12)
            block=l[(l.label_delay==d)&(l.model==name)]
            np.testing.assert_array_equal(p,block.y_pred);count+=1
    assert count==27 and len(l)==54432
    dump(OUT/'INDEPENDENT_VERIFICATION.json',dict(status='COMPLETED',metric_rows=count,point_rows=len(l),
        baseline_predictions_bitwise_equal=True,all_source_and_freeze_hashes_checked=True,all_candidate_score_origins_identical=True,
        L128_retraining_anchors=anchors,reserved_tail_read=False,metrics_recomputed_from_serialized_predictions=True))
def figures(m):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    gate=read(ROOT/'ALL_SELECTIONS_FROZEN.json');colors={'PRISM':'#bd3c36','PRISM_no_A':'#80579d'}
    for d in [4,8,16]:
        f=pd.read_parquet(OUT/f'point_predictions_D{d}.parquet');fig,axes=plt.subplots(3,3,figsize=(18,10),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,ORDER):
            row=m[(m.label_delay==d)&(m.model==name)].iloc[0];ax.plot(f.origin,f.y_true,c='#39434b',lw=.8,label='Observed',alpha=.65)
            ax.plot(f.origin,f[name],c=colors.get(name,'#247ba0'),lw=.8,label='Predicted')
            ax.set_title(f'{name} L={int(row.direct_window)} | R2={row.R2:.4f} RMSE={row.RMSE:.5f}',loc='left',fontsize=9);ax.grid(alpha=.15)
        for ax in axes[-1]:ax.set_xlabel('Original row index')
        for ax in axes[:,0]:ax.set_ylabel('H2S (source scale)')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'SRU H2S | label delay {d} | baselines128, PRISM train-selected window\nFull PRISM includes online A state; neural predictions average3 seeds',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/f'point_predictions_D{d}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(18,11))
    for j,d in enumerate([4,8,16]):
        f=m[m.label_delay==d].set_index('model').loc[ORDER]
        for i,key in enumerate(['R2','RMSE','MAE','MAPE_pct']):
            ax=axes[j,i];ax.barh(range(9),f[key],color=[colors.get(n,'#247ba0') for n in ORDER]);ax.invert_yaxis()
            ax.set_yticks(range(9),ORDER if i==0 else ['']*9);ax.set_title(f'Delay {d} | {key}');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed128 baselines vs train-selected-window PRISM | same2016 development targets',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'metrics_comparison.png',dpi=150);plt.close(fig)
    candidates=pd.read_csv(ROOT/'WINDOW_CANDIDATES.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,d in zip(axes,[4,8,16]):
        for variant in ['full','no_A']:
            f=candidates[(candidates.label_delay==d)&(candidates.variant==variant)].sort_values('window')
            ax.plot(f.window,np.sqrt(f.selection_MSE),'-o',label=variant);chosen=f[f.selected];ax.scatter(chosen.window,np.sqrt(chosen.selection_MSE),s=130,facecolors='none',edgecolors='black')
        ax.set_xscale('log',base=2);ax.set_xticks([8,16,32,64,128],[8,16,32,64,128]);ax.set_title(f'Delay {d}');ax.set_xlabel('Candidate direct window');ax.set_ylabel('Training nested score RMSE');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'window_selection.png',dpi=160);plt.close(fig)
def report(m):
    table='|标签延迟|模型|直接窗口|R²|RMSE|MAE|MAPE (%)|\n|---:|---|---:|---:|---:|---:|---:|\n'
    for d in [4,8,16]:
        for name in ORDER:
            r=m[(m.label_delay==d)&(m.model==name)].iloc[0];table+=f'|{d}|{name}|{int(r.direct_window)}|{r.R2:.6f}|{r.RMSE:.8f}|{r.MAE:.8f}|{r.MAPE_pct:.4f}|\n'
    text='''# SRU H2S：固定128步基线与自选窗口 PRISM

状态：COMPLETED。按用户最新要求，AR、ARX、LSTM、Transformer、iTransformer、TimeXer 保留上一轮固定128步结果；PRISM 将直接窗口纳入模型选择，在8、16、32、64、128中按训练内评分选择。完整在线A保留在主表；no_A独立选窗。Persistence只使用最新成熟标签。

## 窗口选择与数据边界

数据、代码分支和基线权重沿用已核验的最新SRU H2S分支。训练原始区间 `[0,6048)`；同样的2016个开发目标 `[6048,8064)`。保留尾段 `[8064,10081)` 未读取。delay4/8/16为实验性标签成熟延迟，不是预测步长或已确认的分析仪延迟。预测y(t)可用X至t、y至t-delay。

候选L决定X的当前至过去L-1步、y的t-delay至t-delay-L+1。原KC结构可选当前响应/历史响应均值、两点输出端点0/L-1或不使用输出状态；不把压缩后的两个端点伪称L个独立系数。改变L同时改变响应记忆和端点，不是只截短一个固定128模型。完整A还可使用发行预测、成熟残差及递归参数状态，其间接历史可能超过直接窗口L。

15个delay×L候选组各自独立训练full和no_A，共30套冻结模型。上游拟合目标固定 `[192,3328)`；训练内A校准段固定 `[3328+128+2*delay,6048-delay+1)`。所有窗口使用保守统一隔离127+delay步，使窗口候选的拟合/评分日期、嵌套折和冷启动一致。每个外层校准块.6-.75/.75-.9/.9-1均去掉32+delay-1个冷启动位置；完整候选评分索引保存在每个FROZEN.json，选择阶段逐项验证相同。

每个L沿用原KC/W/A端到端训练内选择、273项A策略候选、200样本/20样本每系数支持门槛与FP64数值证书。跨窗口按选中完整结构的joint selection_MSE最小选窗，平分时选更短L；full和no_A分别选择。A未被接纳时使用父模型风险。所有候选和窗口在打开本轮开发数据数值前冻结，开发RMSE不参与选窗。L128重复训练的KCW选择及完整A父/子风险与上一轮一致，作为重训一致性锚点。

其他基线直接复用已冻结的逐点预测，逐值一致，未因本次PRISM结果重新挑参；神经预测仍为种子0/1/2算术均值。可复现的上一轮完整源代码、权重、账本和环境记录随包保存在baseline_reference/SRU_BASELINES_L128_20261023.tar.gz，SHA256为ca4d91e9fbcee61311ab6719ef82675ecf18efc367876aaca6f22198252f2fda。

本轮比较的是“基线固定128、PRISM训练内自选窗口且保留在线A”，窗口搜索预算和在线更新预算有差异。开发段此前已多轮使用；不能称为新的独立最终测试，也不能由这张表证明普遍优越性。两个早先实验均原样保留。

## 完整指标

'''+table+'''

指标在原始数据尺度计算，无裁剪。MAPE以百分数输出，仅排除abs(y)<=1e-8并记录数量；另有sMAPE、WAPE和bias。

## 产物与复现

- [选窗候选分数](WINDOW_CANDIDATES.csv)、[最终选窗与结构](results/selected_windows.csv)、[冻结门槛](ALL_SELECTIONS_FROZEN.json)、[预登记](REGISTRATION.json)。
- [全部指标](results/metrics.csv)、[54432行逐点长表](results/point_predictions_long.csv)、[Parquet长表](results/point_predictions_long.parquet)。
- [delay4逐点](results/point_predictions_D4.csv)、[delay8逐点](results/point_predictions_D8.csv)、[delay16逐点](results/point_predictions_D16.csv)。
- [PRISM与固定128描述性对照](results/prism_vs_fixed128.csv)，RMSE_change_pct负数代表本轮误差更低；结构选择也会变化，不是严格单因素效应。
- [独立复核](results/INDEPENDENT_VERIFICATION.json)；audits/内有全部选中模型未来X/未成熟y扰动及前缀一致性检查，误差容限1e-12且无新增拟合。
- runs/保存全部30套PRISM权重及所有KCW/A候选账本；基线逐种子指标直接随原文件保留。

在空结果目录保留源码、data/及baseline_reference/，原科学计算环境执行 `prism_search.py prepare`、`prism_search.py smoke`、`launch.py`。launch用3个单线程CPU工作进程训练，先冻结全部候选，再选窗，再评估；GPU无需重训。现有冻结模型可仅运行 `prism_search.py evaluate 4/8/16`。report.py只聚合、独立复核和绘图，不重新拟合。

iTransformer与TimeXer采用作者官方TimeXer仓库提交76011909357972bd55a27adba2e1be994d81b327实现，两篇论文均为2024年；不是截至2026年最新SOTA声明。所有文件校验和见SHA256SUMS.json。

![训练内选窗](results/window_selection.png)

![指标](results/metrics_comparison.png)

![delay4](results/point_predictions_D4.png)

![delay8](results/point_predictions_D8.png)

![delay16](results/point_predictions_D16.png)
'''
    (ROOT/'REPORT_CN.md').write_text(text,encoding='utf-8')
    (ROOT/'README.md').write_text('# SRU PRISM window selection\n\n[Report](REPORT_CN.md) | [Metrics](results/metrics.csv)\n',encoding='utf-8')
if __name__=='__main__':
    m,l,anchors=combine();verify(m,l,anchors);figures(m);report(m)
    dump(OUT/'COMPLETED.json',dict(status='COMPLETED',metric_rows=len(m),point_rows=len(l),no_retraining_in_report=True))
    print('REPORT_COMPLETED',flush=True)
