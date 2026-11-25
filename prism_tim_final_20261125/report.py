"""Render tables, publication figures and the final artifact receipt."""

import csv
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import core as c


ROOT=c.ROOT
FIG=ROOT/"figures"
COLORS={"PRISM":"#136f78","Persistence":"#9b604a","ARX":"#4a567a"}
NAMES={"CZ_Rod1":"CZ Rod1","CZ_Rod2":"CZ Rod2","SRU_H2S":"SRU H₂S"}
LABELS={"Persistence":"持续值","ElasticNet":"Elastic Net","Transformer":"扩展 Transformer"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":160,"svg.hashsalt":"prism-tim-final"})


def load(path):
    with (ROOT/path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def figure(fig,name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/(name+".svg"),bbox_inches="tight",metadata={"Date":None})
    fig.savefig(FIG/(name+".png"),bbox_inches="tight",dpi=240)
    plt.close(fig)


def horizon():
    r=load("E3/HORIZON.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.6),layout="constrained",sharex=True)
    for i,name in enumerate(c.RODS):
        axes[0,i].set_title(NAMES[name])
        for j,(metric,lab) in enumerate((("RMSE","RMSE"),("eta_pers_pct","MSE reduction vs. persistence (%)"))):
            ax=axes[j,i]
            for method in ("PRISM","ARX","Persistence"):
                valid=[d for d in r if d["data"]==name and d["split"]=="tail" and d["method"]==method]
                if len(valid)==len(c.HORIZONS):
                    ax.plot([int(d["H"]) for d in valid],[float(d[metric]) for d in valid],"o-",
                            color=COLORS[method],label=method,linewidth=1.6,markersize=4)
            ax.set_xticks(c.HORIZONS)
            ax.grid(axis="y",alpha=.18)
            if i==0:
                ax.set_ylabel(lab)
        axes[1,i].set_xlabel("Forecast horizon H (steps)")
    axes[0,0].legend(frameon=False,fontsize=8)
    figure(fig,"E3_common_target_horizons")


def drift():
    r=load("E3/ROD2_SYNTHETIC.csv")
    x=[int(v["offset"]) for v in r]
    record=c.read(ROOT/"E3/EVIDENCE.json")["synthetic"]
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.0),sharex=True,layout="constrained")
    for i,ax in enumerate(axes):
        ax.axvspan(record["shift_start"],record["shift_end"],color="#eadab7",alpha=.55,
                   label="Input offset applied" if i==0 else None)
        for event in record["transitions"]:
            ax.axvline(event["offset"],color="#765376",alpha=.48,linewidth=.8,linestyle="--")
    axes[0].plot(x,[float(d["trust"]) for d in r],color="#699caa",linewidth=.65,label="Instantaneous support")
    axes[0].plot(x,[float(d["mean"]) for d in r],color="#136f78",linewidth=1.5,label="128-row mean")
    axes[0].step(x,[int(d["fallback"]) for d in r],where="post",color="#765376",linewidth=1.2,label="Fallback active")
    axes[0].set_ylim(-.06,1.08)
    axes[0].set_ylabel("Support / state")
    axes[1].plot(x,[float(d["correction_off"]) for d in r],color="#9b604a",linewidth=.8,label="Guard off")
    axes[1].plot(x,[float(d["correction_on"]) for d in r],color="#136f78",linewidth=1,label="Guard on")
    axes[1].set_ylabel("Prediction minus anchor")
    axes[1].set_xlabel("Synthetic stream index")
    axes[0].legend(ncol=4,loc="lower center",bbox_to_anchor=(.5,1.0),fontsize=7.3,frameon=False)
    axes[1].legend(ncol=2,loc="lower center",bbox_to_anchor=(.5,1.0),fontsize=7.3,frameon=False)
    figure(fig,"E3_Rod2_step_drift")


def lengths():
    r=load("E4/WINDOW_SCALING.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True,layout="constrained")
    for ax,name in zip(axes,c.MAIN):
        ax.set_title(NAMES[name])
        for method in ("PRISM","ARX"):
            q=[d for d in r if d["data"]==name and d["method"]==method]
            x=[int(d["L"]) for d in q]
            ax.plot(x,[float(d["median_ns"]) for d in q],"o-",color=COLORS[method],label=method)
            ax.fill_between(x,[float(d["p10_ns"]) for d in q],[float(d["p90_ns"]) for d in q],
                            alpha=.13,color=COLORS[method])
        ax.set_xscale("log",base=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("History length L")
        ax.grid(axis="y",alpha=.2,which="both")
    axes[0].set_ylabel("Native latency (ns/row)")
    axes[0].legend(frameon=False)
    figure(fig,"E4_window_scaling")


def sensitivity():
    r=load("E5/SENSITIVITY.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout="constrained")
    for ax,name in zip(axes,c.MAIN):
        ax.set_title(NAMES[name])
        for split,style in (("development","s--"),("tail","o-")):
            q=[d for d in r if d["data"]==name and d["split"]==split]
            x=[float(d["enter"]) for d in q]
            ax.plot(x,[float(d["RMSE"]) for d in q],style,color=COLORS["PRISM"],label=split.title(),markersize=4)
        ax.axvline(.25,color="#444444",linestyle=":",linewidth=.9,label="Frozen threshold")
        ax.set_xticks(x)
        ax.set_xlabel("Fallback entry threshold")
        ax.set_ylabel("RMSE")
        ax.legend(frameon=False,fontsize=7.2)
        ax.grid(axis="y",alpha=.18)
    figure(fig,"E5_fallback_sensitivity")


def pareto():
    time=load("E4/NATIVE_LATENCY.csv")
    data=load("E1/METRICS.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.85),layout="constrained")
    marks={"PRISM":"D","AR":"o","ARX":"s","ElasticNet":"^"}
    offsets={"PRISM":(4,5),"AR":(4,5),"ARX":(4,5),"ElasticNet":(4,-12)}
    for ax,name in zip(axes,c.MAIN):
        ax.set_title(NAMES[name])
        for method in marks:
            latency=min(float(d["median_ns"]) for d in time if d["data"]==name and
                (d["method"]==method or method=="ElasticNet" and d["method"]=="ElasticNet_sparse"))
            error=next(float(d["RMSE"]) for d in data if d["data"]==name and d["split"]=="tail" and d["method"]==method)
            color=COLORS.get(method,"#757075")
            ax.scatter(latency,error,marker=marks[method],color=color,zorder=3)
            ax.annotate(LABELS.get(method,method),(latency,error),xytext=offsets[method],textcoords="offset points",
                        fontsize=7.2,color=color)
        ax.margins(x=.20,y=.18)
        ax.set_xlabel("Native latency (ns/row)")
        ax.set_ylabel("Tail RMSE")
        ax.grid(alpha=.18)
    figure(fig,"E4_native_precision_latency")


def table(names,rows):
    return "\n".join(["| "+" | ".join(map(str,names))+" |","| "+" | ".join(["---"]*len(names))+" |",
                      *("| "+" | ".join(map(str,r))+" |" for r in rows)])


def write_report():
    e1=c.read(ROOT/"E1/EVIDENCE.json")
    e2=load("E2/ABLATION.csv")
    e3=c.read(ROOT/"E3/EVIDENCE.json")
    e4=c.read(ROOT/"E4/EVIDENCE.json")
    e5=c.read(ROOT/"E5/EVIDENCE.json")
    tf=c.read(ROOT/"E1/Transformer/FROZEN.json")
    cfgs={name:c.read(ROOT/"E1"/name/"LightGBM/FROZEN.json")["selected"] for name in c.MAIN}
    title=rf"""# PRISM IEEE TIM 实验定稿证据

PRISM 最终代码清单 SHA-256 为 `{c.verify_code()}`。冻结内核在每 65,536 个全局时刻，将当前输入之前的过程历史和差分热身段重新累加，保留成熟残差、预测反馈和漂移状态。原有模型、门控和选型保持锁定。

| 角色 | 数据 | \(H,D,L\) | 主尾段目标索引 | 行数 |
| --- | --- | --- | --- | --- |
| 主实验 | CZ Rod1 | (16, 0, 128) | [16098, 20103) | 4005 |
| 主实验 | SRU H₂S | (0, 8, 64) | [8064, 10081) | 2017 |
| 漂移专项 | CZ Rod2 | (16, 0, 128) | [16517, 20627) | 4110 |

时刻 \(t\) 使用截至 \(t\) 的过程量和截至 \(t-D\) 的标签预测 \(y_{{t+H}}\)。成熟锚点取 \(y_{{t-D}}\)，已发预测的残差在 \(s+H+D\) 到达。水平 \(R_y^2\) 分母是水平真值的中心化平方和，增量 \(R_d^2\) 分母是锚点差分的中心化平方和。持续值改善率为 \(100(1-\mathrm{{MSE}}_m/\mathrm{{MSE}}_{{pers}})\)。

三套尾段属于同实体时间复核。SRU 的相同目标区间曾在 2026-11-19 的 D=8 试验评分。Rod1 历史重叠目标区间为 [18184, 19369)，Rod2 为 [18631, 19882)。两套 CZ 均另报剔除指定重叠目标的结果。这些子集仍来自同一根晶棒。现代基线依各自锁定的预测配方和相同信息时钟评分。
"""
    chunks=[title,"## E1 主基线与配对显著性"]
    for entry in e1:
        name=entry["data"]
        scores=entry["scores"]["tail"]["models"]
        s=lambda d,key: d["mean"][key]
        order=sorted(scores,key=lambda m:s(scores[m],"RMSE"))
        chunks.append(f"### {NAMES[name]}\n\n"+table(
           ("方法","次数","RMSE","MAE","MSE","水平 R²","增量 R²","持续值 MSE 改善"),
           [(LABELS.get(m,m),len(scores[m]["runs"]),f'{s(scores[m],"RMSE"):.6f}'+
             (f' ± {scores[m]["sd"]["RMSE"]:.6f}' if scores[m]["sd"]["RMSE"] is not None else ""),
             f'{s(scores[m],"MAE"):.6f}',f'{s(scores[m],"MSE"):.6e}',f'{s(scores[m],"R2_y"):.6f}',
             f'{s(scores[m],"R2_d"):.6f}',f'{s(scores[m],"eta_pers_pct"):.2f}%') for m in order]))
    chunks.append(rf"""Elastic Net、LightGBM、扩展 Transformer 按五个运行分别评分后取均值与样本标准差。Elastic Net 使用确定性坐标顺序，重复运行可完全一致。Rod1 Transformer 沿用相同协议的 3 个已锁定归档 GPU 种子和 2 个 CPU 补训种子。SRU Transformer 在训练前缀的两个向前折筛查 18 个候选，每候选 3 轮，前 3 名再跑至 8 轮。选中宽度、学习率和目标配方 `{tf['selected']['config']}`，按折内最佳轮数的中位数取整后固定训练 {tf['epochs']} 轮，五种子 CPU 重训。LightGBM 的选中配置为 Rod1 `{cfgs['CZ_Rod1']['config']}`，SRU `{cfgs['SRU_H2S']['config']}`，配置依次表示目标配方、叶数与最少叶样本、迭代轮数、L2 惩罚。

DM 使用同一尾段逐行的「基线平方误差减 PRISM 平方误差」，多种子先在每行对平方误差取均值。正统计量有利于 PRISM，检验使用 Bartlett HAC 和双侧正态近似，数据集内六项比较做 Holm 校正。另在 CSV 给出 \(L-1\) 带宽的未校正敏感性结果。历史重复接触和时间相关性限制这些 p 值的解释。 [Diebold–Mariano 原文](https://users.ssc.wisc.edu/~behansen/718/DieboldMariano1995.pdf)""")
    chunks.append(table(("数据","基线","N","HAC 滞后","DM","双侧 p","Holm p"),[
        (NAMES[e["data"]],LABELS.get(r["baseline"],r["baseline"]),r["n"],r["lag"],
         f'{r["DM"]:.3f}' if r["DM"] is not None else "—",
         f'{r["p"]:.3g}' if r["p"] is not None else "—",f'{r["p_Holm"]:.3g}' if r["p_Holm"] is not None else "—")
        for e in e1 for r in e["comparisons"]]))
    rod=e1[0]["scores"]["tail_nonoverlap_targets"]
    chunks.append(f'Rod1 非重叠子集共 {rod["n"]} 行，PRISM RMSE 为 {rod["models"]["PRISM"]["mean"]["RMSE"]:.6f}。'
                  '各方法的六指标、种子标准差和开发段结果见 [E1 指标](E1/METRICS.csv)。')
    chunks.append("## E2 核心消融\n\n"+table(("变体","Rod1 RMSE","Rod1 增量 R²","SRU RMSE","SRU 增量 R²"),[
        (label,*(f'{float(next(d[k] for d in e2 if d["data"]==name and d["split"]=="tail" and d["variant"]==arm)):.6f}'
         for name in c.MAIN for k in ("RMSE","R2_d")))
        for arm,label in (("full","完整 PRISM"),("A_only","仅 A"),("no_difference","去差分视图"),
                          ("no_A","去 A"),("no_rho","去可信度衰减 ρ"),("force_all","去结构接纳门控"))]))
    chunks.append("去 A 直接关闭最终校正。仅 A、去差分、去 ρ 和强制全开沿原校准协议重拟合受影响的下游。"
        "去 ρ 将信任统一置 1。强制全开仍保留可信域、硬回退和求解证书。Rod1 完整路由只接纳混合 A。"
        "SRU 保留了 K、C、W 与混合 A，其差分视图和 A 对尾段误差均有可见贡献。 [完整消融表](E2/ABLATION.csv)")
    f=next(r for r in e3["fallback"] if r["split"]=="tail" and r["method"]=="guard_on")
    sim=e3["synthetic"]
    h_lines=[]
    for name in c.RODS:
        r=[d for d in e3["common"] if d["data"]==name and d["split"]=="tail" and d["method"]=="PRISM"]
        h_lines.append((NAMES[name],r[0]["N"],*(f'{d["RMSE"]:.6f}' for d in r)))
    chunks.append("## E3 CZ 多步长与 Rod2 漂移回退\n\n"+table(("数据","同一目标行数",*c.HORIZONS),h_lines))
    chunks.append(f"""对每根棒和每个分区先取目标时刻交集，再回推各步长对应的发报时刻。各步长使用合法的各自持续值锚点。Rod2 H16 完整尾段有 {f['guard_rows']}/{f['N']} 行进入回退状态，两条路逐点最大差 {f['on_off_max_abs']:.1e}，RMSE 均为 {f['RMSE']:.6f}。该窗口内软衰减后的过程增量已经为零，真实尾段展示了路由进入回退状态，未测得硬回退的额外降误。

Rod2 合成轨迹使用重复的训练段过程片段，以五个过程通道同时加入阶跃偏置。干预从流索引 {sim['shift_start']} 开始，到 {sim['shift_end']} 结束，状态转换记录为 `{sim['transitions']}`。复制片段的接缝也出现自然过渡，图中用阴影标出实际施加偏置的区间。 [步长原表](E3/HORIZON.csv) · [回退原表](E3/ROD2_FALLBACK.csv) · [步长图](figures/E3_common_target_horizons.svg) · [漂移图](figures/E3_Rod2_step_drift.svg)""")
    native=[]
    for name in c.MAIN:
        d=e4[name]["matched"]["results"]
        native.append((NAMES[name],*(f'{d[m]["median_ns"]:.2f}' for m in
              ("PRISM","AR","ARX",e4[name]["matched"]["dense_or_sparse_fastest"]))))
    chunks.append("## E4 原生耗时与长时稳定性\n\n"+table(("数据","PRISM ns/行","AR","ARX","Elastic Net"),native))
    env=c.read(ROOT/"E4/REGISTRATION.json")
    chunks.append(rf"""在 {env['cpu']} 的 CPU {env['core']} 上，使用同一编译器（{env['compiler']}）与 FP64 AVX2/FMA 标志。每个方法计时覆盖当前行输入入环、模型读出与结果写回，初始历史建立在计时区间外。2048 行窗口内做 35 次配对交错重复。Elastic Net 同时测稠密镜像环和稀疏读出，正文列较快实现。长窗口扫描固定 PRISM 箱数并扩展远箱宽度，每次 8192 行计时包含一次重锚，稠密 ARX 使用同维度固定种子权重。计时衡量核函数本身，短窗结果依活动路由与主机负载而变。固定视图和箱数时，常规步成本相对窗口长度为 \(O(1)\)。重锚需要访问缓冲区内最近 \(L+H+D\) 行，每 65,536 步增加一次 \(O(L+H+D)\) 重算。

两套数据各回放 \(10^7\) 步，以每 256 步重锚的同一 FP64 内核为数值参考。以下最大误差为所测流的经验值。
"""+table(("数据","65,536 步重锚最大差","关闭重锚最大差"),[
    (NAMES[name],*(f'{e4[name]["stability"]["error"][period]["max_abs"]:.3e}' for period in ("65536","0")))
    for name in c.MAIN]))
    chunks.append("[耗时原表](E4/NATIVE_LATENCY.csv) · [窗口扩展曲线](figures/E4_window_scaling.svg) · "
                  "[原生精度–延迟图](figures/E4_native_precision_latency.svg) · [长时审计](E4/LONG_STABILITY.csv)")
    raw=c.read(ROOT/"E4/PREFIX_PROBE.json")
    chunks.append("另外，每 1024 步及重锚后首步，以独立 long double 求和复核原始过程窗口均值。\n\n"+
        table(("数据","重锚 65,536 最大窗口均值误差","关闭重锚最大窗口均值误差"),[
            (NAMES[name],*(f'{next(d["max_raw_mean_absolute_error"] for d in raw if d["data"]==name and d["period"]==period):.3e}'
               for period in (65536,0))) for name in c.MAIN])+
        "\n\n[前缀和探针](E4/PREFIX_PROBE.csv)")
    chunks.append("## E5 可解释与机制审计\n\n"+table(("数据","C/W","实际部署","截距梯度","最大 KKT 残差"),[
        (NAMES[d["data"]],d["module"],"是" if d["deployed"] else "否",
         "—" if d["intercept_gradient"] is None else f'{d["intercept_gradient"]:.3e}',f'{d["max_KKT"]:.3e}')
        for d in e5["ALIGNMENT"]]))
    chunks.append(table(("数据","Python/C 最大差","因果篡改最大差","加法最大误差","平方误差归因恒等式误差"),[
        (NAMES[d["data"]],*(f'{d[key]:.3e}' for key in ("python_C_max_abs","native_poison_max_abs",
          "additive_max_abs","SSE_identity_max_abs"))) for d in e5["CAUSAL"]]))
    groups={}
    for row in e5["BIAS_PRESSURE"]:
        groups.setdefault((row["data"],row["view"].split("/")[-1]),[]).append(row)
    chunks.append("在前缀后把全部过程通道一起抬高至原尺度外，等待历史窗口充分滚动，再比较每种视图。"+
        "\n\n"+table(("数据","视图","平均原始可信度","平均偏置后可信度","最大原始响应变化"),[
          (NAMES[name],view,f'{mean(d["clean_mean_rho"] for d in r):.6f}',f'{mean(d["shifted_mean_rho"] for d in r):.6f}',
           f'{max(d["max_abs_raw_response_change"] for d in r):.3e}') for (name,view),r in groups.items()]))
    gain_w=next(row["mean_SSE_gain"] for row in e5["ATTRIBUTION"] if row["data"]=="SRU_H2S" and row["stage"]=="W")
    chunks.append("敏感性逐一改变回退进入/恢复阈值，滞回区宽度保持 0.15。两套主尾段在五组阈值下均无硬回退，尾段分数完全相同。正文主结果使用预先冻结的 0.25/0.40。"
        "平方误差沿锚点、K、C、W、A 的既定顺序逐层相减，负值表示该层使该步平方误差增加。 "
        f"SRU 的 W 在冻结路由中被接纳，尾段的平均逐层平方误差改善为 {gain_w:.3e}。 "
        "[因果审计](E5/CAUSAL.csv) · [逐层归因汇总](E5/ATTRIBUTION.csv) · [偏置压力表](E5/BIAS_PRESSURE.csv) · "
        "[阈值敏感性图](figures/E5_fallback_sensitivity.svg)")
    (ROOT/"REPORT_CN.md").write_text("\n\n".join(chunks)+"\n",encoding="utf-8")


def finalize():
    bound=c.read(c.LOCK)
    assert c.model_sources()==bound["inherited"]
    assert {str(h):c.archived(h) for h in c.HORIZONS}==bound["parent_models"]
    for plot in (horizon,drift,lengths,sensitivity,pareto):
        plot()
    write_report()
    files=[ROOT/"REPORT_CN.md",c.LOCK,*FIG.iterdir()]
    for module in ("E1","E2","E3","E4","E5"):
        files.extend(path for path in sorted((ROOT/module).glob("*.csv")) if path.name!="ROD2_SYNTHETIC.csv")
        files.extend(ROOT/module/k for k in ("EVIDENCE.json","REGISTRATION.json"))
    for group in (ROOT/"E1/Transformer",*(ROOT/"E1"/name/"LightGBM" for name in c.MAIN)):
        files.append(group/"FROZEN.json")
    files.extend((ROOT/"E1/Transformer/REGISTRATION.json",ROOT/"E1/Transformer/CAUSAL.json"))
    files.extend(ROOT/"E4"/name for name in ("PREFIX_PROBE.json","PREFIX_PROBE_REGISTRATION.json"))
    for name in ("core.py","stream.c","e1.py","e4.py","evidence.py","tf_sweep.py","report.py",
                 "arx_mirror.c","elastic_sparse.c","prefix_probe.py","prefix_probe.c"):
        files.append(ROOT/name)
    baser={str(path.relative_to(ROOT)):c.sha(path) for path in sorted(files)}
    inputs=[c.SUITE/name/"linear/PREDICTIONS.npz" for name in c.MAIN]
    inputs.extend((c.SUITE/"CZ_Rod1/neural/PREDICTIONS.npz",ROOT/"E1/Transformer/PREDICTIONS.npz"))
    c.x.bind(ROOT/"RESULTS_LOCK.json",dict(final_code_lock=c.verify_code(),
        E2_weight_receipt=c.sha(ROOT/"E2/FROZEN.json"),
        private_input_receipts={str(path.relative_to(c.BASE)):c.sha(path) for path in inputs},files=baser))
    with ZipFile(ROOT/"PRISM_TIM_FINAL_EVIDENCE.zip","w",compression=ZIP_DEFLATED,compresslevel=8) as bundle:
        for path in (*sorted(files),ROOT/"RESULTS_LOCK.json"):
            bundle.write(path,path.relative_to(ROOT))
    print("RESULTS_LOCK",c.sha(ROOT/"RESULTS_LOCK.json"),flush=True)


if __name__=="__main__":
    finalize()
