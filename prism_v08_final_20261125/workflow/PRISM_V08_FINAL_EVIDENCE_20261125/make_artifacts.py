"""Render tables and paper figures from aggregate-only evidence."""

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT=Path(__file__).resolve().parent
ENTITIES={"SRU_H2S":"SRU H₂S","CZ_Rod1":"CZ Rod1","CZ_Rod2":"CZ Rod2"}
NAMES={"Persistence":"持续值","AR":"AR","ARX":"ARX","ElasticNet":"Elastic Net",
       "Transformer_expanded":"扩展 Transformer","PRISM_v08":"PRISM v0.8"}
MODELS=("PRISM_v08","AR","ARX","ElasticNet","Transformer_expanded")
COLORS=("#087F71","#596589","#C8662F","#935895","#456FA0")
MARKERS=("D","o","s","^","P")
METRICS=("RMSE","MAE","MSE","R2_y","R2_d","eta_pers_pct")


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def write_csv(path,header,rows):
    with (ROOT/path).open("w",encoding="utf-8-sig",newline="") as file:
        w=csv.writer(file)
        w.writerow(header); w.writerows(rows)


def point(elapsed,model):
    native=elapsed["times"]
    if model=="ElasticNet":
        return statistics.mean(min(native[f"ElasticNet_{i}"]["median_ns"],
                                   native[f"ElasticNetSparse_{i}"]["median_ns"]) for i in range(5))
    if model=="Transformer_expanded":
        return statistics.mean(seed["median_ns"] for seed in elapsed["torch"]["seeds"].values())
    return native[model]["median_ns"]


def theme():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.labelcolor":"#26354A",
        "axes.titlecolor":"#15263B","text.color":"#15263B","xtick.color":"#42546B","ytick.color":"#42546B",
        "axes.edgecolor":"#A9B5C2","grid.color":"#DCE3EC","svg.fonttype":"none"})


def finish(fig,base):
    for suffix in ("png","svg"):
        fig.savefig(ROOT/f"{base}.{suffix}",dpi=220,facecolor="white")
    plt.close(fig)


def pareto(tail,timing):
    fig,axes=plt.subplots(1,3,figsize=(10.9,4.05),sharex=True)
    fig.subplots_adjust(left=.085,right=.984,bottom=.32,top=.80,wspace=.46)
    fig.suptitle("Accuracy and single-issue CPU latency",fontsize=14,fontweight="bold",y=.97)
    data=[]
    for ax,(entity,title) in zip(axes,ENTITIES.items()):
        raw=tail[entity]["subsets"]["full_tail"]["scores"]
        row=timing["series"][entity]
        pairs={name:(point(row,name),raw[name]["mean"]["RMSE"]) for name in MODELS}
        front=[]; best=float("inf")
        for x,y in sorted(pairs.values()):
            if y<best:
                front.append((x,y)); best=y
        if front:
            ax.plot(*zip(*front),color="#7A8796",lw=1,ls="--",zorder=2)
        for name,color,marker in zip(MODELS,COLORS,MARKERS):
            x,y=pairs[name]
            on_front=(x,y) in front
            ax.scatter(x,y,s=68 if name=="PRISM_v08" else 52,color=color,marker=marker,zorder=4,
                       edgecolor="white",linewidth=.7,label=NAMES[name] if name not in ("ElasticNet","Transformer_expanded") else
                       ("Elastic Net" if name=="ElasticNet" else "Transformer"))
            data.append((entity,name,x,y,on_front))
        ax.set(title=title,xscale="log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(True,alpha=.75); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Full-tail RMSE")
    fig.supxlabel("Latency (ns/issue)",y=.205,fontsize=9.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=5,bbox_to_anchor=(.5,.092),frameon=False)
    fig.text(.5,.046,"Latency measured on development rows. Accuracy uses the fixed-model historical tail replay.",
             ha="center",fontsize=8,color="#495A6E")
    finish(fig,"PARETO")
    write_csv("PARETO_TABLE.csv",("entity","model","median_ns_or_mean_of_seed_medians","full_tail_RMSE","nondominated"),data)


def long_window(timing):
    fig,axes=plt.subplots(1,3,figsize=(10.9,3.95),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.32,top=.79,wspace=.23)
    fig.suptitle("Long-window SIMD runtime",fontsize=14,fontweight="bold",y=.97)
    rows=[]; first={}
    for ax,(entity,title) in zip(axes,ENTITIES.items()):
        rec=timing["series"][entity]
        L=[a["L"] for a in rec]
        first[entity]=next((a["L"] for a in rec if a["times"]["PRISM"]["median_ns"]<a["times"]["ARX"]["median_ns"]),None)
        for name,col,marker in (("PRISM","#087F71","D"),("ARX","#C8662F","s")):
            mid=[a["times"][name]["median_ns"] for a in rec]
            p95=[a["times"][name]["p95_ns"] for a in rec]
            ax.plot(L,mid,color=col,marker=marker,markersize=4,lw=1.75,label="PRISM v0.8 structure" if name=="PRISM" else "Dense ARX")
            ax.fill_between(L,mid,p95,color=col,alpha=.13)
            for a in rec:
                x=a["times"][name]
                size=a["prism_prefix_bytes"] if name=="PRISM" else a["arx_ring_bytes"]
                rows.append((entity,a["L"],name,a["N"],a["bins"],x["median_ns"],x["p95_ns"],size))
        ax.set(title=title,xscale="log",yscale="log")
        ax.set_xticks(L,[str(x) for x in L],rotation=32,fontsize=8)
        ax.grid(True,alpha=.75); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Latency (ns/issue)")
    fig.supxlabel("History length L",y=.194,fontsize=9.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,bbox_to_anchor=(.5,.089),frameon=False)
    fig.text(.5,.043,"Fixed K bin counts. Shading spans the median to P95. Runtime only on a repeated development stream.",
             ha="center",fontsize=8,color="#495A6E")
    finish(fig,"LONG_WINDOW")
    write_csv("LONG_WINDOW_TABLE.csv",("entity","L","kernel","rows","PRISM_bins","median_ns","p95_ns","ring_bytes"),rows)
    return first


def fnum(m,k):
    if k=="MSE":
        return f"{m[k]:.6e}"
    if k=="eta_pers_pct":
        return f"{m[k]:.2f}%"
    return f"{m[k]:.6f}"


def tables(tail):
    paragraphs=["# 冻结 v0.8 时间尾段六指标","",
      "五份 Elastic Net 与五种子扩展 Transformer 均逐份评分，表中为六项指标各自的算术均值。",
      "每个子集在同一组起点对所有模型配对评分。剔除历史重叠起点时，递推状态沿实际时间连续推进。",""]
    for e,obj in tail.items():
        for subset,panel in obj["subsets"].items():
            kind="完整尾段" if subset=="full_tail" else "剔除已知重叠起点"
            paragraphs.extend([f"## {ENTITIES[e]} · {kind} · n={panel['N']}","",
                 "| 模型 | RMSE | MAE | MSE | 水平 R² | 增量 R² | 持续值 MSE 改善 |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
            for key,rec in panel["scores"].items():
                m=rec["mean"]
                cells=" | ".join(fnum(m,k) for k in METRICS)
                paragraphs.append(f"| {NAMES[key]} | {cells} |")
            paragraphs.append("")
            if e=="SRU_H2S":
                paragraphs.extend(["同一批目标行在 2026-11-19 的 D=8 协议中曾评分。",""])
    (ROOT/"TAIL_TABLES_CN.md").write_text("\n".join(paragraphs)+"\n",encoding="utf-8")


def main():
    theme()
    tail=read("TAIL_RESULTS.json")
    tables(tail)
    pareto(tail,read("BENCH_PARETO.json"))
    print("First sampled crossover:",long_window(read("BENCH_SCALING.json")))


if __name__=="__main__":
    main()
