"""Export paired ablation scores and publication-readable diagnostic figures."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ENTITIES = ("SRU_H2S", "CZ_Rod1", "CZ_Rod2")
LABELS = ("SRU H₂S", "CZ Rod1", "CZ Rod2")
COLOR = ("#166862", "#2c5b9e", "#a55031", "#795095")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ablations(rows):
    names = ("KCWA", "KCW", "no_C", "no_W", "no_KCW", "no_K_addend", "persistence")
    labels = ("Original KCWA", "Remove A", "Remove C, refit", "Remove W, refit",
              "Remove KCW, refit A", "Omit K direct term", "Persistence")
    fig, axes = plt.subplots(1,3,figsize=(13.4,4.8),layout="constrained")
    fig.suptitle("Original v0.6 KCWA | development RMSE")
    for ax,row,title in zip(axes,rows,LABELS,strict=True):
        values = [row["results"][key]["RMSE"] for key in names]
        yy = np.arange(len(names))
        colors = [COLOR[0],"#85a09d",COLOR[1],COLOR[1],COLOR[1],"#80708f","#9a9a9a"]
        ax.scatter(values,yy,c=colors,s=54,zorder=3)
        ax.axvline(values[0],ls="--",lw=1,color=COLOR[0],alpha=.5)
        if max(values)/min(values)>4:
            ax.set_xscale("log")
            ticks = (.013,.02,.04,.08,.11)
            ax.set_xticks(ticks,labels=[str(v) for v in ticks])
        ax.set(yticks=yy, yticklabels=labels if ax is axes[0] else [], title=title, xlabel="RMSE")
        ax.invert_yaxis()
        ax.grid(alpha=.22,axis="x")
    fig.savefig(ROOT/"ABLATION_RMSE.png",dpi=180)
    plt.close(fig)


def explanations(rows):
    fig, axes = plt.subplots(2,3,figsize=(14.2,8),layout="constrained")
    fig.suptitle("Exact decomposition of the original v0.6 KCWA")
    for ax,row,title in zip(axes[0],rows,LABELS,strict=True):
        stages = row["stages"]
        values = [stages[s]["mean_MSE_reduction"]*1e6 for s in "KCWA"]
        ax.bar(list("KCWA"),values,color=[COLOR[0] if v>=0 else COLOR[2] for v in values],width=.65)
        ax.axhline(0,color="#394347",lw=.7)
        ax.set(title=title,ylabel="Paired MSE reduction (×10⁻⁶)")
        ax.grid(alpha=.18,axis="y")
    sru, rod1, rod2 = rows
    ax = axes[1,0]
    ref = next(ch for ch in sru["K_channels"] if ch["is_reference"])
    for i,term in enumerate(ref["development_bins"]):
        y = term["fitted_response"]
        ax.plot(np.linspace(0,100,len(y)),y,label=f"{term['near']}–{term['far']-1} steps",color=COLOR[i])
    ax.set(title="SRU | u3 fitted K response",xlabel="Position within observed 5th–95th percentile range",ylabel="Bin contribution")
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1,1]
    values = [v["mean_MSE_reduction"]*1e6 for v in rod1["A"]["groups"].values()]
    ax.bar(rod1["A"]["groups"].keys(),values,color=[COLOR[0] if v>=0 else COLOR[2] for v in values])
    ax.axhline(0,color="#394347",lw=.7)
    ax.set(title="Rod1 | contribution of mature residual lags",xlabel="Lag from current issue (steps)",ylabel="A-only MSE reduction (×10⁻⁶)")
    ax = axes[1,2]
    channels = [(key,val) for key,val in rod2["C_channels"].items() if key!="intercept"]
    ax.barh([key.replace("_"," ") for key,_ in channels], [val["mean"] for _,val in channels],color=COLOR[2])
    ax.axvline(0,color="#394347",lw=.7)
    ax.set(title="Rod2 | C correction by channel",xlabel="Mean contribution on development")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(ROOT/"EXPLANATION.png",dpi=180)
    plt.close(fig)


def complexity(meta):
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    fig.suptitle("Native FP64 stream scaling | one pinned CPU core")
    series = (("KCWA, A=16", "fixed", "KCWA"), ("KCWA, A=L", "dense", "KCWA"),
              ("AR", "fixed", "AR"), ("ARX", "fixed", "ARX"))
    for ax,sweep,key,x_label,y_label in (
            (axes[0],lambda s:s["N"]==meta["fixed_window_N"],"L","History window L","Median ns per row"),
            (axes[1],lambda s:s["L"]==meta["fixed_N_window"],"N","Stream length N","Median total ms")):
        for color,(label,mode,model) in zip(COLOR,series,strict=True):
            lines = sorted((v for v in meta["scaling"] if v["A_mode"]==mode and sweep(v)),key=lambda v:v[key])
            field = "median_ns_per_row" if key=="L" else "median_total_ms"
            ax.plot([v[key] for v in lines],[v["models"][model][field] for v in lines],
                    marker="o",lw=1.8,label=label,color=color)
        ax.set(xscale="log",yscale="log",xlabel=x_label,ylabel=y_label)
        ticks = sorted({v[key] for v in meta["scaling"] if sweep(v)})
        ax.set_xticks(ticks,labels=ticks)
        ax.grid(alpha=.23,which="both")
    axes[0].legend(frameon=False,ncols=2,fontsize=9)
    fig.savefig(ROOT/"COMPLEXITY.png",dpi=180)
    plt.close(fig)


def run():
    ablation = [read(ROOT/e/"ABLATION.json") for e in ENTITIES]
    explanation = [read(ROOT/e/"EXPLANATION.json") for e in ENTITIES]
    with (ROOT/"ABLATION_SCORES.csv").open("w",encoding="utf-8",newline="") as handle:
        output = csv.writer(handle)
        output.writerow(("entity","N","arm","RMSE","MAE","R2","bias","RMSE_change_pct","CI95_lo","CI95_hi"))
        for row in ablation:
            for arm,value in row["results"].items():
                effect = value["relative_to_KCWA"]
                output.writerow((row["entity"],row["N"],arm,*(value[k] for k in ("RMSE","MAE","R2","bias")),
                                 effect["change_pct"],*effect["CI95"]))
    for name,rows in (("ABLATION_RESULTS.json",ablation),("EXPLANATION_RESULTS.json",explanation)):
        (ROOT/name).write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    ablations(ablation)
    explanations(explanation)
    complexity(read(ROOT/"COMPLEXITY.json"))


if __name__ == "__main__":
    run()
