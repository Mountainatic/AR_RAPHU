"""Validate the measured panels and export manuscript tables and vector figures."""

import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
CASES=("TEP_D1","H2S","Rod1","Rod2")
LABEL={"TEP_D1":"TEP D=1","H2S":r"SRU $\mathrm{H_2S}$","Rod1":"CZ Rod1","Rod2":"CZ Rod2"}
NAMES={"Full":"Full PRISM","No_difference":"Level views only","FIR_only":"FIR only",
       "Trust_one":"Trust fixed at one","HAC_off":"HAC forced open",
       "Level_state":"Level endogenous state"}
COLORS=("#24748a","#c45531","#5468a0","#799143","#b277a5","#4d565d")
plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"axes.spines.top":False,
                     "axes.spines.right":False,"pdf.fonttype":42,"savefig.facecolor":"white"})


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_output(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig,name):
    fig.savefig(OUT/name)
    plt.close(fig)


def tep_names():
    source=ROOT/"outputs/prism_tim_submission_20261127/source/tep/teprob.f"
    names={}
    for line in source.read_text(encoding="latin1").splitlines():
        m=re.match(r"^C\s+(XMEAS|XMV)\((\d+)\)\s+(.+)",line)
        if m:
            key=m[1]+m[2]
            names.setdefault(key,re.split(r"\s{2,}",m[3].strip())[0].strip())
    return names


def attribution(study):
    ranked=[]
    response=[]
    names=tep_names()
    rank_order={}
    for case in CASES:
        items=sorted(study[case]["attribution"],key=lambda x:-x["l2_one_training_sd"])
        rank_order[case]=items
        for i,x in enumerate(items,1):
            label=names.get(x["channel"],x["channel"])
            ranked.append(dict(protocol=case,rank=i,process=label,
                               **{k:v for k,v in x.items() if not isinstance(v,list)}))
            for lag,(lin,local,normal) in enumerate(zip(x["raw_linear"],x["raw_local"],x["per_sd_local"],strict=True)):
                response.append(dict(protocol=case,channel=x["channel"],lag=lag,
                                     gain_raw_linear=lin,gain_raw_local=local,gain_per_training_sd=normal))
    csv_output("CHANNEL_ATTRIBUTION.csv",ranked)
    csv_output("IMPULSE_RESPONSE.csv",response)
    fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.7),layout="constrained")
    for color,ch in zip(COLORS,rank_order["H2S"],strict=False):
        a.plot(ch["per_sd_local"],label=ch["channel"],color=color,lw=1.15)
    a.axvspan(5,9,color="#979797",alpha=.15)
    a.axhline(0,color="#9d9d9d",lw=.65)
    a.set(xlim=(0,63),xlabel="Lag (samples)",ylabel="Local response per training SD",title="SRU H₂S")
    a.legend(frameon=False,fontsize=6.7,ncol=2)
    highest=rank_order["TEP_D1"][:6][::-1]
    labels=[f'{c["channel"]}  {names.get(c["channel"],"")}' for c in highest]
    b.barh(labels,[r["l2_one_training_sd"] for r in highest],color=COLORS[0])
    b.set(xlabel="64-lag response L₂ norm per training SD",title="TEP D=1",xlim=(0,.013))
    b.set_xticks(np.arange(0,.013,.003))
    b.tick_params(axis="y",labelsize=6.6)
    save(fig,"IMPULSE_AND_ATTRIBUTION.pdf")


def traces(study):
    rows=[]
    summary=[]
    shown=[]
    for case in ("TEP_D1","H2S"):
        for event in study[case]["traces"]:
            summary.append(dict(protocol=case,scenario=event["label"],channel=event["channel"],
                view=event["view"],within_001_for_16=event["recovery_to_clean_001_for_16"],
                max_recovery_above_clean=event["max_recovery_above_clean"]))
            keys=list(event["series"])
            for values in zip(*event["series"].values(),strict=True):
                rows.append(dict(protocol=case,scenario=event["label"],channel=event["channel"],
                                 view=event["view"],**dict(zip(keys,values,strict=True))))
            if case=="TEP_D1" or event["label"]=="oscillation":
                shown.append((case,event))
    csv_output("TRUST_TRAJECTORIES.csv",rows)
    csv_output("TRUST_RECOVERY.csv",summary)
    fig,axs=plt.subplots(3,3,figsize=(7.1,5.1),sharex=True,layout="constrained")
    for i,(case,event) in enumerate(shown):
        s=event["series"]
        x=s["offset"]
        for j in range(3):
            axs[j,i].axvspan(0,512,color="#e9e9e9",zorder=-2)
            axs[j,i].axvline(512,c="#adadad",lw=.65,zorder=-1)
            axs[j,i].set_xlim(-64,767)
        axs[0,i].set_title(f'{LABEL[case]} · {event["label"]}\n{event["channel"]}, {event["view"]} view')
        axs[0,i].plot(x,s["sensor_fault_sigma"],c=COLORS[5],lw=.7)
        if event["label"]=="oscillation":
            detail=axs[0,i].inset_axes([.62,.12,.34,.35])
            part=(np.asarray(x)>=0)&(np.asarray(x)<14)
            detail.plot(np.asarray(x)[part],np.asarray(s["sensor_fault_sigma"])[part],c=COLORS[5],lw=.65)
            detail.set(xticks=[],yticks=[],xlim=(0,13),ylim=(-5,5))
            detail.spines[["top","right","left","bottom"]].set_visible(True)
            detail.patch.set_facecolor("white")
        axs[1,i].plot(x,s["rho_view"],c=COLORS[1],lw=.8,label="View")
        axs[1,i].plot(x,s["rho_active"],c=COLORS[0],lw=1,label="PRISM")
        axs[1,i].plot(x,s["clean_rho"],c=COLORS[5],lw=.75,ls="--",label="Clean reference")
        axs[1,i].set_ylim(-.03,1.05)
        for j,(k,lab) in enumerate((("prism_error","PRISM"),("arx_error","ARX"),("clipped_en_error","Clipped EN"))):
            axs[2,i].plot(x,s[k],c=COLORS[j],lw=.75,label=lab)
        axs[2,i].set_yscale("symlog",linthresh=.008 if case=="TEP_D1" else .025)
        v=.1 if case=="TEP_D1" and i==0 else .01 if case=="TEP_D1" else 1.
        axs[2,i].set_yticks([-v,0,v])
        axs[2,i].set_xlabel("Issue offset")
    axs[0,0].set_ylabel("Injection / train SD")
    axs[1,0].set_ylabel("Support trust")
    axs[2,0].set_ylabel("Prediction error\n(symlog)")
    axs[1,-1].legend(frameon=False,loc="lower right",fontsize=6)
    axs[2,-1].legend(frameon=False,loc="upper right",fontsize=6)
    save(fig,"FAULT_TRUST_TRAJECTORIES.pdf")


def native_tables(result):
    csv_output("MEMORY_SPAN.csv",result["sweep"])
    csv_output("LONG_STREAM_SUMMARY.csv",[dict(protocol=k,**v) for k,v in result["long"].items()])
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.68),layout="constrained")
    for i,(k,title) in enumerate((("median_ns","Median"),("p999_ns","P99.9"))):
        for j,m in enumerate(("PRISM","ARX")):
            points=[r for r in result["sweep"] if r["method"]==m]
            axs[i].plot([r["span"] for r in points],[r[k]/1000. for r in points],marker="o",
                        c=COLORS[j],label="PRISM (fixed kernel)" if m=="PRISM" else "Dense ARX",lw=1.35)
        axs[i].set(xscale="log",yscale="log",title=title,xlabel="Configured dense ARX history L")
        axs[i].set_xscale("log",base=2)
        axs[i].set_xticks(result["spans"],[str(k) for k in result["spans"]])
        axs[i].grid(True,axis="y",alpha=.25)
    axs[0].set_ylabel("Single issue latency (µs)")
    axs[1].legend(frameon=False)
    save(fig,"MEMORY_SPAN.pdf")


def ablation(study):
    table=[r"\begin{table*}[!t]",r"\centering",r"\caption{Clean and oscillation RMSE for controlled ablations}",
           r"\label{tab:architecture-ablation}",r"\small\setlength{\tabcolsep}{3.5pt}",r"\begin{tabular}{lrrrrrrrr}",r"\hline",
           " & "+" & ".join(r"\multicolumn{2}{c}{"+LABEL[c]+"}" for c in CASES)+r" \\",
           r"\cline{2-3}\cline{4-5}\cline{6-7}\cline{8-9}",
           "Variant & "+" & ".join(["Clean","Osc."]*4)+r" \\",r"\hline"]
    record=[]
    for arm,label in NAMES.items():
        formatted=[]
        for case in CASES:
            info=study[case]["ablation"][arm]
            x,y=info["clean_rmse"],info["oscillation_rmse"]
            formatted += [f"{x:.6f}",f"{y:.6f}"]
            record.append(dict(protocol=case,variant=arm,intervention=info["type"],clean_rmse=x,
                               oscillation_rmse=y,oscillation_shift_mae=info["oscillation_shift_mae"],
                               refit_seconds=info.get("refit_seconds")))
        table.append(label+" & "+" & ".join(formatted)+r" \\")
    table += [r"\hline",r"\end{tabular}",r"\par\smallskip",
              r"\parbox{0.96\textwidth}{\footnotesize The level-view, FIR, and endogenous-state variants refit only their restricted training readouts under the frozen recipe. The support and HAC rows switch the frozen deployment gates. CZ has zero observation delay and already uses level views.}",r"\end{table*}"]
    (OUT/"ABLATION.tex").write_text("\n".join(table)+"\n",encoding="utf-8")
    csv_output("ABLATION.csv",record)


def deployment_latex(rows):
    lookup={(r["protocol"],r["method"]):r for r in rows}
    methods=("PRISM","Persistence","AR","DPLS","ARX","ElasticNet","ElasticNet_clipped","LightGBM","GRU","Transformer")
    aliases={"ElasticNet":"Elastic Net","ElasticNet_clipped":"Clipped Elastic Net"}
    table=[r"\begin{table*}[!t]",r"\centering",r"\caption{Online history state, nonzero parameters, and one fixed-recipe fit}",
           r"\label{tab:deployment-inventory}",r"\small\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrrr}",r"\hline",
           r"& \multicolumn{3}{c}{TEP D=1} & \multicolumn{3}{c}{SRU $\mathrm{H_2S}$} \\",
           r"\cline{2-4}\cline{5-7}",r"Method & State (B) & Nonzeros & Fit (s) & State (B) & Nonzeros & Fit (s) \\",r"\hline"]
    for method in methods:
        cells=[aliases.get(method,method)]
        for case in ("TEP_D1","H2S"):
            m=lookup[case,method]
            gpu=r"$^\dagger$" if "NVIDIA" in m["timing_device"] else ""
            cells.extend([str(m["online_state_bytes"]),str(m["nonzero_total"]),
                          f'{m["refit_seconds"]:.3f}'+gpu])
        table.append(" & ".join(cells)+r" \\")
    table += [r"\hline",r"\end{tabular}",r"\par\smallskip",
              r"\parbox{0.96\textwidth}{\footnotesize Nonzeros are summed over the archived seeds. Fit time covers one seed with fixed settings, including input preparation. $^\dagger$ marks GPU fits. State represents the C struct for PRISM and dense linear readouts, and the causal history budget for the other methods. Immutable parameters and temporary activations are outside the state column.}",
              r"\end{table*}"]
    (OUT/"DEPLOYMENT_COMPARISON.tex").write_text("\n".join(table)+"\n",encoding="utf-8")


def run():
    prior=ROOT/"outputs/prism_laguerre_uniform_ecm_20261127"
    papers=ROOT/"outputs/prism_tim_final_tables_20261127"
    lock=read(prior/"SOURCE_LOCK.json")
    code={"study":sha(OUT/"frozen_study.py"),"inventory":sha(OUT/"deployment_inventory.py"),
          "native_complexity.py":sha(OUT/"native_complexity.py"),"long_stream.c":sha(OUT/"long_stream.c"),
          "generator":sha(Path(__file__))}
    assert all(sha(prior/"source"/f)==v for f,v in lock["code"].items())
    result={c:read(OUT/c/"STUDY.json") for c in CASES}
    for case,item in result.items():
        assert item["study"]==code["study"] and item["source_lock"]==sha(prior/"SOURCE_LOCK.json")
        assert item["score_archive"]==sha(papers/case/"RESULTS.json")
        assert item["commit"]==lock["commit"]
        for check in item["isolation"]["probes"]:
            assert check["max_input_shift"]>0. and check["max_shift"]==0.
    native=read(OUT/"native/COMPLEXITY.json")
    assert native["commit"]==lock["commit"] and native["source_lock"]==sha(prior/"SOURCE_LOCK.json")
    for key,value in native["code"].items():
        assert value==code[key]
    ablation(result)
    attribution(result)
    traces(result)
    native_tables(native)
    csv_output("ZERO_LEAKAGE.csv",[dict(protocol=c,**item) for c,val in result.items()
                                   for item in val["isolation"]["probes"]])
    context=read(prior/"RECEIPT.json")
    csv_output("ENDOGENOUS_LEVEL_COEFFICIENTS.csv",
               [dict(protocol=k,level_coefficient=v["level_coefficient"],exo_admitted=v["exo_admitted"])
                for k,v in context["datasets"].items()])
    source=ROOT/"outputs/prism_v2_delayed_benchmarks_20261127/evidence/CZ_CLOSED_LOOP.json"
    cz=read(source)
    csv_output("CZ_DEVELOPMENT_LAG_AUDIT.csv",
               [dict(protocol=r["data"],channel=tag,side=side,positive_lag=cz["positive_lag"],**stats)
                for r in cz["results"] for tag,cols in r["peaks"].items() for side,stats in cols.items()])
    rows=[]
    for case in ("TEP_D1","H2S"):
        table=read(OUT/"inventory"/f"{case}_INVENTORY.json")
        assert table["commit"]==lock["commit"] and table["archive_model"]==result[case]["frozen_model"]
        assert table["code"]==code["inventory"]
        assert set(table["methods"])=={"PRISM","Persistence","AR","ARX","ElasticNet","ElasticNet_clipped",
                                      "DPLS","GRU","LightGBM","Transformer"}
        for method,row in table["methods"].items():
            rows.append(dict(protocol=case,method=method,**row))
    csv_output("DEPLOYMENT_COMPARISON.csv",rows)
    deployment_latex(rows)
    names=("ABLATION.csv","ABLATION.tex","CHANNEL_ATTRIBUTION.csv","IMPULSE_RESPONSE.csv",
           "TRUST_TRAJECTORIES.csv","TRUST_RECOVERY.csv","ZERO_LEAKAGE.csv","MEMORY_SPAN.csv",
           "LONG_STREAM_SUMMARY.csv","ENDOGENOUS_LEVEL_COEFFICIENTS.csv","CZ_DEVELOPMENT_LAG_AUDIT.csv",
           "DEPLOYMENT_COMPARISON.csv","DEPLOYMENT_COMPARISON.tex","IMPULSE_AND_ATTRIBUTION.pdf",
           "FAULT_TRUST_TRAJECTORIES.pdf","MEMORY_SPAN.pdf")
    receipt=dict(algorithm_commit=lock["commit"],source=sha(prior/"SOURCE_LOCK.json"),code=code,
                 signed_main_tables=sha(papers/"RECEIPT.json"),native=sha(OUT/"native/COMPLEXITY.json"),
                 development_lag_source=sha(source),cases={c:sha(OUT/c/"STUDY.json") for c in CASES},
                 outputs={name:sha(OUT/name) for name in names})
    (OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
