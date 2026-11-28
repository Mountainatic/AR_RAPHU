"""Measure memory-span scaling and long-state score agreement on frozen C kernels."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import numpy as np


SPANS=(64,128,256,512,1024,2048)
PERIOD=4096
WARMUP=4096
REPEATS=32
FLAGS=("-std=c11","-O3","-mavx2","-mfma","-ffp-contract=off")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Input:
    def __init__(self,x,anchor):
        self.x=x
        self.values=anchor

    def anchor(self):
        return self.values

    def __len__(self):
        return len(self.values)


def load(a,task):
    from laguerre_screen import lag,read_tep,read_sru_prefix
    if task=="TEP_D1":
        ds=read_tep(a.tep,"train",1)
    else:
        ds=read_sru_prefix(a.sru,"train")
    idx=np.arange(PERIOD+WARMUP)%PERIOD
    return Input(ds.x[idx],lag(ds.lab,ds.delay)[idx])


def copy_reference(a,task,model):
    result=read(a.closure/f"{task}_NATIVE.json")
    assert result["commit"]==read(a.lock)["commit"]
    assert result["source_lock"]==digest(a.lock)
    group=result["cases"][task]
    orig=a.closure/"native_models"/task
    for f,sha in (("model.h",group["native_parameters"]),("parameters.h",group["native_readout"]),
                  ("stream.c",result["kernel"]),("stream.h",result["native_header"])):
        assert digest(orig/f)==sha
        shutil.copy2(orig/f,model/f)


def coefficients(a,L,channels):
    from native_audit import literal,sizes
    path=a.closure/"TEP_D1/cpu/linear.pkl"
    assert digest(path)==read(path.parent/"FIT.json")["weights"][path.name]
    scale,reg,loc,sd=pickle.loads(path.read_bytes())["ARX"]
    coef=np.asarray(reg.coef_)*sd/scale.scale_
    bias=loc+float(reg.intercept_)*sd-scale.mean_@coef
    y=coef[:4].copy()
    y[0]-=y[1:].sum()
    lag=(coef[4+channels:].reshape(-1,channels)/4.).repeat(4,axis=0)
    lag[0]+=coef[4:4+channels]
    if L>64:
        extra=1e-5*np.random.default_rng(20261128).standard_normal((L-64,channels))
        lag=np.vstack((lag,extra))
    assert len(lag)==L
    meta=sizes(CHANNELS=channels,YHIST=4,XHIST=L,BASE_DELTA=True)
    return meta+literal("baseline_bias","",bias)+literal("baseline_y","[YHIST]",y)+literal(
        "baseline_x","[XHIST][CHANNELS]",lag)


def run(a):
    sys.path[:0]=[str(a.source),str(a.closure)]
    from laguerre_blind_so2 import verify_phase_one
    from native_audit import Runner
    lock=verify_phase_one(a.lock)
    assert lock["commit"]=="7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c"
    runner=Runner(None,a.core)
    report=dict(commit=lock["commit"],source_lock=digest(a.lock),platform=runner.platform(),
                spans=SPANS,sweep=[],long={})
    a.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out) as path:
        temp=Path(path)
        for task in ("TEP_D1","H2S"):
            model=temp/task
            model.mkdir()
            copy_reference(a,task,model)
            data=load(a,task)
            if task=="TEP_D1":
                executable=model/"benchmark"
                runner.call("gcc",*FLAGS,model/"stream.c","-lm","-o",executable)
                # One fixed PRISM kernel is measured in each matched ARX span job.
                for L in SPANS:
                    directory=temp/f"L{L}"
                    directory.mkdir()
                    arx=runner.executable(directory,coefficients(a,L,data.x.shape[1]))
                    for name,exe in (("PRISM",executable),("ARX",arx)):
                        _,stats=runner.replay(exe,data,WARMUP,REPEATS)
                        report["sweep"].append(dict(span=L,method=name,**stats))
                        print(L,name,stats,flush=True)
            long_bin=model/"long_stream"
            c=Path(__file__).with_name("long_stream.c").resolve()
            runner.call("gcc",*FLAGS,"-DSTREAM_LIBRARY","-I",model,model/"stream.c",c,"-lm","-o",long_bin)
            inp=model/"period.bin"
            np.c_[data.values[:PERIOD],data.x[:PERIOD]].astype("<f8").tofile(inp)
            output=runner.call("taskset","-c",a.core,long_bin,inp,capture=True)
            ledger=list(csv.DictReader(io.StringIO(output)))
            cols=("prediction_max_abs","endo_max_abs","trust_max_abs")
            assert len(ledger)==99 and all(row["points"]=="128" for row in ledger)
            deltas={k:max(float(row[k]) for row in ledger) for k in cols}
            (a.out/f"{task}_LONG.csv").write_text(output,encoding="utf-8",newline="\n")
            report["long"][task]=dict(issues=1000000,local_reset_history=4096,checkpoints=len(ledger),
                                       checked_points=len(ledger)*128,**deltas)
            print(task,report["long"][task],flush=True)
    report["code"]={n:digest(Path(__file__).parent/n) for n in ("native_complexity.py","long_stream.c")}
    (a.out/"COMPLEXITY.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("source","lock","closure","tep","sru","out"):
        p.add_argument(n,type=Path)
    p.add_argument("--core",type=int,default=126)
    run(p.parse_args())
