"""Stage-one fit and locked replay of the unified gated ECM."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle

import numpy as np

from laguerre_aligned import Alignment, aligned_dynamics, fit_aligned
from laguerre_control import endogenous_gate, load_prefix
from laguerre_ecm import Dynamics, ExogenousMap, admission, aligned_deployment, ridge, scale_columns
from laguerre_screen import (
    CONFIGS, ENDO, SRU_SOURCE, WARMUP, append, digest, endo_fit, lag, pulse,
    read_tep, receipt, rmse, save_json, sru_series, train_rows,
)


MODES = ("level","delta")
EXACT = (4,8,12)
LAG_L1 = (.006,.025,.060)
CURVE = (0.,.025)
INNOVATION_POLES = (.85,.94,.98)
INNOVATION_GRID = ((1,.03),(1,.12),(2,.03),(2,.12))
INNER = (.37,.50,.64)
OUTER_COUNT = 4
H2S_LIMIT = .03150
TEP_LIMIT = .00264
SOURCES = ("laguerre_unified.py","laguerre_aligned.py","laguerre_ecm.py","laguerre_screen.py",
           "laguerre_control.py","laguerre_sparse_views.py","laguerre_pressure.py")


def innovation_bank(near):
    return Dynamics(near,tuple((a,1) for a in INNOVATION_POLES))


@dataclass
class Unified:
    center: float
    endo_coef: np.ndarray
    exo_map: ExogenousMap
    joint: object
    allow_exo: bool
    allow_inn: bool
    innovation_near: int
    innovation_coef: np.ndarray

    def terms(self, data):
        endo = data.endo(self.center)
        anchor = data.anchor()
        e = endo@self.endo_coef
        if self.allow_exo:
            q,rho = self.joint.predict(self.exo_map.design(data.views()),endo)
            return anchor,e,q,np.clip(rho,0.,1.)
        return anchor,e,np.zeros(len(data)),np.zeros(len(data))

    def predict(self, data):
        anchor,e,q,rho = self.terms(data)
        raw = aligned_deployment(anchor,e,q,rho,exo_admitted=self.allow_exo)
        if self.allow_exo and self.allow_inn:
            mature = release_errors(data,raw,np.arange(len(data))>=WARMUP+data.delay)
            f = innovation_bank(self.innovation_near).batch(mature)@self.innovation_coef
            pred = aligned_deployment(anchor,e,q,rho,f,innovation_admitted=True)
        else:
            pred = raw
        return dict(prediction=pred,base=raw,endo=anchor+e,trust=rho)

    def stream(self, delay, channels):
        return UnifiedStream(self,delay,channels)


class UnifiedStream:
    def __init__(self, model, delay, channels):
        if delay < 1:
            raise ValueError("Stage-one stream uses positive measurement delay")
        self.model = model
        self.delay = delay
        self.tick = 0
        self.last_anchor = 0.
        self.history = np.zeros((delay,channels))
        self.pending = np.zeros(delay)
        self.endostate = ENDO.stream(())
        self.exostate = model.exo_map.stream() if model.allow_exo else None
        self.innstate = innovation_bank(model.innovation_near).stream(()) if model.allow_inn else None

    def step(self,x,anchor):
        k = self.tick%self.delay
        if self.tick == 0:
            self.history[:] = x
        change = anchor-self.last_anchor if self.tick > self.delay else 0.
        self.last_anchor = anchor
        endo = np.r_[1.,anchor-self.model.center,self.endostate.step(change)]
        e = endo@self.model.endo_coef
        q = rho = 0.
        if self.exostate is not None:
            v = np.column_stack((x,x-self.history[k])).ravel()
            raw,trust = self.exostate.step(v)
            j = self.model.joint
            rho = float(np.clip(trust@j.weights,0.,1.))
            q = np.einsum("gtp,gtp,g->",raw,j.theta,trust)+rho*(endo@j.adjustment)
        self.history[k] = x
        previous = self.pending[k]
        self.pending[k] = aligned_deployment(anchor,e,q,rho,exo_admitted=self.model.allow_exo)
        active = self.tick >= WARMUP+2*self.delay
        inn = self.innstate.step(anchor-previous if active else 0.)@self.model.innovation_coef if self.innstate is not None else 0.
        self.tick += 1
        y = aligned_deployment(anchor,e,q,rho,inn,exo_admitted=self.model.allow_exo,
                               innovation_admitted=self.model.allow_inn)
        return y,anchor+e,rho


@dataclass
class Prepared:
    center: float
    endo: np.ndarray
    endo_coef: np.ndarray
    mapping: ExogenousMap
    design: object
    anchor: np.ndarray
    endo_increment: np.ndarray
    increment: np.ndarray
    rows: np.ndarray


def prepare(data, stop, spec, original, endo_gate):
    rows = train_rows(data.delay,stop)
    center,endo,b = endo_fit(data,rows)
    if not endo_gate:
        b[:] = 0.
    views = data.views()
    mapping = ExogenousMap.fit(views,rows,aligned_dynamics(spec,original.dynamics),
                               original.padding,original.width)
    anchor = data.anchor()
    return Prepared(center,endo,b,mapping,mapping.design(views),anchor,endo@b,data.truth-anchor,rows)


def train_upstream(p,spec,original):
    j = fit_aligned(p.design,p.mapping.dynamics,p.endo,p.increment,p.endo_increment,p.rows,spec,original)
    q,trust = j.predict(p.design,p.endo)
    rho = np.clip(trust,0.,1.)
    anchor = p.anchor
    safe = anchor+p.endo_increment
    forecast = aligned_deployment(anchor,p.endo_increment,q,rho)
    return j,safe,forecast,rho


def folds(data):
    start = WARMUP+data.delay
    i = start+np.array(INNER).astype(float)*(len(data)-start)
    bounds = i.astype(int)
    a = tuple((lo,np.arange(lo,hi)) for lo,hi in zip(bounds[:-1],bounds[1:]))
    b = np.linspace(bounds[-1],len(data),OUTER_COUNT+1,dtype=int)
    return a,tuple((lo,np.arange(lo,hi)) for lo,hi in zip(b[:-1],b[1:]))


def search_geometry(data, mode, near, original, endo_gate, inner):
    arms = [Alignment(mode,near,l1,curve) for l1 in LAG_L1 for curve in CURVE]
    costs = np.zeros(len(arms))
    n = 0
    for stop,test in inner:
        part = data.through(test[-1]+1)
        p = prepare(part,stop,arms[0],original,endo_gate)
        for i,spec in enumerate(arms):
            _,_,prediction,_ = train_upstream(p,spec,original)
            e = part.truth[test]-prediction[test]
            costs[i] += e@e
        n += len(test)
    return [dict(spec=asdict(arm),inner_rmse=float(np.sqrt(value/n)))
            for arm,value in zip(arms,costs,strict=True)]


def release_errors(data,predicted,issued):
    return lag(np.where(issued,data.lab-predicted,0.),data.delay)


def head(feat,resid,rho,rows,alpha):
    x = rho[rows,None]*feat[rows]
    s = scale_columns(x)
    coef = ridge(x/s,resid[rows],np.full(len(s),alpha))
    return coef/s


def summarize_gate(blocks_,delay):
    result = admission(blocks_,delay)
    result["t_stat"] = result["mean"]/result["se"] if result["se"] else None
    result["positive_blocks"] = sum(m>0 for m in result["block_means"])
    result["block_count"] = len(result["block_means"])
    return result


def fit_chosen(data,prefix,spec,original,endo_gate,inner,outer):
    q = np.zeros(len(data))
    safe,rho = np.zeros(len(data)),np.zeros(len(data))
    issued = np.zeros(len(data),dtype=bool)
    idx = np.arange(len(data))
    audit = []
    for stop,test in (*inner,*outer):
        received = data.through(test[-1]+1)
        p = prepare(received,stop,spec,original,endo_gate)
        j,y0,prediction,trust = train_upstream(p,spec,original)
        q[test],safe[test],rho[test],issued[test] = prediction[test],y0[test],trust[test],True
        audit.append(j.audit)
    gains = [(data.truth[t]-safe[t])**2-(data.truth[t]-q[t])**2 for _,t in outer]
    gate_x = summarize_gate(gains,data.delay)
    mature = release_errors(data,q,issued)
    residual = data.truth-q
    state = {r:innovation_bank(r).batch(mature) for r in (1,2)}
    # Innovation tuning sees only the second inner block, with the first as its calibration.
    split,check = inner[-1]
    before = np.flatnonzero(issued & (idx<split-data.delay))
    local = []
    for near,alpha in INNOVATION_GRID:
        b = head(state[near],residual,rho,before,alpha)
        loss = rmse(rho[check]*(state[near][check]@b),residual[check])
        local.append(dict(near=near,alpha=alpha,inner_rmse=loss))
    choice = min(local,key=lambda x:x["inner_rmse"])
    near,alpha = choice["near"],choice["alpha"]
    gains = []
    for stop,test in outer:
        rows = np.flatnonzero(issued & (idx<stop-data.delay))
        b = head(state[near],residual,rho,rows,alpha)
        corrected = residual[test]-rho[test]*(state[near][test]@b)
        gains.append(residual[test]**2-corrected**2)
    gate_i = summarize_gate(gains,data.delay)
    gate_i["conditional_admitted"] = gate_i["admitted"]
    gate_i["admitted"] = bool(gate_x["admitted"] and gate_i["admitted"])
    rows = np.flatnonzero(issued & (idx<len(data)-data.delay))
    b = head(state[near],residual,rho,rows,alpha) if gate_i["admitted"] else np.zeros(near+3)
    p = prepare(data,len(data),spec,original,endo_gate)
    j,_,_,_ = train_upstream(p,spec,original)
    model = Unified(p.center,p.endo_coef,p.mapping,j,gate_x["admitted"],gate_i["admitted"],near,b)
    dev = slice(len(data),len(prefix))
    pred = model.predict(prefix)
    metrics = dict(spec=asdict(spec),exo_gate=gate_x,innovation_gate=gate_i,
                   innovation_candidates=local,innovation_selection=choice,innovation_coefficients=b.tolist(),
                   joint=j.audit,fold_audit=audit,
                   development_rmse=rmse(pred["prediction"][dev],prefix.truth[dev]),
                   development_upstream_rmse=rmse(pred["base"][dev],prefix.truth[dev]))
    return model,metrics


def fit(task,delay,source,prior,out):
    inputs = receipt(source,task,("train","development"))
    previous = json.loads((prior/"FROZEN.json").read_text())
    if previous["data"]!=inputs or (previous["task"],previous["delay"])!=(task,delay):
        raise ValueError("Previous split does not match")
    old_config = CONFIGS[task][previous["selected"]]
    if json.loads(json.dumps(asdict(old_config))) != previous["configs"][previous["selected"]]:
        raise ValueError("Support parameters changed since the original fit")
    data,dev = load_prefix(task,source,delay)
    prefix = append(data,dev)
    inner,outer = folds(data)
    endo_stop = int(inner[0][0])
    g0 = endogenous_gate(data.through(endo_stop))
    g0["prefix_stop"] = endo_stop
    def search(geom):
        m,r = geom
        result = search_geometry(data,m,r,old_config,g0["admitted"],inner)
        print(task,delay,m,r,min(x["inner_rmse"] for x in result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [x for part in pool.map(search,[(m,r) for m in MODES for r in EXACT]) for x in part]
    choices = {m:min((x for x in rows if x["spec"]["mode"]==m),key=lambda x:x["inner_rmse"]) for m in MODES}
    mode = min(choices,key=lambda m:choices[m]["inner_rmse"])
    models,metrics = {},{}
    for m in MODES:
        models[m],metrics[m] = fit_chosen(data,prefix,Alignment(**choices[m]["spec"]),old_config,
                                          g0["admitted"],inner,outer)
    out.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for name,model in models.items():
        dst = out/f"{name}.pkl"
        dst.write_bytes(pickle.dumps(model,protocol=5))
        hashes[name] = digest(dst)
    ticket = dict(task=task,delay=delay,inputs=inputs,mode=mode,inner_choices=choices,hyperparameters=dict(
        modes=MODES,exact=EXACT,lag_l1=LAG_L1,curve=CURVE,innovation_grid=INNOVATION_GRID,
        innovation_poles=INNOVATION_POLES,innovation_admission_t=1.6448536269514722,
        require_strict_block_majority=True,maximum_h2s_rmse=H2S_LIMIT,maximum_tep_rmse=TEP_LIMIT),
        inner_folds=[[int(a),int(t[-1])+1] for a,t in inner],outer_folds=[[int(a),int(t[-1])+1] for a,t in outer],
        data_prior=digest(prior/"FROZEN.json"),original_support=asdict(old_config),grid=rows,endo_gate=g0,
        fits=metrics,model_sha256=hashes,source={name:digest(Path(__file__).with_name(name)) for name in SOURCES})
    save_json(out/"FROZEN.json",ticket)
    print("FROZEN",out.name,mode,metrics[mode],flush=True)


def full_h2s(train,dev,workbook):
    import pandas as pd
    if digest(workbook)!=SRU_SOURCE:
        raise ValueError("Unexpected SRU workbook")
    # Materialize only five inputs and H2S, which occupies the sixth column.
    a = pd.read_excel(workbook,header=None,skiprows=1,usecols=range(6)).to_numpy(dtype=float)
    if a.shape!=(10081,6):
        raise ValueError("Unexpected H2S layout")
    data = sru_series(a[:,[5,0,1,2,3,4]])
    np.testing.assert_allclose(data.x[:8064],np.vstack((train.x,dev.x)),atol=2e-16,rtol=0)
    np.testing.assert_allclose(data.lab[:8064],np.r_[train.lab,dev.lab],atol=2e-16,rtol=0)
    return data


def apply_bias(series,begin,scale):
    x = series.x.copy()
    x[begin:begin+512] += 4*scale
    return series.with_x(x)


def pressure_metrics(pred,clean,y,endo,protected,trust=None):
    e = pred-y
    summary = dict(n=len(y),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),
                   shift_mae=float(np.mean(abs(pred-clean))),
                   max_distance_to_PRISM_endo=float(np.max(abs(pred-endo))))
    summary["max_fallback_error"] = summary["max_distance_to_PRISM_endo"] if protected else None
    if trust is not None:
        empty = trust==0
        summary.update(zero_trust_fraction=float(empty.mean()),first_step_trust=float(trust[0]),
                       max_error_on_zero_trust=float(np.max(abs(pred[empty]-endo[empty]))) if empty.any() else None)
    return summary


def pressure(model,series,train,start,archive,paired,baselines):
    from laguerre_pressure import predictors
    baseline,history = predictors(archive,paired,baselines)
    rows = np.arange(start,start+512)
    score_rows = np.arange(start,len(series))
    y = series.truth[rows]
    scale = train.x.std(axis=0)
    cases = dict(clean=series,oscillation=pulse(series,start,scale),bias=apply_bias(series,start,scale))
    predictions,baseline_rows = {},{}
    for name,data in cases.items():
        baseline_rows[name] = baseline(np.c_[data.lab,data.x],score_rows if name=="clean" else rows)
        predictions[name] = model.predict(data)
    clean = predictions["clean"]
    safe = clean["endo"][rows]
    result = {}
    for name in ("oscillation","bias"):
        measurements = predictions[name]
        panel = {"Unified":pressure_metrics(measurements["prediction"][rows],clean["prediction"][rows],y,safe,
                                           True,measurements["trust"][rows])}
        for arm,shocks in baseline_rows[name].items():
            clean_seeds = baseline_rows["clean"][arm][:,:len(rows)]
            panel[arm] = pressure_metrics(shocks.mean(axis=0),clean_seeds.mean(axis=0),y,safe,False)
            panel[arm]["mean_seed_rmse"] = float(np.sqrt(np.mean((shocks-y)**2,axis=1)).mean())
        result[name] = panel
    tail_truth = series.truth[score_rows]
    baserisk = {arm:dict(ensemble_rmse=rmse(seeds.mean(axis=0),tail_truth),
                         mean_seed_rmse=float(np.sqrt(np.mean((seeds-tail_truth)**2,axis=1)).mean()))
                for arm,seeds in baseline_rows["clean"].items()}
    return dict(stress=result,clean_baselines=baserisk,baseline_receipts=history,
                aggregation="Mean prediction of five seeds where applicable. Mean seed RMSE separately.")
    

def score(task,delay,source,out,workbook,archive,paired,baselines):
    f = json.loads((out/"FROZEN.json").read_text())
    if (f["task"],f["delay"],f["inputs"])!=(task,delay,receipt(source,task,("train","development"))):
        raise ValueError("Frozen source data changed")
    if any(digest(Path(__file__).with_name(name))!=value for name,value in f["source"].items()):
        raise ValueError("Frozen code changed")
    target = out/"RESULTS.json"
    if target.exists():
        raise FileExistsError(target)
    train,dev = load_prefix(task,source,delay)
    data = full_h2s(train,dev,workbook) if task=="sru" else append(append(train,dev),read_tep(source,"tail",delay))
    start = len(train)+len(dev)
    rows = slice(start,len(data))
    modes = {}
    models = {}
    for mode,expected in f["model_sha256"].items():
        b = out/f"{mode}.pkl"
        if digest(b)!=expected:
            raise ValueError("Frozen weights changed")
        model = pickle.loads(b.read_bytes())
        models[mode] = model
        batch = model.predict(data)
        stream = model.stream(delay,data.x.shape[1])
        actual = np.array([stream.step(x,a) for x,a in zip(data.x,lag(data.lab,delay))])
        expect = np.column_stack([batch[k] for k in ("prediction","endo","trust")])
        np.testing.assert_allclose(actual,expect,rtol=0,atol=2e-9)
        modes[mode] = dict(tail_rmse=rmse(batch["prediction"][rows],data.truth[rows]),
            upstream_tail_rmse=rmse(batch["base"][rows],data.truth[rows]),
            tail_active_trust_mean=float(batch["trust"][rows].mean()),
            tail_active_trust_p10=float(np.quantile(batch["trust"][rows],.10)),
            max_batch_stream_abs=float(abs(actual-expect).max()),n=len(data)-start)
    mode = f["mode"]
    selected = models[mode]
    evidence = dict(task=task,delay=delay,mode=mode,selected_diagnostics=f["fits"][mode],modes=modes,
                    frozen_sha256=digest(out/"FROZEN.json"),
                    tail_source_sha256=digest(workbook if task=="sru" else source/"tail.npz"))
    n = f["fits"][mode]["spec"]["near"]
    if task=="sru":
        evidence["pressure"] = pressure(selected,data,train,start,archive,paired,baselines)
        evidence["clean_accepted"] = bool(mode=="delta" and selected.allow_inn and modes[mode]["tail_rmse"]<=H2S_LIMIT)
    else:
        evidence["clean_accepted"] = bool(mode=="level" and n==4 and modes[mode]["tail_rmse"]<=TEP_LIMIT)
    save_json(target,evidence)
    print(out.name,mode,n,modes[mode]["tail_rmse"],selected.allow_exo,selected.allow_inn,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    for action in ("fit","score"):
        cmd = sub.add_parser(action)
        cmd.add_argument("task",choices=("sru","tep"))
        cmd.add_argument("delay",type=int)
        cmd.add_argument("source",type=Path)
        cmd.add_argument("out",type=Path)
        if action=="fit":
            cmd.add_argument("prior",type=Path)
        else:
            for name in ("workbook","archive","paired","baselines"):
                cmd.add_argument("--"+name,type=Path)
    a = p.parse_args()
    if a.action=="fit":
        fit(a.task,a.delay,a.source,a.prior,a.out)
    else:
        if a.task=="sru" and not all((a.workbook,a.archive,a.paired,a.baselines)):
            p.error("H2S scoring needs the H2S workbook and the four baseline sources")
        score(a.task,a.delay,a.source,a.out,a.workbook,a.archive,a.paired,a.baselines)


if __name__=="__main__":
    main()

