"""Box-Jenkins compensation with a frozen X-only Laguerre process model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
import pickle

import numpy as np

from laguerre_control import endogenous_gate, load_prefix
from laguerre_ecm import Dynamics, ExogenousMap, box_jenkins_deployment
from laguerre_process import CurvatureProblem, LinearProblem, Process, fixed_projection
from laguerre_screen import (
    CONFIGS, ENDO, WARMUP, append, digest, endo_fit, lag, read_tep, receipt,
    rmse, save_json, train_rows,
)
from laguerre_unified import full_h2s, head, pressure, release_errors, summarize_gate


NEAR = (4,8,12)
ALPHA = (.0005,.003,.018)
MIX = (0.,.05,.2,.5)
CURVE = (.008,.04,.16)
HALF_LIVES = (8,24,64)
OBSERVER = tuple(product((1,2),(.015,.06,.15)))
FRACTIONS = (.37,.48,.59,.66,.73)
H2S_TARGET = .0315
H2S_ENET = .0324230062745495
TEP_TARGET = .00265
TEP_ARX = {1:.002673568188608086,2:.0026913179764802863}
SOURCES = ("laguerre_bj.py","laguerre_process.py","laguerre_ecm.py","laguerre_screen.py",
           "laguerre_control.py","laguerre_unified.py","laguerre_aligned.py","laguerre_sparse_views.py",
           "laguerre_pressure.py")


def observer(near):
    return Dynamics(near,tuple((2**(-1/h),1) for h in HALF_LIVES))


@dataclass
class BoxJenkins:
    center: float
    endo_coef: np.ndarray
    exo_map: ExogenousMap
    process: Process
    exo_on: bool
    observer_on: bool
    observer_near: int
    observer_coef: np.ndarray

    def predict(self, data):
        design = self.exo_map.design(data.x)
        terms = self.process.components(design.raw)
        pure = self.process.location+terms.sum(axis=1)
        y0 = data.anchor()+data.endo(self.center)@self.endo_coef
        activity = np.clip(design.trust@self.process.weights,0.,1.) if self.exo_on else np.zeros(len(y0))
        gated = np.sum(design.trust*terms,axis=1)
        base = box_jenkins_deployment(y0,self.process.location,gated,activity,exo_admitted=self.exo_on)
        if self.exo_on and self.observer_on:
            mature = release_errors(data,pure,np.arange(len(data))>=WARMUP+data.delay)
            effect = observer(self.observer_near).batch(mature)@self.observer_coef
            y = box_jenkins_deployment(y0,self.process.location,gated,activity,effect,inn_admitted=True)
        else:
            y = base
        return dict(prediction=y,endo=y0,base=base,trust=activity,pure=pure)

    def stream(self,delay):
        return ProcessStream(self,delay)


class ProcessStream:
    def __init__(self, model, delay):
        if delay < 1:
            raise ValueError("The current benchmark stream uses a positive assay delay")
        self.model = model
        self.delay = delay
        self.i = 0
        self.previous = 0.
        self.endostate = ENDO.stream(())
        self.exostate = model.exo_map.stream()
        self.pending = np.zeros(delay)
        self.observer = observer(model.observer_near).stream(()) if model.exo_on and model.observer_on else None

    def step(self,x,anchor):
        m = self.model
        endo_delta = anchor-self.previous if self.i>self.delay else 0.
        self.previous = anchor
        endo = np.r_[1.,anchor-m.center,self.endostate.step(endo_delta)]
        safe = anchor+endo@m.endo_coef
        raw,trust = self.exostate.step(x)
        components = m.process.components(raw)
        pure = m.process.location+components.sum()
        rho = float(np.clip(trust@m.process.weights,0.,1.)) if m.exo_on else 0.
        q = trust@components
        k = self.i%self.delay
        e = anchor-self.pending[k] if self.i>=WARMUP+2*self.delay else 0.
        self.pending[k] = pure
        disturbance = self.observer.step(e)@m.observer_coef if self.observer is not None else 0.
        score = box_jenkins_deployment(safe,m.process.location,q,rho,disturbance,
                                       exo_admitted=m.exo_on,inn_admitted=m.observer_on)
        self.i += 1
        return score,safe,rho,pure


def blocks(data):
    first = WARMUP+data.delay
    breaks = (first+(len(data)-first)*np.array(FRACTIONS)).astype(int)
    edges = np.r_[breaks,np.linspace(breaks[-1],len(data),5,dtype=int)[1:]]
    all_ = tuple((int(lo),np.arange(lo,hi)) for lo,hi in zip(edges[:-1],edges[1:]))
    return all_[:2],all_[2:4],all_[4:]


def process_design(data,rows,r,original):
    dynamic = replace(original.dynamics,near=r)
    exo = ExogenousMap.fit(data.x,rows,dynamic,original.padding,original.width)
    return exo,exo.design(data.x)


def search_linear(data,original,early):
    grid = tuple(product(ALPHA,MIX))
    def evaluate(r):
        err = np.zeros(len(grid))
        n = 0
        for stop,test in early:
            rows = train_rows(data.delay,stop)
            scope = data.through(test[-1]+1)
            _,design = process_design(scope,rows,r,original)
            p = LinearProblem(design.raw,scope.truth,rows)
            for i,(alpha,mix) in enumerate(grid):
                fitted,_ = p.fit(alpha,mix)
                e = scope.truth[test]-fitted.predict(design.raw[test])
                err[i] += e@e
            n += len(test)
        records = [dict(near=r,alpha=alpha,mix=mix,inner_rmse=float(np.sqrt(v/n)))
                   for (alpha,mix),v in zip(grid,err,strict=True)]
        best = min(records,key=lambda s:s["inner_rmse"])
        print("linear",r,best,flush=True)
        return records
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [entry for batch in pool.map(evaluate,NEAR) for entry in batch]


def paired_process(data,stop,test,spec,original):
    rows = train_rows(data.delay,stop)
    scope = data.through(test[-1]+1)
    r = spec["near"]
    _,design = process_design(scope,rows,r,original)
    linear,audit = LinearProblem(design.raw,scope.truth,rows).fit(spec["alpha"],spec["mix"])
    curves = CurvatureProblem(design,scope.truth,rows,linear,r)
    return scope,design,linear,curves,audit


def tune_nonlinear(data,linear,original,early):
    loss = np.zeros(len(CURVE))
    n = 0
    for stop,test in early:
        section,design,_,problem,_ = paired_process(data,stop,test,linear,original)
        y = section.truth[test]
        for i,alpha in enumerate(CURVE):
            e = y-problem.fit(alpha).predict(design.raw[test])
            loss[i] += e@e
        n += len(test)
    return [dict(curvature=a,inner_rmse=float(np.sqrt(v/n))) for a,v in zip(CURVE,loss,strict=True)]


def nonlinear_admission(data,linear,curve,original,confirmation):
    gains = []
    for stop,test in confirmation:
        section,design,base,problem,_ = paired_process(data,stop,test,linear,original)
        y = section.truth[test]
        x = base.predict(design.raw[test])
        non = problem.fit(curve).predict(design.raw[test])
        gains.append((y-x)**2-(y-non)**2)
    return summarize_gate(gains,data.delay)


def upstream(data,stop,spec,curve,original,endo_on):
    rows = train_rows(data.delay,stop)
    mapping,design = process_design(data,rows,spec["near"],original)
    process,audit = LinearProblem(design.raw,data.truth,rows).fit(spec["alpha"],spec["mix"])
    active_linear = int(np.count_nonzero(process.theta[...,0]))
    count_nonlinear = 0
    if curve is not None:
        nonlinear = CurvatureProblem(design,data.truth,rows,process,spec["near"])
        process = nonlinear.fit(curve)
        count_nonlinear = int(np.prod(nonlinear.dims))
    center,endo,coeff = endo_fit(data,rows)
    if not endo_on:
        coeff[:] = 0
    safe = data.anchor()+endo@coeff
    view = process.components(design.raw)
    pure = process.location+view.sum(axis=1)
    rho = np.clip(design.trust@process.weights,0.,1.)
    mixed = box_jenkins_deployment(safe,process.location,np.sum(view*design.trust,axis=1),rho)
    model = BoxJenkins(center,coeff,mapping,process,True,False,1,np.zeros(4))
    return model,pure,mixed,safe,rho,dict(**audit,linear_active=active_linear,
                nonlinear_parameters=count_nonlinear,active_channels=int(np.count_nonzero(process.weights)))


def observer_fit(data,original,linear,curve,endo_on,early,nonlinear,outer):
    idx = np.arange(len(data))
    issued = np.zeros(len(data),dtype=bool)
    pure,base,safe,rho = (np.zeros(len(data)) for _ in range(4))
    for stop,test in (*early,*nonlinear,*outer):
        part = data.through(test[-1]+1)
        _,p,q,y0,g,_ = upstream(part,stop,linear,curve,original,endo_on)
        issued[test] = True
        pure[test],base[test],safe[test],rho[test] = p[test],q[test],y0[test],g[test]
    exo_gain = [(data.truth[t]-safe[t])**2-(data.truth[t]-base[t])**2 for _,t in outer]
    exo_gate = summarize_gate(exo_gain,data.delay)
    residual = data.truth-base
    # These released residuals use the issued X-only forecasts before support blending.
    mature = release_errors(data,pure,issued)
    features = {near:observer(near).batch(mature) for near in (1,2)}
    stop,test = early[-1]
    rows = np.flatnonzero(issued & (idx<stop-data.delay))
    trials = []
    for near,alpha in OBSERVER:
        b = head(features[near],residual,rho,rows,alpha)
        f = rho[test]*(features[near][test]@b)
        trials.append(dict(near=near,alpha=alpha,inner_rmse=rmse(f,residual[test])))
    pick = min(trials,key=lambda s:s["inner_rmse"])
    near,alpha = pick["near"],pick["alpha"]
    bank = features[near]
    gain = []
    for stop,test in outer:
        rows = np.flatnonzero(issued & (idx<stop-data.delay))
        b = head(bank,residual,rho,rows,alpha)
        effect = rho[test]*(bank[test]@b)
        gain.append(residual[test]**2-(residual[test]-effect)**2)
    gate = summarize_gate(gain,data.delay)
    gate["conditional_admission"] = gate["admitted"]
    gate["admitted"] = bool(gate["admitted"] and exo_gate["admitted"])
    m,_,_,_,_,audit = upstream(data,len(data),linear,curve,original,endo_on)
    rows = np.flatnonzero(issued & (idx<len(data)-data.delay))
    b = head(bank,residual,rho,rows,alpha) if gate["admitted"] else np.zeros(near+3)
    m = replace(m,exo_on=exo_gate["admitted"],observer_on=gate["admitted"],observer_near=near,observer_coef=b)
    return m,dict(exogenous_gate=exo_gate,observer_gate=gate,observer_grid=trials,
                  observer_choice=pick,observer_coefficients=b.tolist(),process=audit)


def fit(task,delay,source,prior,out):
    inputs = receipt(source,task,("train","development"))
    prev = json.loads((prior/"FROZEN.json").read_text())
    if prev["data"]!=inputs or (prev["task"],prev["delay"])!=(task,delay):
        raise ValueError("Original frozen split no longer matches")
    original = CONFIGS[task][prev["selected"]]
    if json.loads(json.dumps(asdict(original)))!=prev["configs"][prev["selected"]]:
        raise ValueError("Original support map parameters changed")
    data,dev = load_prefix(task,source,delay)
    early,confirmation,outer = blocks(data)
    g0 = endogenous_gate(data.through(early[0][0]))
    linear_grid = search_linear(data,original,early)
    linear = min(linear_grid,key=lambda x:x["inner_rmse"])
    nonlinear_grid = tune_nonlinear(data,linear,original,early)
    candidate = min(nonlinear_grid,key=lambda x:x["inner_rmse"])
    non_gate = nonlinear_admission(data,linear,candidate["curvature"],original,confirmation)
    curve = candidate["curvature"] if non_gate["admitted"] else None
    model,info = observer_fit(data,original,linear,curve,g0["admitted"],early,confirmation,outer)
    combined = append(data,dev)
    d = slice(len(data),len(combined))
    score_dev = model.predict(combined)
    dev_metrics = {key:rmse(score_dev[key][d],dev.truth) for key in ("pure","base","prediction")}
    out.mkdir(parents=True,exist_ok=False)
    artifact = out/"MODEL.pkl"
    artifact.write_bytes(pickle.dumps(model,protocol=5))
    def cuts(pairs):
        return [[start,int(rows[-1])+1] for start,rows in pairs]
    save_json(out/"FROZEN.json",dict(task=task,delay=delay,data=inputs,prior=digest(prior/"FROZEN.json"),
        source={s:digest(Path(__file__).with_name(s)) for s in SOURCES},model=digest(artifact),
        grids=dict(near=NEAR,alpha=ALPHA,mix=MIX,curve=CURVE,observer=OBSERVER,observer_half_lives=HALF_LIVES,
                   h2s_limit=H2S_TARGET,tep_limit=TEP_TARGET,tep_arx=TEP_ARX),
        partitions=dict(endo_until=early[0][0],inner=cuts(early),nonlinear=cuts(confirmation),outer=cuts(outer)),
        linear=linear,linear_scores=linear_grid,nonlinear_candidates=nonlinear_grid,
        nonlinear_choice=candidate,nonlinear_gate=non_gate,development=dev_metrics,**info))
    print("LOCK",out.name,linear,info["process"],"g_exo",info["exogenous_gate"]["admitted"],
          "nonlinear",non_gate["admitted"],"observer",info["observer_gate"]["t_stat"],dev_metrics,flush=True)


def score(task,delay,source,out,workbook,archive,paired,baselines):
    f = json.loads((out/"FROZEN.json").read_text())
    if (f["task"],f["delay"],f["data"])!=(task,delay,receipt(source,task,("train","development"))):
        raise ValueError("Score input differs from the frozen protocol")
    if any(digest(Path(__file__).with_name(n))!=h for n,h in f["source"].items()):
        raise ValueError("Frozen code changed")
    path = out/"MODEL.pkl"
    if digest(path)!=f["model"]:
        raise ValueError("Frozen model changed")
    if (out/"RESULTS.json").exists():
        raise FileExistsError(out/"RESULTS.json")
    m = pickle.loads(path.read_bytes())
    train,dev = load_prefix(task,source,delay)
    data = full_h2s(train,dev,workbook) if task=="sru" else append(append(train,dev),read_tep(source,"tail",delay))
    start = len(train)+len(dev)
    window = slice(start,len(data))
    b = m.predict(data)
    s = m.stream(delay)
    actual = np.array([s.step(x,y) for x,y in zip(data.x,lag(data.lab,delay))])
    expected = np.column_stack([b[key] for key in ("prediction","endo","trust","pure")])
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    errors = {key:rmse(b[key][window],data.truth[window]) for key in ("pure","base","prediction","endo")}
    result = dict(task=task,delay=delay,frozen=digest(out/"FROZEN.json"),tail_n=len(data)-start,
         tail_sha=digest(workbook if task=="sru" else source/"tail.npz"),tail_rmse=errors,
         mean_tail_trust=float(b["trust"][window].mean()),p10_tail_trust=float(np.quantile(b["trust"][window],.1)),
         max_stream_diff=float(np.max(abs(actual-expected))))
    if task=="sru":
        tests = pressure(m,data,train,start,archive,paired,baselines)
        for arm,readout in tests["stress"].items():
            readout["BoxJenkins"] = readout.pop("Unified")
        result["pressure"] = tests
        result["acceptance"] = dict(pure_beats_enet=errors["pure"]<=H2S_ENET,
              final_meets_target=errors["prediction"]<=H2S_TARGET,observer_admitted=m.observer_on,
              pressure_rmse_leads=all(p["BoxJenkins"]["rmse"]<=min(v["rmse"] for a,v in p.items() if a!="BoxJenkins")
                                     for p in tests["stress"].values()))
    else:
        result["acceptance"] = dict(beats_arx=errors["prediction"]<TEP_ARX[delay],
                                    meets_rmse_target=errors["prediction"]<=TEP_TARGET)
    save_json(out/"RESULTS.json",result)
    print("SCORE",out.name,errors,result["acceptance"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    for action in ("fit","score"):
        p = sub.add_parser(action)
        p.add_argument("task",choices=("sru","tep"))
        p.add_argument("delay",type=int)
        p.add_argument("source",type=Path)
        p.add_argument("out",type=Path)
        if action=="fit":
            p.add_argument("prior",type=Path)
        else:
            for option in ("workbook","archive","paired","baselines"):
                p.add_argument("--"+option,type=Path)
    a = parser.parse_args()
    if a.action=="fit":
        fit(a.task,a.delay,a.source,a.prior,a.out)
    else:
        if a.task=="sru" and not all((a.workbook,a.archive,a.paired,a.baselines)):
            parser.error("SRU scoring needs the H2S workbook and baseline replay paths")
        score(a.task,a.delay,a.source,a.out,a.workbook,a.archive,a.paired,a.baselines)


if __name__ == "__main__":
    main()

