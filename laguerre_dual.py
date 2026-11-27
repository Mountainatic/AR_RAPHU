"""Late-fold, dual-view Laguerre residual readout with honest nested HAC admission."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import norm

from laguerre_anchored import ALPHA, ENDOGENOUS_RIDGE, FLOOR, SOURCES as PRIOR, X_BANKS, cuts
from laguerre_bj import CURVE, MIX, NEAR, TEP_ARX
from laguerre_control import load_prefix
from laguerre_dual_views import (ENDO_ALPHA, ENDO_MIX, ENDO_NEAR, DualModel, forecast,
                                 independent_taps, mature_problem, prepare, readout)
from laguerre_screen import CONFIGS, append, digest, lag, pulse, read_tep, receipt, rmse, save_json, train_rows
from laguerre_unified import apply_bias, full_h2s, pressure_metrics, summarize_gate


WINDOW = 64
SCORE_FOLDS = ((.70,.85),(.85,1.))
PARITY = 1e-14
SRU_TARGET, SRU_BAND = .0326, .0332
SOURCES = ("laguerre_dual.py","laguerre_dual_views.py",*PRIOR)


def late_folds(data, window):
    rows = train_rows(data.delay,len(data))
    splits = [(int(lo*len(rows)),int(hi*len(rows))) for lo,hi in SCORE_FOLDS]
    return tuple((int(rows[a])-window, rows[a:b]) for a,b in splits)


def select_safe(data, folds):
    error = np.zeros((len(ENDO_NEAR), len(ENDO_ALPHA)))
    n = y0loss = 0
    for stop, t in folds:
        part = data.through(t[-1]+1)
        rows = train_rows(data.delay, stop)
        anchor = part.anchor()
        y = part.truth[t]
        n += len(t)
        y0loss += np.sum((y - anchor[t])**2)
        for i, near in enumerate(ENDO_NEAR):
            _, _, state, problem = mature_problem(part, rows, near)
            for j, alpha in enumerate(ENDO_ALPHA):
                intercept, beta, _ = problem.whole(alpha, ENDO_MIX)
                error[i, j] += np.sum((y - (anchor[t] + intercept + state[t] @ beta))**2)
    i, j = np.unravel_index(np.argmin(error), error.shape)
    return dict(near=ENDO_NEAR[i], alpha=ENDO_ALPHA[j], mix=ENDO_MIX, admitted=bool(error[i, j]<y0loss),
                risk=float(np.sqrt(error[i, j]/n)), persistence=float(np.sqrt(y0loss/n)))


def select(data, config, window):
    folds = late_folds(data,window)
    safe = select_safe(data,folds)
    grid = tuple(product(ALPHA,MIX))

    def scan(near):
        chunks = []
        for stop,t in folds:
            part = data.through(t[-1]+1)
            f = prepare(part,stop,near,safe,config)
            loss = []
            for alpha,mix in grid:
                j,_ = readout(f,alpha,mix)
                diff = part.truth[t]-forecast(f,j,t)
                loss.append(diff@diff)
            chunks.append(np.array(loss))
        gaps = np.array([len(t) for _,t in folds])
        sse = np.stack(chunks,axis=1)
        return [dict(near=near,alpha=a,mix=m,rmse=float(np.sqrt(v.sum()/gaps.sum())),
                     fold_rmse=np.sqrt(v/gaps).tolist()) for (a,m),v in zip(grid,sse,strict=True)]

    with ThreadPoolExecutor(max_workers=3) as pool:
        audit = [row for group in pool.map(scan,NEAR) for row in group]
    spec = min(audit,key=lambda item:item["rmse"])
    costs = []
    for stop,t in folds:
        part = data.through(t[-1]+1)
        f = prepare(part,stop,spec["near"],safe,config)
        gains = []
        for c in (None,*CURVE):
            j,_ = readout(f,spec["alpha"],spec["mix"],c)
            gains.append((part.truth[t]-forecast(f,j,t))**2)
        costs.append(gains)
    candidate = int(np.argmin([np.mean(x) for x in costs[0][1:]]))+1
    losses = [block[0]-block[candidate] for block in costs]
    gate = summarize_gate(losses,data.delay)
    curve = CURVE[candidate-1] if gate["admitted"] else None
    bounds = [dict(fit_last=int(train_rows(data.delay,stop)[-1]),start=int(t[0]),stop=int(t[-1])+1)
              for stop,t in folds]
    return safe,spec,curve,dict(grid=audit,folds=bounds,optional_curvature=gate)


def fit(task,delay,source,out,authority):
    streams = receipt(source,task,("train","development"))
    old = json.loads((authority/"FROZEN.json").read_text())
    if old["data"]!=streams or (old["task"],old["delay"])!=(task,delay):
        raise ValueError("Original data receipt changed")
    original = CONFIGS[task][old["selected"]]
    if json.loads(json.dumps(asdict(original)))!=old["configs"][old["selected"]]:
        raise ValueError("Original envelope changed")
    train,dev = load_prefix(task,source,delay)
    safe,spec,curve,search = select(train,original,WINDOW)
    print("FINAL INNER",out.name,spec,flush=True)
    _,_,edges = cuts(train)
    gains,nested = [],[]
    for stop,t in edges:
        # The outer labels cannot participate in fitting the inner selection for this fold.
        e,s,c,local = select(train.through(stop),original,WINDOW)
        data = train.through(t[-1]+1)
        f = prepare(data,stop,s["near"],e,original)
        j,_ = readout(f,s["alpha"],s["mix"],c)
        y = f.target[t]
        gain = (y-f.safe_hat[t])**2-(y-forecast(f,j,t))**2
        gains.append(gain)
        nested.append(dict(outer=[stop,int(t[-1])+1],inner=local["folds"],spec=s,endo=e,curve=c,
                           mean_gain=float(gain.mean())))
        print("OUTER",out.name,nested[-1],flush=True)
    admission = summarize_gate(gains,delay)
    f = prepare(train,len(train),spec["near"],safe,original)
    joint,fit_receipt = readout(f,spec["alpha"],spec["mix"],curve)
    model = DualModel(f.safe,f.mapping,joint,delay,admission["admitted"])
    combined = append(train,dev)
    predicted = model.predict(combined)
    rows = slice(len(train),len(combined))
    d = model.mapping.design(combined.views())
    a = combined.anchor()
    e = model.safe.dynamics.batch(a)
    opened,_ = joint.mix(d.raw,np.ones_like(d.trust),e,model.safe.predict(a,e))
    devscore = dict(deploy=rmse(dev.truth,predicted["prediction"][rows]),
                    all_trusted=rmse(dev.truth,opened[rows]))
    fit_receipt["train_ungated_rmse"] = rmse(f.target[f.rows],forecast(f,joint,f.rows,True))
    near_mask = independent_taps(spec["near"],delay)
    out.mkdir(parents=True,exist_ok=False)
    state = out/"MODEL.pkl"
    state.write_bytes(pickle.dumps(model,protocol=5))
    save_json(out/"FROZEN.json",dict(task=task,delay=delay,data=streams,
        authority=digest(authority/"FROZEN.json"),model=digest(state),
        sources={name:digest(Path(__file__).with_name(name)) for name in SOURCES},
        grids=dict(window=WINDOW,late_fractions=SCORE_FOLDS,near=NEAR,alpha=ALPHA,mix=MIX,
                   curve=CURVE,endo_near=ENDO_NEAR,endo_alpha=ENDO_ALPHA,endo_mix=ENDO_MIX,
                   joint_endo_ridge=ENDOGENOUS_RIDGE,floor=FLOOR,x_banks=X_BANKS,parity=PARITY,
                   sru_target=SRU_TARGET,sru_band=SRU_BAND,
                   exploratory_tail=False),
        parameters=spec,safe=safe,curvature=curve,linear_taps=near_mask.tolist(),
        search=search,outer_nested=nested,exogenous_gate=admission,fit=fit_receipt,development=devscore,
        endo_level_coef=float(f.safe.beta[-1]),training_issues=[int(f.rows[0]),int(f.rows[-1])+1]))
    print("LOCK",out.name,admission["t_stat"],admission["positive_blocks"],devscore,flush=True)


def paired_dm(diff, delay):
    n = len(diff)
    centered = diff-diff.mean()
    bandwidth = min(n-1,max(delay-1,int(4*(n/100)**(2/9))))
    v = centered@centered/n
    for k in range(1,bandwidth+1):
        v += 2*(1-k/(bandwidth+1))*(centered[k:]@centered[:-k])/n
    se = np.sqrt(max(0,v)/n)
    t = float(diff.mean()/se) if se else None
    return dict(n=n,lag=bandwidth,loss_gain=float(diff.mean()),t=t,
                p=float(2*norm.sf(abs(t))) if t is not None else None)


def pressure_and_dm(model,data,train,start,archive,paired,baseline_dir):
    from laguerre_pressure import predictors
    baseline,receipts = predictors(archive,paired,baseline_dir)
    clean = model.predict(data)
    tail = np.arange(start,len(data))
    pulse_rows = tail[:512]
    pulse_stop = slice(start,start+512)
    y = data.truth[tail]
    actual_loss = (clean["prediction"][tail]-y)**2
    candidates = baseline(np.c_[data.lab,data.x],tail)
    dm = {}
    for name,seeds in candidates.items():
        loss = np.mean((seeds-y)**2,axis=0)
        comparison = paired_dm(loss-actual_loss,data.delay)
        comparison["mean_seed_rmse"] = float(np.sqrt(np.mean((seeds-y)**2,axis=1)).mean())
        dm[name] = comparison
    order = sorted((m for m in dm if dm[m]["p"] is not None),key=lambda m:dm[m]["p"])
    prev = 0.
    for i,name in enumerate(order):
        prev = max(prev,min(1.,dm[name]["p"]*(len(order)-i)))
        dm[name]["p_holm"] = prev
    scale = train.x.std(axis=0)
    result = {}
    for stress,series in (("oscillation",pulse(data,start,scale)),("bias",apply_bias(data,start,scale))):
        p = model.predict(series)
        targets = data.truth[pulse_stop]
        endo = clean["endo"][pulse_stop]
        panel = {"DualARX":pressure_metrics(p["prediction"][pulse_stop],clean["prediction"][pulse_stop],
                   targets,endo,True,p["trust"][pulse_stop])}
        for name,shocks in baseline(np.c_[series.lab,series.x],pulse_rows).items():
            panel[name] = pressure_metrics(shocks.mean(axis=0),candidates[name][:,:512].mean(axis=0),targets,endo,False)
        result[stress] = panel
    return dict(stress=result,clean_dm=dm,baseline_receipts=receipts)


def score(task,delay,source,out,workbook,archive,paired,baseline_dir):
    f = json.loads((out/"FROZEN.json").read_text())
    if (f["task"],f["delay"],f["data"])!=(task,delay,receipt(source,task,("train","development"))):
        raise ValueError("Frozen inputs changed")
    if any(digest(Path(__file__).with_name(n))!=v for n,v in f["sources"].items()):
        raise ValueError("Frozen source changed")
    p = out/"MODEL.pkl"
    if (out/"RESULTS.json").exists() or digest(p)!=f["model"]:
        raise ValueError("Invalid state for a tail replay")
    model = pickle.loads(p.read_bytes())
    train,dev = load_prefix(task,source,delay)
    data = full_h2s(train,dev,workbook) if task=="sru" else append(append(train,dev),read_tep(source,"tail",delay))
    start = len(train)+len(dev)
    y = data.truth[start:]
    batch = model.predict(data)
    live = model.stream(data.x.shape[1])
    stream = np.array([live.step(x,a) for x,a in zip(data.x,lag(data.lab,delay))])
    expected = np.column_stack([batch[k] for k in ("prediction","endo","trust")])
    np.testing.assert_allclose(stream,expected,atol=2e-9,rtol=0)
    err = float(np.max(abs(stream-expected)))
    value = rmse(y,batch["prediction"][start:])
    checks = dict(stream=err<PARITY)
    result = dict(frozen=digest(out/"FROZEN.json"),n=len(y),tail_rmse=value,
         tail_mae=float(np.mean(abs(y-batch["prediction"][start:]))),
         max_stream_abs=err,mean_rho=float(batch["trust"][start:].mean()),
         p10_rho=float(np.quantile(batch["trust"][start:],.1)),checks=checks,
         source=digest(workbook if task=="sru" else source/"tail.npz"))
    if task=="sru":
        evidence = pressure_and_dm(model,data,train,start,archive,paired,baseline_dir)
        panel = evidence["stress"]
        p = evidence["clean_dm"]["ElasticNet"].get("p_holm")
        checks.update(competitive_clean=bool(value<=SRU_TARGET or (value<=SRU_BAND and p is not None and p>=.05)),
              exo_on=model.exogenous_on,exact_oscillation=bool(model.exogenous_on and
                     panel["oscillation"]["DualARX"]["max_fallback_error"]==0.),
              stress_leads=all(all(s["DualARX"][k]<=z[k] for m,z in s.items() if m!="DualARX")
                               for s in panel.values() for k in ("rmse","mae")))
        result["pressure"] = evidence
    else:
        checks["beats_arx"] = value<TEP_ARX[delay]
    save_json(out/"RESULTS.json",result)
    print("SCORED",out.name,value,checks,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    for cmd in ("fit","score"):
        p = sub.add_parser(cmd)
        p.add_argument("task",choices=("sru","tep"))
        p.add_argument("delay",type=int)
        p.add_argument("source",type=Path)
        p.add_argument("out",type=Path)
        if cmd=="fit":
            p.add_argument("authority",type=Path)
        else:
            for name in ("workbook","archive","paired","baselines"):
                p.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    if args.action=="fit":
        fit(args.task,args.delay,args.source,args.out,args.authority)
    else:
        if args.task=="sru" and not all((args.workbook,args.archive,args.paired,args.baselines)):
            parser.error("H2S score needs baseline paths and workbook")
        score(args.task,args.delay,args.source,args.out,args.workbook,args.archive,args.paired,args.baselines)


if __name__=="__main__":
    main()

