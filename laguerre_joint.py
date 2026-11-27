"""Joint support-gated Laguerre ARX with a separate, frozen target-only fallback."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import pickle

import numpy as np

from laguerre_bj import CURVE, MIX, NEAR, TEP_ARX, blocks
from laguerre_control import load_prefix
from laguerre_ecm import Design, Dynamics, ExogenousMap, box_jenkins_deployment
from laguerre_process import CurvatureProblem, LinearProblem, Process
from laguerre_screen import CONFIGS, append, digest, lag, read_tep, receipt, rmse, save_json, train_rows
from laguerre_unified import full_h2s, pressure, summarize_gate


X_BANKS = ((.55,4),(.86,5),(.95,1),(.98,1))
Y_BANKS = ((.86,1),(.95,1),(.98,1))
Y_NEAR = (2,4)
ALPHA = (.0005,.003,.018,.06,.18)
H2S_LIMIT = .03260
PARITY_LIMIT = 1e-15
SOURCES = ("laguerre_joint.py","laguerre_bj.py","laguerre_process.py","laguerre_ecm.py",
           "laguerre_screen.py","laguerre_control.py","laguerre_unified.py","laguerre_aligned.py",
           "laguerre_sparse_views.py","laguerre_pressure.py")


@dataclass(frozen=True)
class Endo:
    dynamics: Dynamics
    location: float
    beta: np.ndarray

    def predict(self, anchor, state):
        return anchor+self.location+state@self.beta


@dataclass(frozen=True)
class Joint:
    location: float
    beta_y: np.ndarray
    theta: np.ndarray
    center: np.ndarray
    weights: np.ndarray

    @classmethod
    def make(cls, raw_intercept, beta_y, theta, means):
        center = np.einsum("gtp,gtp->g",theta,means)
        size = np.linalg.norm(theta.reshape(len(theta),-1),axis=1)
        w = size/size.sum() if size.any() else size
        return cls(float(raw_intercept+center.sum()),beta_y,theta,center,w)

    def components(self, raw):
        return np.einsum("...gtp,gtp->...g",raw,self.theta,optimize=True)-self.center

    def mix(self, raw, trust, state, safe, enabled=True):
        if not enabled:
            return safe,np.zeros_like(safe)
        rho = np.clip(trust@self.weights,0.,1.)
        terms = self.components(raw)
        location = self.location+state@self.beta_y
        out = box_jenkins_deployment(safe,location,np.sum(trust*terms,axis=-1),rho)
        return out,rho


@dataclass(frozen=True)
class LaguerreARX:
    safe: Endo
    mapping: ExogenousMap
    joint: Joint
    exogenous_on: bool

    def predict(self, data):
        anchor = data.anchor()
        e = self.safe.dynamics.batch(anchor)
        y0 = self.safe.predict(anchor,e)
        if self.exogenous_on:
            x = self.mapping.design(data.x)
            y,rho = self.joint.mix(x.raw,x.trust,e,y0)
        else:
            y,rho = y0,np.zeros(len(y0))
        return dict(prediction=y,endo=y0,trust=rho)

    def stream(self):
        return JointStream(self)


class JointStream:
    def __init__(self, model):
        self.model = model
        self.maturity = model.safe.dynamics.stream(())
        self.exo = model.mapping.stream() if model.exogenous_on else None

    def step(self, x, anchor):
        z = self.maturity.step(anchor)
        y0 = self.model.safe.predict(anchor,z)
        if self.exo is None:
            return y0,y0,0.
        raw,trust = self.exo.step(x)
        y,rho = self.model.joint.mix(raw,trust,z,y0)
        return y,y0,rho


def fit_endo(data, rows, dynamics, state, alpha, enabled=True):
    if not enabled:
        return Endo(dynamics,0.,np.zeros(state.shape[1]))
    y = data.truth-data.anchor()
    p = LinearProblem(state[:,None,:,None],y,rows)
    model,_ = p.fit(alpha,0.)
    return Endo(dynamics,model.location,model.theta.ravel())


def endo_search(data, inner):
    grid = tuple(product(Y_NEAR,ALPHA))
    value = np.zeros(len(grid))
    error_zero = n = 0
    for stop,test in inner:
        part = data.through(test[-1]+1)
        rows = train_rows(data.delay,stop)
        truth = part.truth[test]
        anchor = part.anchor()
        error_zero += np.sum((truth-anchor[test])**2)
        n += len(test)
        for i,(near,alpha) in enumerate(grid):
            states = Dynamics(near,Y_BANKS)
            e = states.batch(anchor)
            m = fit_endo(part,rows,states,e,alpha)
            value[i] += np.sum((truth-m.predict(anchor[test],e[test]))**2)
    best = int(np.argmin(value))
    near,alpha = grid[best]
    return dict(near=near,alpha=alpha,admitted=bool(value[best]<error_zero),
                inner_rmse=float(np.sqrt(value[best]/n)),persistence_inner_rmse=float(np.sqrt(error_zero/n)),
                candidates=[dict(near=r,alpha=a,rmse=float(np.sqrt(v/n))) for (r,a),v in zip(grid,value,strict=True)])


@dataclass
class Fold:
    design: Design
    mapping: ExogenousMap
    state: np.ndarray
    safe: Endo
    safe_prediction: np.ndarray
    rows: np.ndarray
    problem: LinearProblem
    means: np.ndarray
    truth: np.ndarray


def prepare(data, stop, near, endo, original):
    rows = train_rows(data.delay,stop)
    dynamic_y = Dynamics(endo["near"],Y_BANKS)
    anchor = data.anchor()
    state = dynamic_y.batch(anchor)
    y0 = fit_endo(data,rows,dynamic_y,state,endo["alpha"],endo["admitted"])
    mapping = ExogenousMap.fit(data.x,rows,Dynamics(near,X_BANKS),original.padding,original.width)
    design = mapping.design(data.x)
    linear = design.raw[...,0].reshape(len(data),-1)
    regressors = np.column_stack((state,linear))
    problem = LinearProblem(regressors[:,None,:,None],data.truth,rows)
    return Fold(design,mapping,state,y0,y0.predict(anchor,state),rows,problem,
                design.raw[rows].mean(axis=0),data.truth)


def evaluate_joint(f, alpha, mix, curve=None):
    linear,audit = f.problem.fit(alpha,mix)
    b = linear.theta.ravel()
    nendo = f.state.shape[1]
    beta_y = b[:nendo]
    theta = np.zeros(f.design.raw.shape[1:])
    theta[...,0] = b[nendo:].reshape(theta.shape[:2])
    bias = linear.location
    active = np.any(theta[...,0]!=0,axis=1)
    nonlinear = 0
    if curve is not None and active.any():
        design = Design(f.design.raw[:,active],f.design.trust[:,active],f.design.curvature[active])
        process = Process.build(bias,theta[active])
        target = f.truth-f.state@beta_y
        p = CurvatureProblem(design,target,f.rows,process,f.mapping.dynamics.near)
        fitted = p.fit(curve)
        theta[active] = fitted.theta
        bias = fitted.location
        nonlinear = int(np.prod(p.dims))
    model = Joint.make(bias,beta_y,theta,f.means)
    centered = model.components(f.design.raw[f.rows]).mean(axis=0)
    audit.update(active_linear_x=int(np.count_nonzero(theta[...,0])),active_linear_y=int(np.count_nonzero(beta_y)),
                 active_nonlinear=nonlinear,active_channels=int(active.sum()),
                 max_abs_training_channel_mean=float(np.max(abs(centered))))
    return model,audit


def joint_search(data, original, endo, inner):
    grid = tuple(product(ALPHA,MIX))
    def score_near(near):
        loss = np.zeros(len(grid))
        n = 0
        for stop,test in inner:
            part = data.through(test[-1]+1)
            f = prepare(part,stop,near,endo,original)
            for i,(alpha,mix) in enumerate(grid):
                m,_ = evaluate_joint(f,alpha,mix)
                y,_ = m.mix(f.design.raw[test],f.design.trust[test],f.state[test],f.safe_prediction[test])
                loss[i] += np.sum((part.truth[test]-y)**2)
            n += len(test)
        detail = [dict(r=near,alpha=a,mix=m,inner_rmse=float(np.sqrt(v/n)))
                  for (a,m),v in zip(grid,loss,strict=True)]
        print("INNER",near,min(detail,key=lambda x:x["inner_rmse"]),flush=True)
        return detail
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [x for g in pool.map(score_near,NEAR) for x in g]


def curve_losses(data, original, endo, choice, splits, options):
    losses = []
    for stop,t in splits:
        part = data.through(t[-1]+1)
        p = prepare(part,stop,choice["r"],endo,original)
        x,trust,state,safe = p.design.raw[t],p.design.trust[t],p.state[t],p.safe_prediction[t]
        fold = []
        for c in options:
            j,_ = evaluate_joint(p,choice["alpha"],choice["mix"],c)
            y,_ = j.mix(x,trust,state,safe)
            fold.append((part.truth[t]-y)**2)
        losses.append(np.stack(fold))
    return losses


def fit(task,delay,source,out,support):
    inputs = receipt(source,task,("train","development"))
    frozen = json.loads((support/"FROZEN.json").read_text())
    if frozen["data"]!=inputs or (frozen["task"],frozen["delay"])!=(task,delay):
        raise ValueError("Support authority and current inputs differ")
    original = CONFIGS[task][frozen["selected"]]
    if json.loads(json.dumps(asdict(original)))!=frozen["configs"][frozen["selected"]]:
        raise ValueError("Original envelope parameters changed")
    data,dev = load_prefix(task,source,delay)
    inner,confirm,outer = blocks(data)
    endo = endo_search(data,inner)
    results = joint_search(data,original,endo,inner)
    spec = min(results,key=lambda x:x["inner_rmse"])
    errors = curve_losses(data,original,endo,spec,inner,CURVE)
    risk = np.sqrt(np.concatenate(errors,axis=1).mean(axis=1))
    best = CURVE[int(np.argmin(risk))]
    confirm_errors = curve_losses(data,original,endo,spec,confirm,(None,best))
    curve_gate = summarize_gate([err[0]-err[1] for err in confirm_errors],delay)
    curve = best if curve_gate["admitted"] else None
    gains = []
    for stop,t in outer:
        part = data.through(t[-1]+1)
        p = prepare(part,stop,spec["r"],endo,original)
        j,_ = evaluate_joint(p,spec["alpha"],spec["mix"],curve)
        y0 = p.safe_prediction[t]
        y,_ = j.mix(p.design.raw[t],p.design.trust[t],p.state[t],y0)
        truth = part.truth[t]
        gains.append((truth-y0)**2-(truth-y)**2)
    gate = summarize_gate(gains,delay)
    f = prepare(data,len(data),spec["r"],endo,original)
    j,audit = evaluate_joint(f,spec["alpha"],spec["mix"],curve)
    model = LaguerreARX(f.safe,f.mapping,j,gate["admitted"])
    prefix = append(data,dev)
    risk_dev = rmse(dev.truth,model.predict(prefix)["prediction"][len(data):])
    out.mkdir(parents=True,exist_ok=False)
    obj = out/"MODEL.pkl"
    obj.write_bytes(pickle.dumps(model,protocol=5))
    def annotate(folds):
        return [[a,int(t[-1])+1] for a,t in folds]
    save_json(out/"FROZEN.json",dict(task=task,delay=delay,data=inputs,model=digest(obj),
      support_authority=digest(support/"FROZEN.json"),source={n:digest(Path(__file__).with_name(n)) for n in SOURCES},
      protocol=dict(x_banks=X_BANKS,y_banks=Y_BANKS,near_x=NEAR,near_y=Y_NEAR,alpha=ALPHA,mix=MIX,curvature=CURVE,
        inner=annotate(inner),nonlinear_confirmation=annotate(confirm),outer=annotate(outer),
        h2s_limit=H2S_LIMIT,arx=TEP_ARX,max_stream_diff=PARITY_LIMIT),
      endo=endo,linear_grid=results,linear=spec,curve_grid=[dict(curvature=k,inner_rmse=float(v))
         for k,v in zip(CURVE,risk,strict=True)],curve_gate=curve_gate,curvature=curve,
      exogenous_gate=gate,development_rmse=risk_dev,fit=audit))
    print("FROZEN",out.name,spec,"g_exo",gate["t_stat"],"curve",curve,"development",risk_dev,audit,flush=True)


def score(task,delay,source,out,workbook,archive,paired,baselines):
    f = json.loads((out/"FROZEN.json").read_text())
    if (f["task"],f["delay"],f["data"])!=(task,delay,receipt(source,task,("train","development"))):
        raise ValueError("Frozen score inputs changed")
    if any(digest(Path(__file__).with_name(n))!=h for n,h in f["source"].items()):
        raise ValueError("Frozen code changed")
    obj = out/"MODEL.pkl"
    output = out/"RESULTS.json"
    if digest(obj)!=f["model"] or output.exists():
        raise ValueError("Invalid score state")
    model = pickle.loads(obj.read_bytes())
    train,dev = load_prefix(task,source,delay)
    data = full_h2s(train,dev,workbook) if task=="sru" else append(append(train,dev),read_tep(source,"tail",delay))
    start = len(train)+len(dev)
    win = slice(start,len(data))
    p = model.predict(data)
    s = model.stream()
    online = np.array([s.step(x,a) for x,a in zip(data.x,lag(data.lab,delay))])
    batch = np.column_stack([p[k] for k in ("prediction","endo","trust")])
    np.testing.assert_allclose(online,batch,atol=2e-9,rtol=0)
    err = float(np.max(abs(online-batch)))
    y = data.truth[win]
    risk = rmse(y,p["prediction"][win])
    checks = dict(stream=err<=PARITY_LIMIT)
    result = dict(frozen=digest(out/"FROZEN.json"),task=task,delay=delay,tail_n=len(y),tail_rmse=risk,
       endo_tail_rmse=rmse(y,p["endo"][win]),mean_rho=float(p["trust"][win].mean()),
       p10_rho=float(np.quantile(p["trust"][win],.1)),stream_error=err,
       tail_sha=digest(workbook if task=="sru" else source/"tail.npz"),acceptance=checks)
    if task=="sru":
        suite = pressure(model,data,train,start,archive,paired,baselines)
        for scenario,rows in suite["stress"].items():
            rows["LaguerreARX"] = rows.pop("Unified")
        oscillation = suite["stress"]["oscillation"]["LaguerreARX"]
        checks.update(competitive_clean=risk<=H2S_LIMIT and risk<.0338567489400913,
                      exact_oscillation_fallback=oscillation["max_fallback_error"]==0.,
                      stress_leads_both_metrics=all(
                         all(panel["LaguerreARX"][k]<=m[k] for arm,m in panel.items() if arm!="LaguerreARX")
                         for panel in suite["stress"].values() for k in ("rmse","mae")))
        result["pressure"] = suite
    else:
        checks["beats_arx"] = risk<TEP_ARX[delay]
    save_json(output,result)
    print("SCORED",out.name,risk,checks,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    for action in ("fit","score"):
        q = sub.add_parser(action)
        q.add_argument("task",choices=("sru","tep"))
        q.add_argument("delay",type=int)
        q.add_argument("source",type=Path)
        q.add_argument("out",type=Path)
        if action=="fit":
            q.add_argument("support",type=Path)
        else:
            for x in ("workbook","archive","paired","baselines"):
                q.add_argument("--"+x,type=Path)
    a = p.parse_args()
    if a.action=="fit":
        fit(a.task,a.delay,a.source,a.out,a.support)
    else:
        if a.task=="sru" and not all((a.workbook,a.archive,a.paired,a.baselines)):
            p.error("SRU pressure replay requires the H2S workbook and baseline archives")
        score(a.task,a.delay,a.source,a.out,a.workbook,a.archive,a.paired,a.baselines)


if __name__=="__main__":
    main()

