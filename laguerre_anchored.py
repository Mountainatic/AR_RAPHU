"""Fold-standardized residual Laguerre ARX with selective elastic-net penalty."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.linalg import solve
from sklearn.linear_model import lars_path_gram

from laguerre_bj import CURVE, MIX, NEAR, TEP_ARX
from laguerre_control import load_prefix
from laguerre_ecm import Dynamics, ExogenousMap, ridge
from laguerre_joint import Endo, Joint, LaguerreARX, SOURCES as PRIOR_SOURCES, Y_BANKS, Y_NEAR
from laguerre_process import fixed_projection, kkt_error, polish
from laguerre_screen import CONFIGS, WARMUP, append, digest, lag, read_tep, receipt, rmse, save_json, train_rows
from laguerre_unified import full_h2s, pressure, summarize_gate


FLOOR = 1e-8
X_BANKS = ((.55,4),(.86,5))
ALPHA = (.0001,.0005,.0015,.003,.008,.018,.04,.1)
ENDOGENOUS_RIDGE = .003
FOLD_SPLIT = (.25,.355,.46,.565,.67,.722,.774)
TRAIN_H2S_LIMIT = .030
H2S_LIMIT = .03260
PARITY_LIMIT = 1e-15
SOURCES = ("laguerre_anchored.py", *PRIOR_SOURCES)


def elastic_coefficients(gram, rhs, alpha, mix):
    g = gram.copy()
    g.flat[::len(g)+1] += alpha*(1.-mix)
    l1 = alpha*mix
    if not l1:
        beta = solve(g,rhs,assume_a="pos")
    else:
        _,_,beta = lars_path_gram(rhs,g,n_samples=1,alpha_min=l1,method="lasso",
                                 max_iter=4*len(g),return_path=False)
    if kkt_error(g,rhs,l1,beta)>2e-7:
        beta = polish(g,rhs,l1,beta)
    error = kkt_error(g,rhs,l1,beta)
    if error>2e-7:
        raise RuntimeError(f"Elastic net did not satisfy KKT: {error:.4g}")
    return beta,error


class Standardized:
    def __init__(self, matrix, target, floor=FLOOR, target_floor=FLOOR):
        self.mean = matrix.mean(axis=0)
        self.scale = np.maximum(matrix.std(axis=0),floor)
        x = (matrix-self.mean)/self.scale
        self.ymean = float(target.mean())
        self.yscale = max(float(target.std()),target_floor)
        y = (target-self.ymean)/self.yscale
        self.g = x.T@x/len(y)
        self.h = x.T@y/len(y)

    def physical(self, beta):
        physical = beta*self.yscale/self.scale
        return float(self.ymean-self.mean@physical),physical

    def whole(self, alpha,mix):
        b,kkt = elastic_coefficients(self.g,self.h,alpha,mix)
        return *self.physical(b),kkt


class Profiled:
    """Eliminate the lightly ridge-penalized target state from the same joint objective."""

    def __init__(self, problem, endo_columns):
        self.standardized = problem
        s = endo_columns
        self.hyy = problem.g[:s,:s]+ENDOGENOUS_RIDGE*np.eye(s)
        self.gyx = problem.g[:s,s:]
        self.hy = problem.h[:s]
        self.y0 = solve(self.hyy,self.hy,assume_a="pos")
        self.response = solve(self.hyy,self.gyx,assume_a="pos")
        self.gx = problem.g[s:,s:]-self.gyx.T@self.response
        self.gx = (self.gx+self.gx.T)*.5
        self.hx = problem.h[s:]-self.gyx.T@self.y0

    def fit(self, alpha, mix):
        bx,error = elastic_coefficients(self.gx,self.hx,alpha,mix)
        by = self.y0-self.response@bx
        ky = float(np.max(abs(self.hyy@by+self.gyx@bx-self.hy)))
        return *self.standardized.physical(np.r_[by,bx]),max(error,ky)


@dataclass(frozen=True)
class ResidualHead(Joint):
    def mix(self, raw, trust, state, safe, enabled=True):
        if not enabled:
            return safe,np.zeros_like(safe)
        rho = np.clip(trust@self.weights,0.,1.)
        delta = rho*(self.location+state@self.beta_y)
        change = np.sum(trust*self.components(raw),axis=-1)
        return safe+delta+change,rho


def cuts(data):
    start = WARMUP+data.delay
    values = (start+(len(data)-start)*np.array(FOLD_SPLIT)).astype(int)
    edges = np.r_[values,np.linspace(values[-1],len(data),5,dtype=int)[1:]]
    result = tuple((int(a),np.arange(a,b)) for a,b in zip(edges[:-1],edges[1:]))
    return result[:4],result[4:6],result[6:]


def safe_fit(dynamics, anchor, state, truth, rows, alpha, admitted):
    if not admitted:
        return Endo(dynamics,0.,np.zeros(state.shape[1]))
    target = truth[rows]-anchor[rows]
    intercept,beta,_ = Standardized(state[rows],target).whole(alpha,0.)
    return Endo(dynamics,intercept,beta)


def safe_search(data, inner):
    grid = tuple(product(Y_NEAR,ALPHA))
    errors = np.zeros(len(grid))
    count = persist = 0
    for stop,test in inner:
        section = data.through(test[-1]+1)
        selected = train_rows(data.delay,stop)
        anchor = section.anchor()
        y = section.truth[test]
        persist += np.sum((y-anchor[test])**2)
        count += len(test)
        for ix,(near,alpha) in enumerate(grid):
            dynamics = Dynamics(near,Y_BANKS)
            state = dynamics.batch(anchor)
            safe = safe_fit(dynamics,anchor,state,section.truth,selected,alpha,True)
            errors[ix] += np.sum((y-safe.predict(anchor[test],state[test]))**2)
    i = int(np.argmin(errors))
    return dict(near=grid[i][0],alpha=grid[i][1],admitted=bool(errors[i]<persist),
                inner_rmse=float(np.sqrt(errors[i]/count)),
                persistence_inner_rmse=float(np.sqrt(persist/count)))


@dataclass
class Prepared:
    rows: np.ndarray
    state: np.ndarray
    safe: Endo
    safe_hat: np.ndarray
    mapping: ExogenousMap
    design: object
    fitter: Profiled
    means: np.ndarray
    truth: np.ndarray


def prepare(data, stop, near, safe_protocol, original):
    rows = train_rows(data.delay,stop)
    dy = Dynamics(safe_protocol["near"],Y_BANKS)
    anchor = data.anchor()
    state = dy.batch(anchor)
    m = safe_fit(dy,anchor,state,data.truth,rows,safe_protocol["alpha"],safe_protocol["admitted"])
    y0 = m.predict(anchor,state)
    exo = ExogenousMap.fit(data.x,rows,Dynamics(near,X_BANKS),original.padding,original.width)
    d = exo.design(data.x)
    x = d.raw[rows,:,:,0].reshape(len(rows),-1)
    reg = np.column_stack((state[rows],x))
    p = Profiled(Standardized(reg,(data.truth-y0)[rows]),state.shape[1])
    return Prepared(rows,state,m,y0,exo,d,p,d.raw[rows].mean(axis=0),data.truth)


def nonlinear_adjustment(prep, theta, bias, by, selected, alpha):
    rows = prep.rows
    raw = prep.design.raw[rows][:,selected]
    taps = raw.shape[2]
    basis = fixed_projection(prep.mapping.dynamics.near,taps)
    z = np.einsum("ngtp,tk->ngkp",raw[...,1:],basis,optimize=True).reshape(len(rows),-1)
    output = prep.truth[rows]-prep.safe_hat[rows]-bias-(prep.state[rows]@by)
    output -= np.einsum("ngtp,gtp->n",raw,theta[selected],optimize=True)
    p = Standardized(z,output)
    ncols = basis.shape[1]*(raw.shape[-1]-1)
    penalty = np.zeros_like(p.g)
    matrices = prep.design.curvature[selected,1:,1:]
    for i,k in enumerate(matrices):
        block = slice(i*ncols,(i+1)*ncols)
        s = p.scale[block]
        px = np.kron(np.eye(basis.shape[1]),k+.01*np.eye(k.shape[0]))
        penalty[block,block] = px/np.outer(s,s)
    bstd = solve(p.g+alpha*penalty,p.h,assume_a="pos")
    intercept,coef = p.physical(bstd)
    theta[selected,:,1:] = np.einsum("tk,gkp->gtp",basis,coef.reshape(int(selected.sum()),basis.shape[1],-1))
    return bias+intercept,int(len(coef))


def fit_joint(f, alpha, mix, curve=None):
    bias,b,kkt = f.fitter.fit(alpha,mix)
    s = f.state.shape[1]
    theta = np.zeros(f.design.raw.shape[1:])
    theta[...,0] = b[s:].reshape(theta.shape[:2])
    by = b[:s]
    selected = np.any(theta[...,0]!=0,axis=1)
    curved = 0
    if curve is not None and selected.any():
        bias,curved = nonlinear_adjustment(f,theta,bias,by,selected,curve)
    joint = ResidualHead.make(bias,by,theta,f.means)
    return joint,dict(kkt_residual=kkt,active_x=int(np.count_nonzero(theta[...,0])),
        active_y=int(np.count_nonzero(by)),active_nonlinear=curved,
        centered_training_max=float(np.max(abs(joint.components(f.design.raw[f.rows]).mean(axis=0)))))


def predict(f, model, rows):
    return model.mix(f.design.raw[rows],f.design.trust[rows],f.state[rows],f.safe_hat[rows])[0]


def linear_search(data, safe, original, inner):
    grid = tuple(product(ALPHA,MIX))
    def run(near):
        loss = np.zeros((len(grid),len(inner)))
        size = []
        for j,(stop,test) in enumerate(inner):
            section = data.through(test[-1]+1)
            f = prepare(section,stop,near,safe,original)
            size.append(len(test))
            for i,(a,m) in enumerate(grid):
                model,_ = fit_joint(f,a,m)
                e = f.truth[test]-predict(f,model,test)
                loss[i,j] = e@e
        records = [dict(r=near,alpha=a,mix=m,inner_rmse=float(np.sqrt(v.sum()/sum(size))),
                     fold_rmse=np.sqrt(v/size).tolist()) for (a,m),v in zip(grid,loss,strict=True)]
        print("INNER",near,min(records,key=lambda x:x["inner_rmse"]),flush=True)
        return records
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [x for group in pool.map(run,NEAR) for x in group]


def loss_by_curve(data,safe,original,spec,folds,grid):
    result = []
    for stop,t in folds:
        part = data.through(t[-1]+1)
        f = prepare(part,stop,spec["r"],safe,original)
        risk = []
        for penalty in grid:
            m,_ = fit_joint(f,spec["alpha"],spec["mix"],penalty)
            risk.append((part.truth[t]-predict(f,m,t))**2)
        result.append(np.stack(risk))
    return result


def fit(task,delay,source,out,authority):
    table = receipt(source,task,("train","development"))
    previous = json.loads((authority/"FROZEN.json").read_text())
    if previous["data"]!=table or (previous["task"],previous["delay"])!=(task,delay):
        raise ValueError("Different data in the envelope authority")
    original = CONFIGS[task][previous["selected"]]
    if json.loads(json.dumps(asdict(original)))!=previous["configs"][previous["selected"]]:
        raise ValueError("Original envelope changed")
    train,dev = load_prefix(task,source,delay)
    inner,confirm,outer = cuts(train)
    safe = safe_search(train,inner)
    candidates = linear_search(train,safe,original,inner)
    choice = min(candidates,key=lambda p:p["inner_rmse"])
    nonlin_loss = loss_by_curve(train,safe,original,choice,inner,CURVE)
    risk = np.sqrt(np.concatenate(nonlin_loss,axis=1).mean(axis=1))
    best = CURVE[int(np.argmin(risk))]
    contrast = loss_by_curve(train,safe,original,choice,confirm,(None,best))
    non = summarize_gate([a[0]-a[1] for a in contrast],delay)
    curve = best if non["admitted"] else None
    gain = []
    for stop,t in outer:
        section = train.through(t[-1]+1)
        f = prepare(section,stop,choice["r"],safe,original)
        m,_ = fit_joint(f,choice["alpha"],choice["mix"],curve)
        gain.append((f.truth[t]-f.safe_hat[t])**2-(f.truth[t]-predict(f,m,t))**2)
    exo = summarize_gate(gain,delay)
    f = prepare(train,len(train),choice["r"],safe,original)
    joint,audit = fit_joint(f,choice["alpha"],choice["mix"],curve)
    model = LaguerreARX(f.safe,f.mapping,joint,exo["admitted"])
    in_domain = joint.mix(f.design.raw[f.rows],np.ones_like(f.design.trust[f.rows]),
                         f.state[f.rows],f.safe_hat[f.rows])[0]
    audit["train_ungated_rmse"] = rmse(f.truth[f.rows],in_domain)
    audit["train_n"] = len(f.rows)
    combined = append(train,dev)
    dev_rmse = rmse(dev.truth,model.predict(combined)["prediction"][len(train):])
    out.mkdir(parents=True,exist_ok=False)
    file = out/"MODEL.pkl"
    file.write_bytes(pickle.dumps(model,protocol=5))
    def annotations(seq):
        return [[a,int(t[-1])+1] for a,t in seq]
    save_json(out/"FROZEN.json",dict(task=task,delay=delay,data=table,model=digest(file),
        source={n:digest(Path(__file__).with_name(n)) for n in SOURCES},
        authority=digest(authority/"FROZEN.json"),fit=audit,linear=choice,linear_grid=candidates,
        safe=safe,curve=curve,nonlinear_gate=non,exogenous_gate=exo,development_rmse=dev_rmse,
        protocol=dict(floor=FLOOR,x_banks=X_BANKS,y_banks=Y_BANKS,alpha=ALPHA,l1_ratio=MIX,near=NEAR,
          endo_ridge=ENDOGENOUS_RIDGE,inner=annotations(inner),nonlinear_confirmation=annotations(confirm),
          outer=annotations(outer),h2s_clean_limit=H2S_LIMIT,h2s_train_limit=TRAIN_H2S_LIMIT,
          parity_limit=PARITY_LIMIT,tep_arx=TEP_ARX)))
    print("FIT",out.name,choice,audit,"T",exo["t_stat"],"blocks",exo["positive_blocks"],"development",dev_rmse,flush=True)


def score(task,delay,source,out,workbook=None,archive=None,paired=None,baselines=None):
    f = json.loads((out/"FROZEN.json").read_text())
    if (f["task"],f["delay"],f["data"])!=(task,delay,receipt(source,task,("train","development"))):
        raise ValueError("Frozen inputs changed")
    if any(digest(Path(__file__).with_name(n))!=h for n,h in f["source"].items()):
        raise ValueError("Frozen source changed")
    path = out/"MODEL.pkl"
    if digest(path)!=f["model"] or (out/"RESULTS.json").exists():
        raise ValueError("Invalid scoring state")
    m = pickle.loads(path.read_bytes())
    train,dev = load_prefix(task,source,delay)
    if task=="sru":
        data = full_h2s(train,dev,workbook)
    else:
        data = append(append(train,dev),read_tep(source,"tail",delay))
    start = len(train)+len(dev)
    y = data.truth[start:]
    pred = m.predict(data)
    live = m.stream()
    replay = np.array([live.step(x,a) for x,a in zip(data.x,lag(data.lab,delay))])
    batch = np.column_stack([pred[k] for k in ("prediction","endo","trust")])
    np.testing.assert_allclose(replay,batch,atol=2e-9,rtol=0)
    discrepancy = float(np.max(abs(replay-batch)))
    trust = pred["trust"][start:]
    risk = rmse(y,pred["prediction"][start:])
    checks = dict(stream=discrepancy<=PARITY_LIMIT)
    result = dict(frozen=digest(out/"FROZEN.json"),tail_n=len(y),rmse=risk,
        tail_source=digest(workbook if task=="sru" else source/"tail.npz"),
        fallback_rmse=rmse(y,pred["endo"][start:]),tail_trust_mean=float(trust.mean()),
        tail_trust_p10=float(np.quantile(trust,.1)),max_batch_stream=discrepancy,checks=checks)
    if task=="sru":
        a = data.anchor()
        e = m.safe.dynamics.batch(a)
        y0 = m.safe.predict(a,e)
        d = m.mapping.design(data.x)
        full,_ = m.joint.mix(d.raw,np.ones_like(d.trust),e,y0)
        results = pressure(m,data,train,start,archive,paired,baselines)
        for panel in results["stress"].values():
            panel["AnchoredARX"] = panel.pop("Unified")
        osc = results["stress"]["oscillation"]["AnchoredARX"]
        checks.update(training=f["fit"]["train_ungated_rmse"]<=TRAIN_H2S_LIMIT,
            alpha_interior=ALPHA[0]<f["linear"]["alpha"]<ALPHA[-1],
            exogenous_admitted=m.exogenous_on and f["exogenous_gate"]["positive_blocks"]>=3,
            clean=risk<=H2S_LIMIT,exact_fallback=bool(m.exogenous_on and osc["max_fallback_error"]==0.),
            stress_leads=all(all(panel["AnchoredARX"][k]<=z[k] for n,z in panel.items() if n!="AnchoredARX")
                             for panel in results["stress"].values() for k in ("rmse","mae")))
        result.update(pressure=results,ungated_tail_rmse=rmse(y,full[start:]))
    else:
        checks["beats_arx"] = risk<TEP_ARX[delay]
    save_json(out/"RESULTS.json",result)
    print("SCORE",out.name,risk,checks,flush=True)


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
            p.add_argument("authority",type=Path)
        else:
            for name in ("workbook","archive","paired","baselines"):
                p.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    if args.action=="fit":
        fit(args.task,args.delay,args.source,args.out,args.authority)
    else:
        if args.task=="sru" and not all((args.workbook,args.archive,args.paired,args.baselines)):
            parser.error("H2S needs its workbook and baseline paths")
        score(args.task,args.delay,args.source,args.out,args.workbook,args.archive,args.paired,args.baselines)


if __name__=="__main__":
    main()

