"""Causal truncated levels and complete measurement-span differences."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from laguerre_anchored import Endo, Profiled, ResidualHead, Standardized, X_BANKS
from laguerre_ecm import Dynamics, ExogenousMap
from laguerre_process import fixed_projection
from laguerre_screen import lag,train_rows


ENDO_NEAR = (4, 8, 16)
ENDO_ALPHA = (.0001, .0005, .0015, .003, .008)
ENDO_MIX = .2


@dataclass(frozen=True)
class MatureState:
    near: int
    span: int
    anchor_delay: int
    center: float

    def batch(self, anchor):
        increment = np.zeros_like(anchor)
        first = self.anchor_delay + self.span
        if first < len(anchor):
            increment[first:] = anchor[first:] - anchor[self.anchor_delay:-self.span]
        level = np.zeros_like(anchor)
        level[self.anchor_delay:] = anchor[self.anchor_delay:] - self.center
        return np.column_stack((Dynamics(self.near, ()).batch(increment), level))

    def stream(self, shape):
        return MatureStream(self, shape)


class MatureStream:
    def __init__(self, design, shape):
        self.design = design
        self.history = np.zeros(design.span)
        self.tick = 0
        self.increments = Dynamics(design.near, ()).stream(shape)

    def step(self, anchor):
        tick = self.tick
        pos = tick % self.design.span
        level = anchor - self.design.center if tick >= self.design.anchor_delay else 0.
        increment = anchor - self.history[pos] if tick >= self.design.anchor_delay + self.design.span else 0.
        self.history[pos] = anchor
        self.tick += 1
        return np.r_[self.increments.step(increment), level]


def mature_problem(data, rows, near):
    anchor = data.anchor()
    dynamics = MatureState(near, data.delay, data.anchor_delay, float(anchor[rows].mean()))
    state = dynamics.batch(anchor)
    problem = Standardized(state[rows], data.truth[rows] - anchor[rows])
    return dynamics, anchor, state, problem


def prepare_safe(data, rows, spec):
    dynamics, anchor, state, problem = mature_problem(data, rows, spec["near"])
    if spec["admitted"]:
        intercept, beta, _ = problem.whole(spec["alpha"], ENDO_MIX)
    else:
        intercept, beta = 0., np.zeros(state.shape[1])
    safe = Endo(dynamics, intercept, beta)
    return safe, state, safe.predict(anchor, state)


@lru_cache(maxsize=None)
def independent_taps(near, delay):
    """Give exact taps first priority, then retain independent IIR transfers."""
    if near < 1 or delay < 1:
        raise ValueError("Positive near lag and measurement delay required")
    h = Dynamics(near,X_BANKS).batch(np.r_[1.,np.zeros(4095+delay)])
    d = h-lag(h,delay)
    level = min(near,delay)
    mask = np.zeros((2,h.shape[1]),dtype=bool)
    base = np.empty((len(h),0))
    exact = [(0,k) for k in range(level)]+[(1,k) for k in range(near)]
    # Difference tails take precedence when two transfers are algebraically redundant.
    tail = [(j,k) for j in (1,0) for k in range(near,h.shape[1])]
    for view,t in (*exact,*tail):
        column = (h,d)[view][:,t]
        res = column-base@(base.T@column)
        res -= base@(base.T@res)
        norm = np.linalg.norm(res)
        if norm > 1e-10*np.linalg.norm(column):
            mask[view,t] = True
            base = np.column_stack((base,res/norm))
        elif (view,t) in exact:
            raise ValueError("Rank-deficient exact-lag transfer")
    mask.flags.writeable = False
    return mask


@dataclass(frozen=True)
class DualModel:
    safe: Endo
    mapping: ExogenousMap
    joint: ResidualHead
    delay: int
    exogenous_on: bool

    def predict(self, data):
        if data.delay!=self.delay:
            raise ValueError("Unexpected anchor span")
        a = data.anchor()
        state = self.safe.dynamics.batch(a)
        y0 = self.safe.predict(a,state)
        if not self.exogenous_on:
            return dict(prediction=y0,endo=y0,trust=np.zeros(len(a)))
        d = self.mapping.design(data.views())
        pred,rho = self.joint.mix(d.raw,d.trust,state,y0)
        return dict(prediction=pred,endo=y0,trust=rho)

    def stream(self, channels):
        return DualStream(self,channels)


class DualStream:
    def __init__(self, model, channels):
        self.model = model
        self.history = np.zeros((model.delay,channels))
        self.tick = 0
        self.y = model.safe.dynamics.stream(())
        self.x = model.mapping.stream() if model.exogenous_on else None

    def step(self, x, anchor):
        state = self.y.step(anchor)
        safe = self.model.safe.predict(anchor,state)
        if self.x is None:
            return safe,safe,0.
        if self.tick==0:
            self.history[:] = x
        k = self.tick%self.model.delay
        raw,trust = self.x.step(np.column_stack((x,x-self.history[k])).ravel())
        self.history[k] = x
        self.tick += 1
        y,rho = self.model.joint.mix(raw,trust,state,safe)
        return y,safe,rho


@dataclass
class Fold:
    mapping: ExogenousMap
    design: object
    rows: np.ndarray
    state: np.ndarray
    safe: Endo
    safe_hat: np.ndarray
    mask: np.ndarray
    means: np.ndarray
    target: np.ndarray
    proof: Profiled
    near: int
    delay: int


def prepare(data, fit_stop, near, endo, config):
    rows = train_rows(data.delay,fit_stop)
    y0, e, safe = prepare_safe(data, rows, endo)
    view = data.views()
    mapping = ExogenousMap.fit(view,rows,Dynamics(near,X_BANKS),config.padding,config.width)
    design = mapping.design(view)
    mask = np.tile(independent_taps(near,data.delay),(data.x.shape[1],1))
    x = design.raw[rows,:,:,0][:,mask]
    p = Profiled(Standardized(np.column_stack((e[rows],x)),(data.truth-safe)[rows]),e.shape[1])
    return Fold(mapping,design,rows,e,y0,safe,mask,design.raw[rows].mean(axis=0),data.truth,p,near,data.delay)


def nonlinear_delta(f, theta, ycoef, bias, curvature):
    active = np.flatnonzero(np.any(theta[...,0]!=0,axis=1))
    if not len(active):
        return bias,0
    r = f.rows
    raw = f.design.raw[r]
    components = np.einsum("ngtp,gtp->n",raw,theta,optimize=True)
    y = f.target[r]-f.safe_hat[r]-bias-f.state[r]@ycoef-components
    projected,transforms,widths = [],[],[]
    for group in active:
        p = fixed_projection(f.near,raw.shape[2])
        if group%2==0:
            p[min(f.near,f.delay):f.near] = 0.
            norm = np.linalg.norm(p,axis=0)
            p = p[:,norm>0]/norm[norm>0]
        curves = np.einsum("ntp,tk->nkp",raw[:,group,:,1:],p,optimize=True).reshape(len(r),-1)
        projected.append(curves)
        transforms.append(p)
        widths.append(curves.shape[1])
    model = Standardized(np.column_stack(projected),y)
    penalty = np.zeros_like(model.g)
    ends = np.r_[0,np.cumsum(widths)]
    for pos,group in enumerate(active):
        k = f.design.curvature[group,1:,1:]
        px = np.kron(np.eye(transforms[pos].shape[1]),k+.01*np.eye(k.shape[0]))
        block = slice(ends[pos],ends[pos+1])
        scale = model.scale[block]
        penalty[block,block] = px/np.outer(scale,scale)
    from scipy.linalg import solve
    intercept,coef = model.physical(solve(model.g+curvature*penalty,model.h,assume_a="pos"))
    for pos,group in enumerate(active):
        p = transforms[pos]
        theta[group,:,1:] = p@coef[ends[pos]:ends[pos+1]].reshape(p.shape[1],-1)
    return bias+intercept,len(coef)


def readout(f, alpha, mix, curve=None):
    intercept,b,kkt = f.proof.fit(alpha,mix)
    ycount = f.state.shape[1]
    ycoef = b[:ycount]
    firm = np.zeros(f.mask.shape)
    firm[f.mask] = b[ycount:]
    theta = np.zeros(f.design.raw.shape[1:])
    theta[...,0] = firm
    active = 0
    if curve is not None:
        intercept,active = nonlinear_delta(f,theta,ycoef,intercept,curve)
    j = ResidualHead.make(intercept,ycoef,theta,f.means)
    return j,dict(kkt=kkt,linear_x=int(np.count_nonzero(firm)),linear_y=int(np.count_nonzero(ycoef)),
                  nonlinear=active)


def forecast(f, joint, rows, opened=False):
    trust = f.design.trust[rows]
    if opened:
        trust = np.ones_like(trust)
    return joint.mix(f.design.raw[rows],trust,f.state[rows],f.safe_hat[rows])[0]

