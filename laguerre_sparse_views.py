"""Lag-sparse joint readout with short-term differences and slow levels."""

from dataclasses import dataclass

import numpy as np

from laguerre_ecm import Dynamics, Joint, _augment, ridge


@dataclass(frozen=True)
class Readout:
    near: int
    canonical: bool
    adjustment: bool
    l1: float
    curve: float = 0.


def basis_geometry(arm, original):
    if arm.canonical:
        return Dynamics(arm.near, (*original.banks, (.97, 2)))
    return Dynamics(arm.near, original.banks)


def blocks(dynamics, canonical, views, nonlinear):
    taps = dynamics.near+sum(order for _,order in dynamics.banks)
    if canonical:
        for g in range(views):
            yield g, np.r_[0,taps-2,taps-1] if g % 2 == 0 else np.arange(taps-2), None
    else:
        for g in range(views):
            yield g,np.arange(taps),None
    if nonlinear:
        r = dynamics.near
        for g in range(views):
            intervals = [(0,1)] if canonical and g % 2 == 0 else (
                [(0,min(4,r)),(4,r),(r,taps-(2 if canonical else 0))])
            vec = []
            for start,stop in intervals:
                if start < stop:
                    b = np.zeros(taps)
                    b[start:stop] = 1/np.sqrt(stop-start)
                    vec.append(b)
            yield g,np.arange(taps),np.array(vec).T


def group_fit(x, y, penalty, spans, families, fraction, lag_l1):
    """Sparse-group proximal fit, with lasso applied to linear lags only."""
    spans = np.asarray(spans)
    gram = x.T@x/len(y)+penalty
    score = x.T@y/len(y)
    stops = np.cumsum(spans)
    starts = np.r_[0,stops[:-1]]
    raw = score[:stops[-1]]
    gnorm = np.sqrt(np.add.reduceat(raw**2,starts))
    limits = np.zeros(len(spans))
    for category, ratio in enumerate((fraction,.08)):
        sel = families == category
        if sel.any():
            limits[sel] = ratio*max(gnorm[sel])
    nlinear = int(np.sum(spans[families == 0]))
    l1 = float(lag_l1*np.max(abs(score[:nlinear])))
    axis = np.random.default_rng(412).standard_normal(len(score))
    for _ in range(26):
        axis = gram@axis
        axis /= max(np.linalg.norm(axis),1e-30)
    step = .92/max(float(axis@(gram@axis)),1e-8)

    def prox(z):
        linear = z[:nlinear]
        linear[:] = np.sign(linear)*np.maximum(abs(linear)-l1*step,0.)
        groups = z[:stops[-1]]
        norm = np.sqrt(np.add.reduceat(groups**2,starts))
        factors = np.maximum(0.,1.-step*limits/np.maximum(norm,1e-300))
        groups *= np.repeat(factors,spans)
        return z

    previous = value = np.zeros(len(score))
    speed = 1.
    scale = max(np.linalg.norm(score,ord=np.inf),1e-12)
    for i in range(1,7001):
        gradient = gram@value-score
        candidate = prox(value-step*gradient)
        if i == 1:
            delta = candidate-value
            while delta@(gram@delta) > (delta@delta)/step+1e-16:
                step *= .7
                candidate = prox(value-step*gradient)
                delta = candidate-value
        if i % 25 == 0:
            error = candidate-prox(candidate-step*(gram@candidate-score))
            kkt = float(np.linalg.norm(error,ord=np.inf)/(scale*step))
            if kkt <= 3e-5:
                return candidate[:stops[-1]],dict(kkt=kkt,iterations=i,lag_l1=l1)
        next_speed = (1+np.sqrt(1+4*speed*speed))/2
        if (value-candidate)@(candidate-previous)>0:
            next_speed = 1.
        value = candidate+(speed-1)/next_speed*(candidate-previous) if next_speed>1 else candidate
        previous,speed = candidate,next_speed
    raise RuntimeError(f"Failed sparse group convergence: {kkt:.4g}")


def fit_readout(design, dynamic, endo, delta, rows, arm, original):
    nviews,taps,width = design.raw.shape[1:]
    specs = list(blocks(dynamic,arm.canonical,nviews,bool(arm.curve)))
    raw = design.raw[rows]
    trust = design.trust[rows]
    matrices, scales, curvatures = [], [], []
    for g,indices,proj in specs:
        if proj is None:
            matrix = raw[:,g,indices,0]
        else:
            matrix = np.einsum("ntp,tk->nkp",raw[:,g,:,1:],proj).reshape(len(rows),-1)
        matrix *= trust[:,g,None]
        scale = np.maximum(np.sqrt(np.mean(matrix*matrix,axis=0)),.08)
        matrices.append(matrix/scale)
        scales.append(scale)
        if proj is not None:
            curve = np.kron(np.eye(proj.shape[1]),design.curvature[g,1:,1:]+.01*np.eye(width-1))
            curvatures.append(arm.curve*curve/np.outer(scale,scale))
    spans = [len(s) for s in scales]
    families = np.array([int(proj is not None) for _,_,proj in specs])
    stops = np.r_[0,np.cumsum(spans)]
    nfeatures = stops[-1]
    x = np.column_stack(matrices)
    y = delta[rows]
    sy = endo[rows] if arm.adjustment else np.empty((len(rows),0))
    e_scale = np.maximum(np.sqrt(np.mean(sy*sy,axis=0)),1e-7)
    sy = sy/e_scale
    pooled = trust.mean(axis=1)
    p = np.zeros((nfeatures,nfeatures))
    for position,curve in zip(np.flatnonzero(families),curvatures,strict=True):
        sl = slice(stops[position],stops[position+1])
        p[sl,sl] = curve
    p.flat[::len(p)+1] += np.where(np.repeat(families,spans),0.,.008)
    if not arm.l1 and not arm.curve:
        # Dense linear diagnostic isolates adjustment and geometry from selection.
        physical = ridge(np.column_stack((x,pooled[:,None]*sy)), y,
                         _augment(p,len(e_scale),.008))[:nfeatures]
        selected = np.ones(nfeatures,dtype=bool)
        audit = dict(kkt=None,iterations=0,lag_l1=0.)
    else:
        physical,audit = group_fit(np.column_stack((x,pooled[:,None]*sy)),y,
                                  _augment(p,len(e_scale),.008),spans,families,
                                  original.sparsity,arm.l1)
        selected = physical != 0
        # Nonlinear groups re-enter as a block, linear lags retain exact zeros.
        for pos in np.flatnonzero(families):
            sl = slice(stops[pos],stops[pos+1])
            selected[sl] = selected[sl].any()
    weights = np.zeros(nviews)
    for pos,(g,_,_) in enumerate(specs):
        sl = slice(stops[pos],stops[pos+1])
        weights[g] += np.sum((physical[sl]/scales[pos])**2)
    weights = np.sqrt(weights)
    nlinear = int(stops[nviews])
    audit.update(linear_total=nlinear,linear_active=int(selected[:nlinear].sum()),
                 nonlinear_active=int(selected[nlinear:].sum()),adjustment=arm.adjustment)
    if not weights.any():
        return Joint(np.zeros((nviews,taps,width)),np.zeros(endo.shape[1]),weights,audit)
    weights /= weights.sum()
    rho = trust@weights
    # Weak ridge on the retained lags and gated endogenous re-adjustment.
    refit = max(original.refit,.002) if arm.l1 else original.refit
    p.flat[::len(p)+1] += np.where(np.repeat(families,spans),0.,refit-.008)
    pen = p[np.ix_(selected,selected)]
    coeff = ridge(np.column_stack((x[:,selected],rho[:,None]*sy)),y,
                  _augment(pen,len(e_scale),refit))
    full = np.zeros(nfeatures)
    full[selected] = coeff[:int(selected.sum())]
    adjustment = np.zeros(endo.shape[1])
    if arm.adjustment:
        adjustment[:] = coeff[int(selected.sum()):]/e_scale
    theta = np.zeros((nviews,taps,width))
    for pos,(g,idx,proj) in enumerate(specs):
        sl = slice(stops[pos],stops[pos+1])
        c = full[sl]/scales[pos]
        if proj is None:
            theta[g,idx,0] = c
        else:
            theta[g,:,1:] = proj@c.reshape(proj.shape[1],width-1)
    audit.update(adjustment_intercept=float(adjustment[0]),adjustment_anchor=float(adjustment[1]))
    return Joint(theta,adjustment,weights,audit)

