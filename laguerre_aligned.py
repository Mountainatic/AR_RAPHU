"""Aligned sparse levels or differences with deployment-consistent refitting."""

from dataclasses import dataclass

import numpy as np

from laguerre_ecm import Dynamics, Joint, _augment, aligned_exogenous_target, ridge
from laguerre_sparse_views import group_fit


@dataclass(frozen=True)
class Alignment:
    mode: str
    near: int
    l1: float
    curve: float


def aligned_dynamics(spec, original):
    return Dynamics(spec.near,(*original.banks,(.97,2)))


def aligned_blocks(spec,dynamics,views):
    if spec.mode not in ("level","delta"):
        raise ValueError("Unknown alignment")
    primary = int(spec.mode=="delta")
    taps = dynamics.near+sum(k for _,k in dynamics.banks)
    for g in range(views):
        if g%2 == primary:
            idx = np.arange(taps-2)
        elif primary:
            idx = np.r_[0,taps-2,taps-1]
        else:
            idx = np.arange(taps-2,taps)
        yield g,idx,None
    if spec.curve:
        r = dynamics.near
        for g in range(views):
            if g%2 == primary:
                spans = ((0,min(4,r)),(4,r),(r,taps-2))
            elif primary:
                spans = ((0,1),)
            else:
                continue
            directions = []
            for lo,hi in spans:
                if lo < hi:
                    v = np.zeros(taps)
                    v[lo:hi] = 1./np.sqrt(hi-lo)
                    directions.append(v)
            yield g,np.arange(taps),np.array(directions).T


def fit_aligned(design,dynamic,endo,increment,base_increment,rows,spec,original):
    groups,taps,width = design.raw.shape[1:]
    layout = list(aligned_blocks(spec,dynamic,groups))
    raw,trust = design.raw[rows],design.trust[rows]
    dt,et = increment[rows],base_increment[rows]
    matrices, scales, curves = [],[],[]
    for g,idx,v in layout:
        fields = raw[:,g,idx,0] if v is None else np.einsum(
            "ntp,tk->nkp",raw[:,g,:,1:],v).reshape(len(rows),-1)
        fields *= trust[:,g,None]
        scale = np.maximum(np.sqrt(np.mean(fields*fields,axis=0)),.08)
        matrices.append(fields/scale)
        scales.append(scale)
        if v is not None:
            gram = np.kron(np.eye(v.shape[1]),design.curvature[g,1:,1:]+.01*np.eye(width-1))
            curves.append(spec.curve*gram/np.outer(scale,scale))
    sizes = np.array([len(s) for s in scales])
    stop = np.r_[0,np.cumsum(sizes)]
    families = np.array([int(v is not None) for _,_,v in layout])
    x = np.column_stack(matrices)
    fc = np.repeat(families,sizes)
    nl = int(stop[groups])
    weights_anc = endo[rows,:2] if spec.mode=="level" else np.empty((len(rows),0))
    aux_scale = np.maximum(np.sqrt(np.mean(weights_anc*weights_anc,axis=0)),1e-7)
    aux = weights_anc/aux_scale
    p = np.diag(np.where(fc==0,.008,0.))
    for k,curve in zip(np.flatnonzero(families),curves,strict=True):
        p[stop[k]:stop[k+1],stop[k]:stop[k+1]] = curve
    provisional = trust.mean(axis=1)
    preliminary = aligned_exogenous_target(dt,et,provisional)
    a,audit = group_fit(np.column_stack((x,provisional[:,None]*aux)),preliminary,
                        _augment(p,len(aux_scale),.008),sizes,families,original.sparsity,spec.l1)
    active = a != 0
    norm2 = np.zeros(groups)
    for k,(g,_,v) in enumerate(layout):
        local = slice(stop[k],stop[k+1])
        norm2[g] += np.sum((a[local]/scales[k])**2)
        if v is not None:
            active[local] = active[local].any()
    weights = np.sqrt(norm2)
    theta = np.zeros((groups,taps,width))
    beta = np.zeros(endo.shape[1])
    if weights.any():
        weights /= weights.sum()
        rho = trust@weights
        y = aligned_exogenous_target(dt,et,rho)
        lam = max(original.refit,.0015)
        p.flat[::len(p)+1] += np.where(fc==0,lam-.008,0.)
        b = ridge(np.column_stack((x[:,active],rho[:,None]*aux)),y,
                  _augment(p[np.ix_(active,active)],len(aux_scale),lam))
        estimates = np.zeros(len(a))
        estimates[active] = b[:int(active.sum())]
        if spec.mode=="level":
            beta[:2] = b[int(active.sum()):]/aux_scale
        for k,(g,idx,v) in enumerate(layout):
            value = estimates[stop[k]:stop[k+1]]/scales[k]
            if v is None:
                theta[g,idx,0] = value
            else:
                theta[g,:,1:] = v@value.reshape(v.shape[1],width-1)
    audit.update(linear_total=nl,linear_active=int(active[:nl].sum()),
        nonlinear_active=int(active[nl:].sum()),active_views=int(np.count_nonzero(weights)),
        gated_intercept=float(beta[0]),gated_anchor=float(beta[1]))
    return Joint(theta,beta,weights,audit)

