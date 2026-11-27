"""Level-only process identification with independent curvature refits."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from sklearn.linear_model import lars_path_gram

from laguerre_ecm import ridge


@dataclass
class Process:
    location: float
    theta: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(cls, location, theta):
        norms = np.linalg.norm(theta.reshape(len(theta),-1),axis=1)
        weights = norms/norms.sum() if norms.any() else norms
        return cls(float(location),theta,weights)

    def components(self, raw):
        return np.einsum("...gtp,gtp->...g",raw,self.theta,optimize=True)

    def predict(self, raw):
        return self.location+self.components(raw).sum(axis=-1)


def kkt_error(hessian, rhs, l1, beta):
    gradient = hessian@beta-rhs
    error = np.where(beta!=0,abs(gradient+l1*np.sign(beta)),np.maximum(abs(gradient)-l1,0.))
    return float(np.max(error))


def polish(hessian, rhs, l1, initial):
    """Resolve sign changes and inactive constraints in a strictly convex L1 fit."""
    beta = initial.copy()
    signs = np.sign(beta)
    for _ in range(4*len(beta)):
        idx = np.flatnonzero(signs)
        w = solve(hessian[np.ix_(idx,idx)],rhs[idx]-l1*signs[idx],assume_a="pos")
        violates = w*signs[idx] <= 0
        if violates.any():
            cand = idx[violates]
            denom = beta[cand]-w[violates]
            crossings = np.divide(beta[cand],denom,out=np.zeros_like(denom),where=denom!=0)
            fraction = crossings.min()
            beta[idx] += fraction*(w-beta[idx])
            drop = cand[crossings<=fraction+1e-12]
            signs[drop] = 0.
            beta[drop] = 0.
            continue
        beta[idx] = w
        inactive = signs==0
        if not inactive.any():
            return beta
        gradient = hessian@beta-rhs
        worst = int(np.argmax(np.where(inactive,abs(gradient),-np.inf)))
        if abs(gradient[worst])<=l1+1e-10:
            return beta
        signs[worst] = -np.sign(gradient[worst])
    return beta


class LinearProblem:
    def __init__(self, raw, truth, rows):
        a = raw[rows,:,:,0].reshape(len(rows),-1)
        self.xmean = a.mean(axis=0)
        self.xscale = np.maximum(a.std(axis=0),.08)
        self.x = np.ascontiguousarray((a-self.xmean)/self.xscale)
        self.ymean = truth[rows].mean()
        self.yscale = max(truth[rows].std(),1e-10)
        self.y = (truth[rows]-self.ymean)/self.yscale
        self.gram = self.x.T@self.x
        self.rhs = self.x.T@self.y/len(rows)
        self.shape = raw.shape[1:]

    def fit(self, alpha, mix):
        n = len(self.y)
        l1 = alpha*mix
        gram = self.gram.copy()
        gram.flat[::len(gram)+1] += n*alpha*(1.-mix)
        if l1==0:
            beta = solve(gram/n,self.rhs,assume_a="pos")
            iterations = 1
        else:
            # The ridge-augmented Gram converts the elastic net to a LARS lasso.
            _,_,beta,iterations = lars_path_gram(
                self.rhs*n,gram,n_samples=n,alpha_min=l1,method="lasso",
                max_iter=4*gram.shape[0],return_path=False,return_n_iter=True)
        hessian = gram/n
        kkt = kkt_error(hessian,self.rhs,l1,beta)
        if kkt>2e-7:
            beta = polish(hessian,self.rhs,l1,beta)
            kkt = kkt_error(hessian,self.rhs,l1,beta)
        if kkt>2e-7:
            raise RuntimeError(f"Elastic-net KKT residual {kkt:.3g}")
        audit = dict(iterations=int(iterations),kkt_residual=kkt)
        theta = np.zeros(self.shape)
        beta = beta*self.yscale/self.xscale
        theta[...,0] = beta.reshape(self.shape[:2])
        model = Process.build(self.ymean-self.xmean@beta,theta)
        return model,audit


def fixed_projection(near, taps):
    intervals = [(0,min(near,4))]
    intervals.extend((lo,min(near,lo+2)) for lo in range(4,near,2))
    intervals.append((near,taps))
    matrix = np.zeros((taps,len(intervals)))
    for i,(lo,hi) in enumerate(intervals):
        matrix[lo:hi,i] = 1./np.sqrt(hi-lo)
    return matrix


class CurvatureProblem:
    def __init__(self, design, truth, rows, linear, near):
        self.linear = linear
        group,taps,width = design.raw.shape[1:]
        self.project = fixed_projection(near,taps)
        directions = self.project.shape[1]
        raw = design.raw[rows]
        self.y = truth[rows]-linear.predict(raw)
        self.ymean = self.y.mean()
        a = np.einsum("ngtp,tk->ngkp",raw[...,1:],self.project,optimize=True).reshape(len(rows),-1)
        self.xmean = a.mean(axis=0)
        self.scale = np.maximum(a.std(axis=0),.08)
        self.x = (a-self.xmean)/self.scale
        self.dims = (group,directions,width-1)
        self.penalty = np.zeros((a.shape[1],a.shape[1]))
        stride = directions*(width-1)
        # Each fixed temporal direction receives the same integrated spline curvature.
        for g in range(group):
            k = np.kron(np.eye(directions),design.curvature[g,1:,1:]+.01*np.eye(width-1))
            sl = slice(g*stride,(g+1)*stride)
            self.penalty[sl,sl] = k/np.outer(self.scale[sl],self.scale[sl])

    def fit(self, alpha):
        theta = self.linear.theta.copy()
        b = ridge(self.x,self.y-self.ymean,alpha*self.penalty)/self.scale
        theta[...,1:] = np.einsum("tk,gkp->gtp",self.project,b.reshape(self.dims))
        return Process.build(self.linear.location+self.ymean-self.xmean@b,theta)

