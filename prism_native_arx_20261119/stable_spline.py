"""Degree-one feature-separable splines with explicit per-column boundaries.

This is a scientific semantics correction, not a frozen-prediction optimization.
Fit is inherited unchanged; transform keeps the degree local to each spline.
"""
import numpy as np
from sklearn.preprocessing import SplineTransformer

class StableLinearSpline(SplineTransformer):
    def transform(self,X):
        X=np.asarray(X)
        if self.degree!=1 or self.extrapolation!='linear' or self.include_bias or self.sparse_output:
            raise ValueError('This repair is registered only for degree1/linear/no-bias/dense')
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('Invalid spline inputs')
        result=[]
        for i,bs in enumerate(self.bsplines_):
            lo,hi=bs.t[bs.k],bs.t[-bs.k-1];x=X[:,i]
            out=bs(np.clip(x,lo,hi))
            below=x<lo;above=x>hi
            if below.any():out[below]=bs(lo)+(x[below,None]-lo)*bs(lo,nu=1)
            if above.any():out[above]=bs(hi)+(x[above,None]-hi)*bs(hi,nu=1)
            result.append(out[:,:-1])
        return np.concatenate(result,axis=1)
