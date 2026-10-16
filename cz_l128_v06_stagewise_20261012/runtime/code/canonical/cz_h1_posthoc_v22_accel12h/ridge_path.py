"""Share authority batch normalization, Gram and RHS across registered alphas."""
import numpy as np

class RidgePath:
    def __init__(self,train,target):
        from prism_benchmark.v2_c import _standardize
        self.x,_,self.mean,self.scale=_standardize(train,train)
        self.target_mean=float(np.mean(target,dtype=np.float64))
        self.gram=self.x.T@self.x
        self.rhs=self.x.T@(target-self.target_mean)
        self.rows=len(self.x)
    def solve(self,evaluation,alpha):
        from prism_benchmark.v2_numerics import solve_certified_gram
        coefficient,certificate=solve_certified_gram(self.gram,self.rhs,alpha)
        z=(evaluation-self.mean)/self.scale
        return z@coefficient+self.target_mean,dict(mean=self.mean.tolist(),scale=self.scale.tolist(),
            coefficient=coefficient.tolist(),intercept=self.target_mean,alpha=alpha,
            numerical_certificate=certificate.to_json())

class MatureARPath(RidgePath):
    """Authority A equations with shared design statistics across alpha/mu."""
    def __init__(self,train,target,upstream):
        super().__init__(train,target)
        self.upstream=np.asarray(upstream,dtype=np.float64) if upstream is not None else np.empty((len(self.x),0))
        if self.upstream.ndim==1:self.upstream=self.upstream[:,None]
        self.overlap_penalties={}
    def solve(self,evaluation,alpha,mu):
        from prism_benchmark.v2_numerics import solve_certified_gram
        from prism_benchmark.v21_w import soft_overlap_penalty
        if float(mu) not in self.overlap_penalties:
            self.overlap_penalties[float(mu)]=soft_overlap_penalty(self.x,self.upstream,float(mu))
        penalty=np.eye(self.x.shape[1],dtype=np.float64)*float(alpha)
        penalty+=self.overlap_penalties[float(mu)]
        coefficient,certificate=solve_certified_gram(self.gram,self.rhs,penalty)
        z=(evaluation-self.mean)/self.scale
        prediction=z@coefficient+self.target_mean
        train_prediction=self.x@coefficient+self.target_mean
        upstream=self.upstream;overlap=0.
        if upstream.shape[1] and np.std(train_prediction)>0:
            up_centered=upstream-upstream.mean(axis=0,dtype=np.float64)
            up_scale=upstream.std(axis=0,dtype=np.float64)
            keep=up_scale*up_scale>=1e-12
            if np.any(keep):
                up_standardized=up_centered[:,keep]/up_scale[keep]
                pred_standardized=(train_prediction-np.mean(train_prediction))/np.std(train_prediction)
                overlap=float(np.linalg.norm(up_standardized.T@pred_standardized/len(self.x)))
        return prediction,dict(family='MATURE_RESIDUAL_AR',mean=self.mean.tolist(),scale=self.scale.tolist(),
            coefficient=coefficient.tolist(),intercept=self.target_mean,alpha=float(alpha),soft_overlap_mu=float(mu),
            soft_overlap_norm=overlap,hard_feature_residualization=False,uses_input_features=False,
            numerical_certificate=certificate.to_json(),parameter_count=len(coefficient)+1)

