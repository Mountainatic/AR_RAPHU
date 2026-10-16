"""Explicit partition-fitted PCA for the inherited baseline feature APIs."""
import copy
import numpy as np
from prism_benchmark.cpu_data import BaseAccessor
from canonical.cz_crossrod_l128_20261008.joint_lift import fit_pca

class FoldPCAAccessor(BaseAccessor):
    def __init__(self,shared,dataset,split,columns):
        super().__init__(shared,dataset,split,list(dict.fromkeys([*columns,'crystal_lift','crucible_lift'])))
        self.original_entities=self.entities;self.pca=None

    def for_fit(self,samples):
        raw=copy.copy(self);raw.entities=self.original_entities
        return self.with_pca(fit_pca(raw,samples))

    def with_pca(self,pca):
        result=copy.copy(self);result.pca=pca
        result.entities={}
        for entity,(rows,arrays) in self.original_entities.items():
            values=dict(arrays)
            values['joint_lift']=sum((values[c]-pca['mean'][j])/pca['scale'][j]*pca['pc1_loading'][j] for j,c in enumerate(['crystal_lift','crucible_lift']))
            result.entities[entity]=(rows,values)
        # BaseAccessor prefixes are derived from entities and must be rebuilt.
        result._count_prefix={};result._value_prefix={}
        return result

    def gather(self,samples,columns,indices):
        if 'joint_lift' in columns and self.pca is None:raise RuntimeError('PCA_MUST_BE_FIT_EXPLICITLY_BEFORE_BASELINE_FEATURES')
        indices=np.asarray(indices)
        lower=samples.origin.to_numpy(np.int64)-128
        upper=samples.origin.to_numpy(np.int64)-1
        checked=indices[:,None] if indices.ndim==1 else indices
        if np.any(checked<lower[:,None]) or np.any(checked>upper[:,None]):raise ValueError('Baseline feature outside strict-past L128')
        return super().gather(samples,columns,indices)

    def block_means(self,samples,column,intervals):
        if column=='joint_lift' and self.pca is None:raise RuntimeError('PCA_MUST_BE_FIT_EXPLICITLY')
        if any(near<0 or far>128 for near,far in intervals):raise ValueError('Profile exceeds L128')
        return super().block_means(samples,column,intervals)
