"""Fit PCA only on the unique strict-past raw rows of a fit partition."""
import hashlib
import numpy as np


def fit_pca(accessor, samples, *, history=128):
    if history!=128:raise ValueError('This task freezes L128')
    pieces=[];identities=[]
    for entity,group in samples.groupby('entity_id',sort=False):
        rows,arrays=accessor.entities[str(entity)]
        origins=group.origin.to_numpy(np.int64)
        starts=np.searchsorted(rows,origins-history);stops=np.searchsorted(rows,origins)
        if np.any(stops-starts!=history):raise ValueError('Missing strict-past PCA raw rows')
        difference=np.zeros(len(rows)+1,np.int64)
        np.add.at(difference,starts,1);np.add.at(difference,stops,-1)
        used=np.flatnonzero(np.cumsum(difference[:-1])>0)
        pieces.append(np.column_stack([arrays['crystal_lift'][used],arrays['crucible_lift'][used]]))
        identities.extend(f'{entity}:{int(r)}' for r in rows[used])
    if not pieces:raise ValueError('Empty PCA fit partition')
    raw=np.vstack(pieces);mean=raw.mean(0);scale=raw.std(0)
    scale[scale<=np.finfo(float).eps]=1.
    values=(raw-mean)/scale
    eig,vec=np.linalg.eigh(np.cov(values,rowvar=False,ddof=0))
    loading=vec[:,np.argmax(eig)]
    if loading[0]<0:loading=-loading
    return dict(mean=mean.tolist(),scale=scale.tolist(),pc1_loading=loading.tolist(),
        fit_scope='UNIQUE_L128_STRICT_PAST_ROWS_OF_THIS_FIT_PARTITION',
        raw_rows=len(raw),raw_support_hash=hashlib.sha256('\n'.join(identities).encode()).hexdigest())


class PCAAccessor:
    def __init__(self,raw,pca):self.raw=raw;self.pca=pca
    def __getattr__(self,name):return getattr(self.raw,name)
    def block_means(self,samples,column,intervals):
        if column!='joint_lift':return self.raw.block_means(samples,column,intervals)
        if max((far for _,far in intervals),default=0)>128:raise ValueError('Profile exceeds L128')
        result=np.zeros((len(samples),len(intervals)),float)
        for j,name in enumerate(['crystal_lift','crucible_lift']):
            x=self.raw.block_means(samples,name,intervals)
            result+=(x-self.pca['mean'][j])/self.pca['scale'][j]*self.pca['pc1_loading'][j]
        return result
