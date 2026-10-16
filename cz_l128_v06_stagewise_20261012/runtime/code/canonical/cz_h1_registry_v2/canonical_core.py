"""Canonical v2.2 routing/support primitives. No legacy loss-table selector."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np

class ProtocolSupportError(RuntimeError):pass
class NumericallyInvalid(RuntimeError):pass

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    tmp.replace(path)

def prediction_hash(values):
    x=np.ascontiguousarray(values,dtype='<f8')
    return hashlib.sha256(x.tobytes()).hexdigest()

def mse_bound(y,p):
    """FP64 subtraction/product/fsum/division bound; no max(1,risk) floor."""
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
    if y.shape!=p.shape or y.ndim!=1 or not len(y):raise ValueError('Invalid paired risk support')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise NumericallyInvalid('nonfinite prediction')
    e=y-p;sq=e*e;n=len(e);u=np.finfo(np.float64).eps/2
    risk=math.fsum(map(float,sq))/n
    scale=np.abs(y)+np.abs(p)
    # Bounds on subtraction and product plus compensated sum and division.
    point=4*u*np.abs(e)*scale+4*u*u*scale*scale+u*sq
    bound=math.fsum(map(float,point))/n+4*u*abs(risk)
    return risk,bound

def risk(y,p):return mse_bound(y,p)[0]

def metrics(y,p):
    e=np.asarray(y,float)-np.asarray(p,float)
    return dict(n=len(e),mse=risk(y,p),rmse=math.sqrt(risk(y,p)),mae=math.fsum(map(float,abs(e)))/len(e),bias=math.fsum(map(float,e))/len(e))

def paired_route(y,parent,child,*,domain=True):
    r0,b0=mse_bound(y,parent);r1,b1=mse_bound(y,child)
    eps=b0+b1+np.finfo(np.float64).eps*(abs(r0)+abs(r1))
    return dict(routing_status=('NOT_APPLICABLE' if not domain else 'ADMITTED' if r0-r1>eps else 'BYPASSED'),parent_oof_risk=r0,child_oof_risk=r1,incremental_gain=r0-r1,epsilon_fp=eps,n=len(y),numeric_contract=dict(dtype='float64',normalization='mean_squared_error',summation='math.fsum of FP64 squared residuals',sample_count=len(y),scaling='observed abs(y)+abs(prediction); no absolute scale floor',subtraction_and_product_bounds=True),fold_count_selection_authority=False)

def deterministic_best(losses):
    valid=[(key,float(value)) for key,value in losses.items() if np.isfinite(value)]
    if not valid:raise NumericallyInvalid('No certified registered MODEL recipe')
    # Registration insertion order is the deterministic tie-break.
    lowest=min(v for _,v in valid);u=np.finfo(float).eps
    return next(k for k,v in valid if v-lowest<=4*u*(abs(v)+abs(lowest)))

def folds(frame,block=8,count=4):
    """Past-only tail scoring; original raw dependency purge, without relaxation."""
    if len(frame)<=count*block:raise ProtocolSupportError('No inner scoring blocks')
    result=[]
    for j in range(count):
        start=len(frame)-count*block+j*block
        ei=np.arange(start,start+block)
        cutoff=int(frame.iloc[ei].dependency_start.min())-1
        fi=np.flatnonzero(frame.dependency_stop_exclusive.to_numpy(np.int64)<=cutoff)
        if not len(fi):raise ProtocolSupportError('Empty purged fit')
        assert int(frame.iloc[fi].dependency_stop_exclusive.max())<int(frame.iloc[ei].dependency_start.min())
        result.append((fi,ei))
    return result

def cf_blocks(frame,block=8,warmup=640):
    ordinary=folds(frame,block)
    result=[]
    for j,(fi,ei) in enumerate(ordinary):
        if j==0:
            start=int(ei[0])-warmup
            if start<0:raise ProtocolSupportError('No CF warmup rows')
            ei=np.arange(start,int(ei[-1])+1)
            cutoff=int(frame.iloc[ei].dependency_start.min())-1
            fi=np.flatnonzero(frame.dependency_stop_exclusive.to_numpy(np.int64)<=cutoff)
            if not len(fi):raise ProtocolSupportError('No purged fit for CF warmup')
        assert int(frame.iloc[fi].dependency_stop_exclusive.max())<int(frame.iloc[ei].dependency_start.min())
        result.append((fi,ei,j))
    return result

def common_mature(frame,source,history,h_steps,w_steps,native_mask):
    latest=frame.latest_available_target_index.to_numpy(np.int64)-h_steps-w_steps+1
    mask=native_mask(frame,history)
    for entity in frame.entity_id.unique():
        ix=np.flatnonzero((frame.entity_id==entity).to_numpy())
        known=set(source.loc[source.entity_id==entity,'origin'].astype(int))
        mask[ix]&=np.array([all(int(t)-o in known for o in range(history)) for t in latest[ix]])
    return mask
