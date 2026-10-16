"""One FP64 MSE error-bound contract for MODEL equivalence and paired routing."""
from dataclasses import dataclass
import math
import numpy as np
from .invariants import ProtocolError,NumericallyInvalid


@dataclass(frozen=True)
class RiskEvidence:
    value:float
    bound:float
    rows:int

    def to_json(self):
        return dict(value=self.value,bound=self.bound,rows=self.rows,
            dtype='float64',normalization='mean_squared_error',summation='math.fsum',
            scaling='observed abs(target)+abs(prediction); no absolute floor',
            bound_contract='INHERITED_CANONICAL_MSE_SUBTRACTION_PRODUCT_FSUM_DIVISION_V1')


def mse_evidence(y,p):
    y=np.asarray(y,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or y.ndim!=1 or not len(y):
        raise ProtocolError('Invalid paired MSE support')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise NumericallyInvalid('Nonfinite risk target/prediction')
    error=y-p
    squares=error*error
    if not np.isfinite(squares).all():
        raise NumericallyInvalid('Risk overflow')
    n=len(y)
    u=np.finfo(np.float64).eps/2
    value=math.fsum(map(float,squares))/n
    scale=np.abs(y)+np.abs(p)
    point=4*u*np.abs(error)*scale+4*u*u*scale*scale+u*squares
    bound=math.fsum(map(float,point))/n+4*u*abs(value)
    return RiskEvidence(value,bound,n)


def aggregate(evidence):
    evidence=list(evidence)
    if not evidence:
        raise ProtocolError('No risk observations to aggregate')
    n=sum(e.rows for e in evidence)
    total=math.fsum(e.value*e.rows for e in evidence)
    value=total/n
    bound=math.fsum(e.bound*e.rows for e in evidence)/n
    # Additional FP64 weighting/summation/division; never a statistical band.
    bound += 4*np.finfo(float).eps*abs(value)
    return RiskEvidence(value,bound,n)


def equivalence_bound(left,right):
    if left.rows!=right.rows:
        raise ProtocolError('MODEL risk equivalence requires equal scoring rows')
    return left.bound+right.bound+np.finfo(float).eps*(abs(left.value)+abs(right.value))


def registered_best(evidence):
    """Exact minimum first; equivalent risks use only registration order."""
    valid=[(key,risk) for key,risk in evidence.items()
        if np.isfinite(risk.value) and np.isfinite(risk.bound)]
    if not valid:
        raise NumericallyInvalid('No numerically valid registered MODEL recipe')
    if len({risk.rows for _,risk in valid})!=1:
        raise ProtocolError('MODEL recipes have different scoring supports')
    minimum=min((risk for _,risk in valid),key=lambda risk:risk.value)
    return next(key for key,risk in valid
        if risk.value-minimum.value<=equivalence_bound(risk,minimum))


def paired_route(y,parent,child,*,domain=True):
    r0=mse_evidence(y,parent)
    r1=mse_evidence(y,child)
    epsilon=equivalence_bound(r0,r1)
    gain=r0.value-r1.value
    return dict(routing_status='NOT_APPLICABLE' if not domain else 'ADMITTED' if gain>epsilon else 'BYPASSED',
        parent_oof_risk=r0.value,child_oof_risk=r1.value,incremental_gain=gain,
        epsilon_fp=epsilon,n=r0.rows,parent_risk_evidence=r0.to_json(),child_risk_evidence=r1.to_json(),
        statistical_evidence_has_selection_authority=False)
