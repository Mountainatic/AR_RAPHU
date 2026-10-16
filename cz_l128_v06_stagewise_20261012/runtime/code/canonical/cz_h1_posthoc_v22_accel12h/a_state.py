"""Explicit CZ maturity and distinct CF-training / deployment-inference state."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .invariants import ProtocolError,frame_hash,digest,prediction_hash

TRAINING_CF='A_TRAINING_CROSS_FITTED_W_RESIDUAL'
INFERENCE_DEPLOYED='A_INFERENCE_DEPLOYED_W_RESIDUAL'


def cz_target_end(frame,h_steps,w_steps=1):
    """Inclusive target end for exclusive CZ origins, checked against metadata."""
    if h_steps<1 or w_steps<1:
        raise ProtocolError('CZ requires explicit positive horizon/window')
    expected=frame.origin.to_numpy(np.int64)+int(h_steps)+int(w_steps)-2
    if 'target_stop_exclusive' not in frame:
        raise ProtocolError('Explicit target-window metadata is required')
    actual=frame.target_stop_exclusive.to_numpy(np.int64)-1
    if not np.array_equal(actual,expected):
        raise ProtocolError('CZ target end metadata disagrees with registered head')
    return actual


@dataclass(frozen=True)
class AState:
    role:str
    upstream_id:str
    frame:pd.DataFrame
    content_hash:str
    h_steps:int
    w_steps:int

    def validate(self,expected_role=None,upstream_id=None):
        if expected_role is not None and self.role!=expected_role:
            raise ProtocolError('A training CF state cannot seed deployment inference')
        if upstream_id is not None and self.upstream_id!=upstream_id:
            raise ProtocolError('A inference residuals belong to another deployed W')
        if frame_hash(self.frame)!=self.content_hash:
            raise ProtocolError('A residual state changed after binding')

    @property
    def state_id(self):
        return digest(dict(role=self.role,upstream_id=self.upstream_id,
            content_hash=self.content_hash,h=self.h_steps,w=self.w_steps))


def _state(frame,role,upstream_id,h_steps,w_steps):
    if not upstream_id:
        raise ProtocolError('A state requires upstream provenance')
    required={'entity_id','origin','residual','target_stop_exclusive'}
    if not required.issubset(frame):
        raise ProtocolError('A state lacks explicit residual/target-end metadata')
    if frame[['entity_id','origin']].duplicated().any():
        raise ProtocolError('Duplicate residual origins; CF and deployed sources must never merge')
    if not np.isfinite(frame.residual.to_numpy(np.float64)).all():
        raise ProtocolError('A state has nonfinite residuals')
    selected=frame.copy(deep=True)
    selected['residual_target_end']=cz_target_end(selected,h_steps,w_steps)
    return AState(role,str(upstream_id),selected,frame_hash(selected),int(h_steps),int(w_steps))


def build_a_training_state_from_cf_residual(cf_residuals,*,policy_id,h_steps,w_steps=1):
    if 'residual_source_role' not in cf_residuals or set(cf_residuals.residual_source_role)!={'CROSS_FITTED_PREQUENTIAL'}:
        raise ProtocolError('A training state requires explicitly cross-fitted residuals')
    return _state(cf_residuals,TRAINING_CF,policy_id,h_steps,w_steps)


def build_a_inference_state_from_deployment_residual(observations,deployed_w_prediction,*,deployment_model_id,h_steps,w_steps=1):
    """Use this exact deployed W realization; caller prediction API never fits."""
    prediction=np.asarray(deployed_w_prediction,dtype=np.float64)
    if prediction.shape!=(len(observations),) or not np.isfinite(prediction).all():
        raise ProtocolError('Invalid deployed W predictions for inference state')
    frame=observations.copy(deep=True)
    frame['residual']=frame.y_true.to_numpy(np.float64)-prediction
    frame['residual_source_role']='DEPLOYED_W_ONLINE'
    return _state(frame,INFERENCE_DEPLOYED,deployment_model_id,h_steps,w_steps)


def offsets(profile,maximum_lags):
    delta,history=map(int,profile)
    if delta<1 or history<1 or maximum_lags<1:
        raise ProtocolError('Invalid frozen A profile')
    count=min(int(maximum_lags),max(1,history//delta))
    return np.unique(np.rint(np.linspace(0,max(0,history-delta),count)).astype(np.int64))


def feature_rows(samples,state,profile,maximum_lags,mean,*,expected_role=None,upstream_id=None):
    """Return complete mature rows only. No missing-history numerical imputation."""
    state.validate(expected_role,upstream_id)
    cz_target_end(samples,state.h_steps,state.w_steps)
    latest=samples.latest_available_target_index.to_numpy(np.int64)
    origin=samples.origin.to_numpy(np.int64)
    if np.any(latest>=origin):
        raise ProtocolError('CZ latest target must precede prediction origin')
    newest=latest-state.h_steps-state.w_steps+2
    lag_offsets=offsets(profile,maximum_lags)
    complete=np.zeros(len(samples),dtype=bool)
    columns=np.full((len(samples),len(lag_offsets)),np.nan,dtype=np.float64)
    entities=samples.entity_id.astype(str).to_numpy()
    for entity in np.unique(entities):
        ix=np.flatnonzero(entities==entity)
        source=state.frame.loc[state.frame.entity_id.astype(str)==entity].sort_values('origin')
        if not len(source):
            continue
        origins=source.origin.to_numpy(np.int64)
        query=newest[ix,None]-lag_offsets[None,:]
        position=np.searchsorted(origins,query)
        safe=np.minimum(position,len(origins)-1)
        present=(position<len(origins))&(origins[safe]==query)
        ends=source.residual_target_end.to_numpy(np.int64)[safe]
        mature=ends<=latest[ix,None]
        usable=np.all(present&mature,axis=1)
        keep=ix[usable]
        complete[keep]=True
        if len(keep):
            residual=source.residual.to_numpy(np.float64)
            columns[keep]=residual[safe[usable]]-float(mean)
    audit=dict(state_role=state.role,state_id=state.state_id,upstream_id=state.upstream_id,
        source_content_hash=state.content_hash,offsets=lag_offsets.tolist(),
        sample_support_hash=frame_hash(samples),
        sample_rows=len(samples),complete_rows=int(complete.sum()),
        missing_history_rows=int((~complete).sum()),zero_fill=False,
        mature_feature_hash=digest(dict(native_support_hash=frame_hash(samples.loc[complete]),
            shape=columns[complete].shape,values_hash=prediction_hash(columns[complete]))),
        newest_residual_origin_formula='latest_target-h-W+2 (exclusive CZ origin)',
        no_cross_entity_reads=True,only_mature_target_windows=True)
    return columns[complete],complete,audit


def common_scoring_mask(samples,state,profiles,maximum_lags):
    mask=np.ones(len(samples),dtype=bool)
    for profile in profiles:
        _,native,_=feature_rows(samples,state,profile,maximum_lags,0.)
        mask &= native
    return mask
