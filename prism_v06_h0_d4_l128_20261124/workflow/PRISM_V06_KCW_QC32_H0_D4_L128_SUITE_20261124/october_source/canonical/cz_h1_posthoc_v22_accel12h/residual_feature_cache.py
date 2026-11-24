"""Memoize unchanged mature-CF feature preparation across A alpha/mu recipes.

This cache contains target-dependent residual features, is explicitly separate
from the K input-only feature cache, and never memoizes a selected winner.
"""
from collections import OrderedDict
from .invariants import digest,frame_hash
from .a_state import TRAINING_CF,build_a_training_state_from_cf_residual,feature_rows,common_scoring_mask

def _cached(store,key,compute,limit):
    if key in store:value=store.pop(key)
    else:value=compute()
    store[key]=value
    while len(store)>limit:store.popitem(last=False)
    return value

def prepare_a(pipe,profile,fit,ev,source,*,comparison):
    from .invariants import ProtocolError
    if not hasattr(pipe,'_residual_state_cache'):
        pipe._residual_state_cache=OrderedDict();pipe._a_preparation_cache=OrderedDict();pipe._a_common_cache=OrderedDict()
    source_key=digest(dict(policy=pipe.policy_id,source=frame_hash(source),role=TRAINING_CF,h_steps=1))
    key=digest(dict(source=source_key,fit=frame_hash(fit),eval=frame_hash(ev),profile=profile,
        maximum=pipe.A_max,comparison=bool(comparison)))
    def compute():
        state=_cached(pipe._residual_state_cache,source_key,
            lambda:build_a_training_state_from_cf_residual(source,policy_id=pipe.policy_id,h_steps=1),32)
        _,native,_=feature_rows(fit,state,profile,pipe.A_max,0.,expected_role=TRAINING_CF)
        if native.sum()<2:return None,None,None,None,0.,dict(candidate_status='DOMAIN_NOT_APPLICABLE')
        lawful=fit.loc[native].copy();mean=float(lawful.residual.mean())
        x,_,fa=feature_rows(lawful,state,profile,pipe.A_max,mean,expected_role=TRAINING_CF)
        z,mask,audit=feature_rows(ev,state,profile,pipe.A_max,mean,expected_role=TRAINING_CF)
        if comparison:
            common_key=digest(dict(source=source_key,ev=frame_hash(ev),profiles=pipe.A_profiles,maximum=pipe.A_max))
            common=_cached(pipe._a_common_cache,common_key,
                lambda:common_scoring_mask(ev,state,pipe.A_profiles,pipe.A_max),128)
            mask &= common
            z=feature_rows(ev.loc[mask],state,profile,pipe.A_max,mean,expected_role=TRAINING_CF)[0]
        if not mask.any():return None,None,None,mask,mean,dict(residual_mean=mean)
        return lawful,x,z,mask,mean,audit
    return _cached(pipe._a_preparation_cache,key,compute,128)

def cached_ar_path(pipe,x,target,up):
    from .invariants import prediction_hash
    from .ridge_path import MatureARPath
    if not hasattr(pipe,'_ar_path_cache'):pipe._ar_path_cache=OrderedDict()
    key=digest(dict(policy=pipe.policy_id,shape=x.shape,design=prediction_hash(x),
        target=prediction_hash(target),upstream_shape=up.shape,upstream=prediction_hash(up)))
    return _cached(pipe._ar_path_cache,key,lambda:MatureARPath(x,target,up),64)
