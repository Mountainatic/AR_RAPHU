"""Candidate A feature initialization from an already issued stream.

History supplies features only, never extra parameter updates. Callers must
establish that the upstream model was frozen before every history origin.
This module alone is not a trained or validated replacement model.
"""
import numpy as np
import pandas as pd


def predict_spec(module, current, initializer, spec, *, history=None, update=True):
    if history is None or len(history) == 0 or spec['family'] == 'BIAS':
        return module.predict_spec(current, initializer, spec, update=update)
    if len(current) == 0:
        raise ValueError('Empty current stream')
    if initializer['fit_last_target_end'] > int(current.latest_available_target_index.iloc[0]):
        raise ValueError('Initializer contains an unavailable target')
    # Never join separate physical streams or silently fill a missing interval.
    for field in ('entity_id', 'upstream_realization_id', 'segment_id'):
        if field in current or field in history:
            if field not in current or field not in history:
                raise ValueError('History metadata mismatch: ' + field)
            if current[field].nunique() != 1 or history[field].nunique() != 1:
                raise ValueError('One uninterrupted stream required')
            if current[field].iloc[0] != history[field].iloc[-1]:
                raise ValueError('History belongs to a different stream: ' + field)
    origins = np.r_[history.origin.to_numpy(), current.origin.to_numpy()]
    if not np.all(np.diff(origins) == 1):
        raise ValueError('History and current origins must be contiguous')
    joined = pd.concat([history, current], ignore_index=True)
    # A dataset partition boundary alone does not reset the physical stream.
    joined['split_id'] = 'EXPLICIT_CONTIGUOUS_HISTORY'
    offset = len(history)
    if spec['family'] in module.RESPONSE_FAMILIES:
        x, native, fa = module.dynamic_features(joined, spec['p'], spec['family'])
    else:
        x, native, fa = module.rls.features(joined, spec['p'])
    # Only current issue rows enter the learning queue; history is not replayed
    # through a parameter state fitted later than those historical origins.
    pred, audit = module.rls.replay(current, x[offset:], native[offset:], initializer,
                                    spec['forget'], update=update)
    log = dict(latest_feature_label=fa['latest_feature_target_end'][offset:],
               latest_parameter_label=audit['latest_parameter_target_end'],
               updates=audit['total_updates'], resets=audit['resets'],
               stored_issued_features=True, feature_history_rows=offset,
               history_parameter_updates=0)
    if np.any(log['latest_feature_label'] > current.latest_available_target_index.to_numpy()):
        raise AssertionError('Unmatured feature label')
    return spec['gain'] * pred, log


def predict(module, current, fitted, *, history=None, update=True):
    pred, log = predict_spec(module, current, fitted['model'], fitted['selected'],
                             history=history, update=update)
    return pred if fitted['admitted'] else np.zeros(len(current)), log
