"""Read-only CZ index adapter. Inclusive endpoints, exclusive prediction origin.

This adapter reports invalid support; it NEVER filters model or scoring rows.
The registered CZ implementation uses project h=1 for mother-window h=0.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CZHeadAuditConfig:
    head_id: str
    horizon_steps: int
    history_steps: int = 128
    stride: int = 1
    delivery_delay_steps: int = 0

    def __post_init__(self):
        if self.horizon_steps < 1 or self.history_steps < 1 or self.stride < 1 or self.delivery_delay_steps < 0:
            raise ValueError('Invalid registered CZ head audit config')


def observation_rows(data, spec):
    """Assign diagnostic runs without sorting, joining gaps, or imputing values.

    Required: entity_id, raw_index, recorded_level; segment_id recommended.
    timestamp_seconds is numeric seconds. Explicit timestamp_field wins.
    """
    required = {'entity_id', 'raw_index', 'recorded_level'}
    if not required.issubset(data):
        raise ValueError('Observation rows lack ' + str(required - set(data)))
    result = data.copy(deep=True)
    if 'segment_id' not in result:
        result['segment_id'] = 'UNSPECIFIED_SEGMENT'
    if 'split_id' not in result:
        result['split_id'] = 'UNSPECIFIED_SPLIT'
    timestamp = spec.timestamp_field
    if timestamp is not None:
        if timestamp not in result:
            raise ValueError('Declared timestamp field absent')
        if pd.api.types.is_numeric_dtype(result[timestamp]):
            result['timestamp_seconds'] = result[timestamp].astype(float)
        else:
            t = pd.to_datetime(result[timestamp], errors='coerce', utc=True)
            result['timestamp_seconds'] = t.astype('int64').to_numpy(dtype=float) / 1e9
            result.loc[t.isna(), 'timestamp_seconds'] = np.nan
        result['timestamp_evidence'] = 'RECORDED_TIMESTAMP'
    else:
        ts = spec.sample_period_seconds
        result['timestamp_seconds'] = np.nan if ts is None else result.raw_index.to_numpy(float) * ts
        result['timestamp_evidence'] = 'NOT_RECORDED_PROTOCOL_COORDINATE' if ts else 'NOT_AVAILABLE'
    keys = ['entity_id', 'segment_id', 'split_id']
    boundary = result[keys].ne(result[keys].shift()).any(axis=1).to_numpy()
    index = result.raw_index.to_numpy(float)
    t = result.timestamp_seconds.to_numpy(float)
    if len(result) > 1:
        boundary[1:] |= np.diff(index) != 1
        if timestamp is not None:
            dt = np.diff(t)
            boundary[1:] |= (~np.isfinite(dt)) | (dt <= 0)
            ts = spec.sample_period_seconds
            # A declared physical cadence requires a regular diagnostic run.
            if ts is not None:
                tolerance = float((spec.gap_rule or {}).get('cadence_tolerance_seconds', max(1e-8, ts * 1e-6)))
                boundary[1:] |= np.abs(dt - ts) > tolerance
            elif spec.gap_rule and 'maximum_gap_seconds' in spec.gap_rule:
                boundary[1:] |= dt > float(spec.gap_rule['maximum_gap_seconds'])
            else:
                # Without known cadence, do not claim a regular physical PSD.
                boundary[:] = True
    if len(result):
        boundary[0] = True
    result['diagnostic_segment_id'] = np.cumsum(boundary)
    result['observation_present'] = np.isfinite(result.recorded_level.to_numpy(float))
    return result


def audit_target_contract(data, head_config, observation_spec, *, samples=None):
    """Explain the unchanged legacy D[t+h-1]-D[t-1] target.

    Optional samples supplies legacy origins/entity_id. Otherwise all recorded
    origins are audited, including boundary/short-segment rows. split boundary
    uses dependent observation rows and reports violations, without dropping.
    Optional sample controls_available_until, expected_entity_id and
    purge_valid provide explicit scenario/entity/purge checks.
    """
    head = head_config
    rows = observation_rows(data, observation_spec)
    if rows[['entity_id', 'raw_index']].duplicated().any():
        raise ValueError('Ambiguous raw indices within entity')
    lookup = {(str(r.entity_id), int(r.raw_index)): r for r in rows.itertuples(index=False)}
    if samples is None:
        samples = rows.rename(columns={'raw_index': 'origin'})[['entity_id', 'origin', 'segment_id', 'split_id']]
        samples = samples.iloc[::head.stride]
    records = []
    for sample in samples.itertuples(index=False):
        entity, origin = str(sample.entity_id), int(sample.origin)
        anchor, target = origin - 1, origin + head.horizon_steps - 1
        support = list(range(origin - head.history_steps, target + 1))
        dep = [lookup.get((entity, i)) for i in support]
        known = [r for r in dep if r is not None]
        at = lookup.get((entity, origin)) or lookup.get((entity, anchor))
        reasons = []
        if len(known) != len(dep):
            reasons.append('MISSING_OR_SHORT_SEGMENT_SUPPORT')
        if any(not r.observation_present for r in known):
            reasons.append('NONFINITE_OBSERVATION')
        if len({r.diagnostic_segment_id for r in known}) > 1:
            reasons.append('CROSS_SEGMENT_OR_GAP')
        if len({r.split_id for r in known}) > 1:
            reasons.append('CROSS_SPLIT_DEPENDENCY_REQUIRES_PROTOCOL_REVIEW')
        segment = getattr(sample, 'segment_id', getattr(at, 'segment_id', 'NOT_AVAILABLE'))
        split = getattr(sample, 'split_id', getattr(at, 'split_id', 'NOT_AVAILABLE'))
        if any(r.segment_id != segment for r in known):
            reasons.append('CROSS_EXPLICIT_SEGMENT')
        if any(r.entity_id != entity for r in known) or str(getattr(sample, 'expected_entity_id', entity)) != entity:
            reasons.append('CROSS_ENTITY')
        if not bool(getattr(sample, 'purge_valid', True)):
            reasons.append('PURGE_VIOLATION')
        last_input = int(getattr(sample, 'last_input_available_index', origin - 1))
        last_output = int(getattr(sample, 'last_output_available_index', origin - 1))
        if anchor > min(last_input, last_output):
            reasons.append('ANCHOR_NOT_AVAILABLE')
        if last_input >= origin or bool(getattr(sample, 'target_used_as_predictor', False)):
            reasons.append('FUTURE_PREDICTOR_LEAK')
        if getattr(sample, 'controls_available_until', origin - 1) < origin - 1:
            reasons.append('CONTROL_HISTORY_NOT_AVAILABLE')
        if bool(getattr(sample, 'requires_future_controls', False)) and getattr(sample, 'controls_available_until', origin - 1) < target:
            reasons.append('UNKNOWN_FUTURE_CONTROL_SCENARIO')
        a, y = lookup.get((entity, anchor)), lookup.get((entity, target))
        target_value = float(y.recorded_level - a.recorded_level) if a and y else np.nan
        mismatch = hasattr(sample, 'y_true') and not np.isclose(float(sample.y_true), target_value, atol=1e-12, rtol=0, equal_nan=False)
        if mismatch:
            reasons.append('LEGACY_LABEL_MISMATCH')
        for name, expected in [('anchor_index', anchor), ('current_start', anchor),
                ('current_stop_exclusive', origin), ('target_index', target),
                ('target_start', target), ('target_stop_exclusive', target+1),
                ('history_start', origin-head.history_steps), ('history_stop_exclusive', origin),
                ('dependency_start', origin-head.history_steps), ('dependency_stop_exclusive', target+1)]:
            if hasattr(sample, name) and int(getattr(sample, name))!=expected:
                reasons.append('DECLARED_'+name.upper()+'_MISMATCH')
        ts = observation_spec.sample_period_seconds
        # Missing timestamps remain explicit; time coordinates inferred from
        # cadence are labeled and never claimed as measured feedback times.
        time_at = lambda i: getattr(lookup.get((entity, i)), 'timestamp_seconds', np.nan)
        maturity = target + head.delivery_delay_steps
        ready_time = time_at(target)
        if ts is not None and np.isfinite(ready_time):
            ready_time += head.delivery_delay_steps * ts
        records.append(dict(entity_id=entity, segment_id=segment, split_id=split,
            head_id=head.head_id, observation_head_contract_id=observation_spec.head_contract_id,
            origin_index=origin, origin_timestamp_seconds=time_at(origin),
            origin_raw_global_index=getattr(lookup.get((entity, origin)), 'raw_global_index', None),
            anchor_raw_global_index=getattr(a, 'raw_global_index', None),
            target_raw_global_index=getattr(y, 'raw_global_index', None),
            anchor_start_index=anchor, anchor_end_index=anchor,
            target_start_index=target, target_end_index=target,
            anchor_timestamp_seconds=time_at(anchor), target_timestamp_seconds=time_at(target),
            last_input_available_index=last_input, last_output_available_index=last_output,
            history_start_index=origin-head.history_steps, history_end_index=origin-1,
            label_maturity_index=maturity, target_ready_time=ready_time,
            feedback_available_time=ready_time, horizon_steps=head.horizon_steps,
            mother_window_h=head.horizon_steps-1, stride=head.stride,
            target_formula='D[t+h-1]-D[t-1]', registered_target=target_value,
            prediction_value_unit=observation_spec.output_unit,
            target_value_unit=observation_spec.output_unit,
            legal_support=not reasons, invalid_reason=';'.join(reasons),
            audit_status='OK' if not reasons else 'POTENTIAL_LEAK_OR_INVALID_SUPPORT',
            timestamp_evidence=getattr(at, 'timestamp_evidence', 'NOT_AVAILABLE')))
    return pd.DataFrame(records)
