"""Optional immutable observation metadata. No imports from model/selection code."""
from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType
import json
import math


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite observation metadata')
        return value
    raise TypeError('Observation metadata must be JSON serializable')


def _thaw(value):
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class DiscreteObservationSpec:
    sample_period_seconds: float | None = None
    time_origin_policy: str = 'UNKNOWN'
    timestamp_field: str | None = None
    output_unit: str = 'recorded_unit_unverified'
    output_record_grid_candidate: float | None = None
    grid_evidence_status: str = 'unknown'
    segment_policy: str = 'EXPLICIT_ENTITY_SEGMENT_AND_RAW_INDEX'
    gap_rule: Mapping | None = None
    observation_availability_policy: str = 'UNKNOWN'
    head_contract_id: str = 'UNSPECIFIED'

    def __post_init__(self):
        for name in ('sample_period_seconds', 'output_record_grid_candidate'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(name + ' must be positive or unknown')
        if self.grid_evidence_status not in {'unknown', 'observed_grid_candidate', 'instrument_verified'}:
            raise ValueError('Unknown grid evidence status')
        if self.grid_evidence_status != 'unknown' and self.output_record_grid_candidate is None:
            raise ValueError('Grid evidence needs an explicit candidate')
        if self.gap_rule is not None:
            if not isinstance(self.gap_rule, Mapping):
                raise TypeError('gap_rule must be a mapping')
            object.__setattr__(self, 'gap_rule', _freeze(self.gap_rule))

    def to_dict(self):
        return {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))


def optional_observation_spec(config):
    """Called by audit clients only; missing field does not alter legacy config."""
    value = config.get('discrete_observation')
    return None if value is None else DiscreteObservationSpec(**value)
