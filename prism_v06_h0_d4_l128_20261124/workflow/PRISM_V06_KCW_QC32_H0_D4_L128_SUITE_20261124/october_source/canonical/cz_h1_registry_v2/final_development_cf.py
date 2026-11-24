"""Full-development replay; selection folds and scientific operators stay upstream."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from canonical_core import ProtocolSupportError, NumericallyInvalid, digest, dump, file_sha, prediction_hash
from prism_benchmark.v211_support import support_id_hash, native_support_mask

PROTOCOL_PATH = Path(__file__).with_name('FINAL_DEVELOPMENT_CF_PROTOCOL.json')
PROTOCOL_SHA256_LF = '049362ddfb77ab0aa9b3c668dac4298fd79367da871f9c99423bb62149dd6dc3'


def frozen_contract():
    raw = PROTOCOL_PATH.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256_LF:
        raise ProtocolSupportError('Frozen final-development CF protocol file changed')
    return json.loads(raw)


def require_contract(pipe):
    contract = frozen_contract()
    if pipe.registration.get('final_development_cf') != contract:
        raise ProtocolSupportError('Missing/mismatched frozen final-development CF contract')
    if pipe.policy_id != digest(pipe.registration):
        raise ProtocolSupportError('Final replay policy digest is stale')
    if pipe.block != contract['inner_selection_block_rows'] or pipe.warmup != contract['inner_selection_warmup_rows']:
        raise ProtocolSupportError('Inner-selection geometry changed')
    return contract


def frame_hash(frame):
    schema = digest([(str(c), str(frame[c].dtype)) for c in frame])
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype='<u8')
    return hashlib.sha256(schema.encode() + values.tobytes()).hexdigest()


class RecursiveSupport:
    """Support-only call graph of learn/tune(C/W), including nested K folds.

    Original strict raw purge and common K scoring history are reproduced.
    Domain/numerical success is deliberately not inferred from metadata.
    All registered representations must be constructable, including the long
    profiles prepared by the authority structural grid before certification.
    """
    def __init__(self, pipe, frame):
        self.block = pipe.block
        self.warmup = pipe.warmup
        self.common = max(h for _, h in pipe.k_grid('main_heater_power')[0])
        self.start = frame.dependency_start.to_numpy(np.int64)
        self.stop = frame.dependency_stop_exclusive.to_numpy(np.int64)
        self.native = native_support_mask(frame, self.common)
        self.origin = frame.origin.to_numpy(np.int64)
        self.entity = frame.entity_id.astype(str).to_numpy()
        self.memo = {}

    def splits(self, rows, cf=False):
        n = len(rows)
        if n <= 4 * self.block:
            raise ProtocolSupportError('No inner scoring blocks')
        result = []
        for j in range(4):
            first = n - 4 * self.block + j * self.block
            if cf and j == 0:
                first -= self.warmup
                if first < 0:
                    raise ProtocolSupportError('No CF warmup rows')
            ev = rows[first:n - 4 * self.block + (j + 1) * self.block]
            fit = rows[self.stop[rows] < self.start[ev].min()]
            if not len(fit) or np.any(fit >= ev[0]):
                raise ProtocolSupportError('No strictly past purged recursive fit')
            result.append((fit, ev))
        return result

    def ready(self, stage, rows, operation='learn'):
        rows = np.asarray(rows, dtype=np.int64)
        key = (stage, operation, hashlib.sha256(rows.tobytes()).digest())
        if key in self.memo:
            return self.memo[key]
        try:
            if operation == 'learn':
                parent = {'C': 'K', 'W': 'C'}.get(stage)
                ok = parent is None or self.ready(parent, rows)
                if ok:
                    ok = all(self.ready(stage, fit, 'tune') for fit, _ in self.splits(rows))
                if ok:
                    ok = self.ready(stage, rows, 'tune')
            elif stage == 'K':
                ok = True
                for fit, ev in self.splits(rows):
                    # Long-profile preparation needs nonempty native inputs;
                    # the common scoring rows must retain every registered ID.
                    ok &= bool(self.native[fit].any() and self.native[ev].all())
                    for entity in np.unique(self.entity[ev]):
                        ee = ev[self.entity[ev] == entity]
                        if entity in self.entity[fit]:
                            ok &= bool(np.all(self.origin[ee] - self.common >= self.start[ee].min()))
                    if not ok:
                        break
            elif stage == 'C':
                ok = self.ready('K', rows) and all(self.ready('K', fit) for fit, _ in self.splits(rows))
            elif stage == 'W':
                ok = self.ready('C', rows) and all(self.ready('C', fit) for fit, _ in self.splits(rows, cf=True))
            else:
                raise ValueError(stage)
        except ProtocolSupportError:
            ok = False
        self.memo[key] = bool(ok)
        return bool(ok)


def plan_replay(pipe, upstream, frame):
    contract = require_contract(pipe)
    if upstream not in ('C', 'W'):
        raise ValueError('Only C->W and W->A final replay are registered')
    if set(frame.split.astype(str)) != {'train'}:
        raise ProtocolSupportError('Replay is restricted to the development train partition')
    if frame[['entity_id', 'origin']].duplicated().any() or frame.base_origin_id.duplicated().any():
        raise ProtocolSupportError('Duplicate replay IDs')
    entities = frame.entity_id.astype(str).to_numpy()
    seen = set()
    runs = []
    first = 0
    while first < len(frame):
        entity = entities[first]
        if entity in seen:
            raise ProtocolSupportError('Replay entities must be contiguous')
        seen.add(entity)
        end = first + 1
        while end < len(frame) and entities[end] == entity:
            end += 1
        if np.any(np.diff(frame.origin.iloc[first:end].to_numpy(np.int64)) <= 0):
            raise ProtocolSupportError('Replay origins must increase within entity')
        runs.append((first, end))
        first = end
    support = RecursiveSupport(pipe, frame)
    blocks = []
    excluded = []
    for first, end in runs:
        cursor = first
        while cursor < end:
            # Start depends only on the current origin and earlier metadata.
            # Block length never affects the eligibility test for its first row.
            fit = np.flatnonzero(support.stop[:cursor] < support.start[cursor])
            if not support.native[cursor] or not support.ready(upstream, fit):
                excluded.append(cursor)
                cursor += 1
                continue
            stop = min(cursor + int(contract['block_rows']), end)
            ev = np.arange(cursor, stop, dtype=np.int64)
            # Monotone dependency starts inside a lawful contiguous entity run.
            if np.any(support.start[ev] < support.start[cursor]) or not support.native[ev].all():
                raise ProtocolSupportError('Nonmonotone replay support inside block')
            blocks.append((fit, ev))
            cursor = stop
    covered = np.concatenate([ev for _, ev in blocks]) if blocks else np.array([], dtype=np.int64)
    omitted = np.asarray(excluded, dtype=np.int64)
    assert len(covered) + len(omitted) == len(frame)
    assert len(np.unique(covered)) == len(covered)
    audit = dict(role='FINAL_DEVELOPMENT_CF_REPLAY', upstream_stage=upstream,
        contract=contract, partition_hash=support_id_hash(frame), partition_rows=len(frame),
        planned_replay_rows=len(covered), planned_support_hash=support_id_hash(frame.iloc[covered]),
        block_count=len(blocks), block_lengths=[len(ev) for _, ev in blocks],
        excluded_rows=len(omitted), excluded_support_hash=support_id_hash(frame.iloc[omitted]),
        exclusion_reason='NO_COMPLETE_RECURSIVE_METADATA_SUPPORT_BEFORE_THIS_ORIGIN',
        exclusions_by_entity=[dict(entity=str(entity),rows=len(g),first_origin=int(g.origin.min()),last_origin=int(g.origin.max()))
            for entity,g in frame.iloc[omitted].groupby('entity_id',sort=False)],
        no_fixed_final_warmup=True, tail_retained=True,
        numerical_and_domain_execution='NOT_YET_RUN',
        upstream_domain_assumption='complete recursive geometry; no outcome-dependent shortcut')
    return blocks, audit


def replay(pipe, upstream, frame, gid, selection_freeze, channels):
    if pipe.metadata_only:
        raise ProtocolSupportError('NO_TRAINING_INTEGRATION_PREFLIGHT')
    contract = require_contract(pipe)
    freeze = dict(selection_freeze)
    required = dict(policy_id=pipe.policy_id, partition_id=support_id_hash(frame),
        upstream_stage=upstream, role='FROZEN_RESIDUAL_RECIPE_BEFORE_FINAL_REPLAY',
        replay_contract=contract)
    if any(freeze.get(k) != value for k,value in required.items()) or 'selected_spec' not in freeze:
        raise ProtocolSupportError('Final replay needs this partition/policy/selected recipe freeze')
    binding = dict(role='FINAL_DEVELOPMENT_CF_REPLAY', selection_freeze=freeze,
        input_frame_hash=frame_hash(frame), growth=gid)
    key = digest(binding)
    path = pipe.root/'final_development_streams'/upstream/key/'prequential.parquet'
    metadata = path.with_suffix('.json')
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ProtocolSupportError('Incomplete final-development replay cache')
        audit = json.loads(metadata.read_text(encoding='utf-8'))
        if audit['binding'] != json.loads(json.dumps(binding,default=str)) or audit['file_sha256'] != file_sha(path):
            raise ProtocolSupportError('Final-development replay cache binding/hash mismatch')
        source = pd.read_parquet(path)
        if support_id_hash(source) != audit['coverage']['planned_support_hash'] or frame_hash(source) != audit['source_frame_hash']:
            raise ProtocolSupportError('Final-development replay cache support mismatch')
        return source, audit, str(path)
    blocks, coverage = plan_replay(pipe, upstream, frame)
    if not blocks:
        raise ProtocolSupportError('No lawful full-development replay blocks')
    sources = []
    provenance = []
    for index,(fi,ei) in enumerate(blocks):
        fit = frame.iloc[fi].copy()
        ev = frame.iloc[ei].copy()
        parent = pipe.learn(upstream, fit, gid)
        if parent['stage'] != upstream or parent['partition_id'] != support_id_hash(fit) or parent['admission_policy_id'] != pipe.policy_id:
            raise ProtocolSupportError('Replay upstream realization belongs to another prefix/policy')
        if len(pipe.support_for(parent,ev)) != len(ev):
            raise ProtocolSupportError('Frozen upstream realization loses planned replay support')
        prediction = np.asarray(pipe.emit(parent,ev),dtype=np.float64)
        if prediction.shape != (len(ev),) or not np.isfinite(prediction).all():
            raise NumericallyInvalid('Invalid full-development upstream prediction')
        g = ev.copy()
        g['parent_prediction'] = prediction
        g['residual'] = g.y_true.to_numpy(np.float64)-prediction
        g['replay_block'] = index
        ks = parent['parent'] if upstream == 'C' else parent['parent']['parent']
        comp,_ = pipe.features(ks,ev)
        for column,ch in enumerate(channels):
            g['up_'+ch] = comp[:,column]
        if upstream == 'W':
            g['up_w'] = prediction-pipe.emit(parent['parent'],ev)
        if not np.isfinite(g[['residual', *[c for c in g if c.startswith('up_')]]].to_numpy(float)).all():
            raise NumericallyInvalid('Invalid full-development residual/overlap coordinates')
        provenance.append(dict(block=index,role='FINAL_DEVELOPMENT_REPLAY_ROWS',
            fit_n=len(fit),replay_n=len(ev),fit_support_hash=support_id_hash(fit),
            replay_support_hash=support_id_hash(ev),fit_last_partition_position=int(fi[-1]),
            replay_first_partition_position=int(ei[0]),
            fit_dependency_stop_exclusive=int(fit.dependency_stop_exclusive.max()),
            evaluation_dependency_start=int(ev.dependency_start.min()),
            upstream_model_id=parent['deployment_model_id'],upstream_policy_id=parent['admission_policy_id'],
            prediction_hash=prediction_hash(prediction),residual_hash=prediction_hash(g.residual),
            target_used_to_fit_own_predictor=False,full_partition_deployment_parent_used=False,
            raw_purge_preserved=True,strictly_past_partition_rows=True))
        sources.append(g)
    source = pd.concat(sources,ignore_index=True)
    if len(source) != coverage['planned_replay_rows'] or support_id_hash(source) != coverage['planned_support_hash']:
        raise ProtocolSupportError('Full-development replay failed coverage contract')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp.parquet')
    source.to_parquet(temporary,index=False)
    temporary.replace(path)
    audit = dict(binding=binding,coverage=coverage,provenance=provenance,
        source_frame_hash=frame_hash(source),file_sha256=file_sha(path),
        status='COMPLETED',numerical_and_domain_execution='UPSTREAM_LEARN_EXECUTED',
        target_used_to_fit_own_predictor=False,uses_in_sample_parent_residual=False)
    dump(metadata,audit)
    return source,audit,str(path)
