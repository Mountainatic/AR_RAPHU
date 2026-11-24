"""Input-only row-addressed lag primitives shared by overlapping partitions."""
import json,pickle,zlib,os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .invariants import digest,dump,file_sha,ProtocolError

@contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        else:
            import msvcrt
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
        try:yield
        finally:
            if os.name=='posix':fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
            else:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

def materialize(pipe,T,channel,profile,mt):
    with locked(Path(pipe.root)/'lag_cache'/channel/'.materialization.lock'):
        return _materialize_locked(pipe,T,channel,profile,mt)

def _materialize_locked(pipe,T,channel,profile,mt):
    from .feature_cache import input_frame_hash,_write_payload
    binding=dict(policy=pipe.policy_id,channel=channel,profile=profile,m_tau=int(mt),
        source=digest(pipe.context.get('C1_manifest')),
        registry=pipe.registration['canonical_protocol_sha256']['K_MODEL_REGISTRY.json'],
        authority=digest(pipe.registration['authority_bindings_sha256_lf']))
    key=digest(binding);root=Path(pipe.root)/'lag_cache'/channel
    path=root/(key+'.pickle');meta=path.with_suffix('.json')
    cache=getattr(pipe,'_lag_row_materializations',None)
    if cache is None:cache={};pipe._lag_row_materializations=cache
    versions=getattr(pipe,'_lag_row_versions',None)
    if versions is None:versions={};pipe._lag_row_versions=versions
    # Content version, never filesystem timestamps: the data volume may expose
    # coarse timestamps even though Python's stat API returns nanoseconds.
    receipt=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else None
    stamp=None if receipt is None else receipt['sha256']
    if key not in cache or versions.get(key)!=stamp:
        if meta.exists():
            if receipt['binding']!=json.loads(json.dumps(binding)) or receipt['sha256']!=file_sha(path):
                raise ProtocolError('Row-addressed lag cache binding/hash drift')
            cache[key]=pickle.loads(zlib.decompress(path.read_bytes()))
        else:cache[key]=dict(entities={},intervals=None,rows_constructed=0)
        versions[key]=stamp
    data=cache[key];entities=T.entity_id.astype(str).to_numpy();origins=T.origin.to_numpy(np.int64)
    missing=[]
    for entity in dict.fromkeys(entities):
        positions=np.flatnonzero(entities==entity);entry=data['entities'].get(entity)
        if entry is None:missing.extend(positions);continue
        stored=entry['origins'];indices=np.searchsorted(stored,origins[positions])
        safe=np.minimum(indices,max(len(stored)-1,0))
        present=(indices<len(stored)) & (stored[safe]==origins[positions])
        missing.extend(positions[~present])
    if missing or data['intervals'] is None:
        missing=np.asarray(sorted(missing),dtype=np.int64)
        values,intervals=pipe.v2_k.profile_values(pipe.accessor,T.iloc[missing],channel,profile,int(mt))
        values=np.asarray(values,dtype=np.float64)
        if data['intervals'] is not None and list(map(tuple,data['intervals']))!=list(map(tuple,intervals)):
            raise ProtocolError('Frozen lag interval mapping changed')
        data['intervals']=intervals
        for entity in dict.fromkeys(entities[missing]):
            local=np.flatnonzero(entities[missing]==entity);rows=missing[local]
            new_origins=origins[rows];new_values=values[local]
            entry=data['entities'].get(entity)
            if entry is not None:
                new_origins=np.concatenate((entry['origins'],new_origins))
                new_values=np.concatenate((entry['values'],new_values))
            order=np.argsort(new_origins,kind='stable')
            if len(np.unique(new_origins))!=len(new_origins):raise ProtocolError('Duplicate immutable lag source row')
            data['entities'][entity]=dict(origins=new_origins[order],values=new_values[order])
        data['rows_constructed']+=len(missing)
        _write_payload(path,data)
        dump(meta,dict(binding=binding,sha256=file_sha(path),target_independent=True,contains_winner=False,
            rows_constructed=data['rows_constructed'],format='UNIQUE_ENTITY_ORIGIN_INPUT_ROWS'))
        versions[key]=file_sha(path)
    output=np.empty((len(T),len(data['intervals'])),dtype=np.float64)
    for entity in dict.fromkeys(entities):
        positions=np.flatnonzero(entities==entity);entry=data['entities'][entity]
        output[positions]=entry['values'][np.searchsorted(entry['origins'],origins[positions])]
    request=dict(input_frame=input_frame_hash(T),primitive_key=key,policy=pipe.policy_id)
    receipt=root/'partitions'/(digest(request)+'.json')
    if not receipt.exists():dump(receipt,dict(binding=request,target_independent=True))
    return output,data['intervals']
