"""Sealed input-only basis materialization, separate from target statistics."""
import json,pickle,hashlib,zlib
from pathlib import Path
import numpy as np
from .invariants import digest,frame_hash,file_sha,dump,ProtocolError
from .candidate_status import require_k_support

def input_frame_hash(frame):
    columns=[c for c in ('entity_id','origin','split','sample_id','view_sample_id','latest_available_target_index',
        'target_stop_exclusive','dependency_start','dependency_stop_exclusive','causal_history_floor',
        'anchor_history_steps','sample_support_contract') if c in frame]
    return frame_hash(frame[columns])

def profile_values(pipe,T,channel,profile,mt):
    # Partition receipts bind each request's input frame. The reusable primitive
    # is one immutable source row, rather than another dense copy of its prefix.
    from .lag_row_cache import materialize
    return materialize(pipe,T,channel,profile,mt)

class DesignShape:
    """Shape witness for authority's exact sufficient-statistics-only path.

    It deliberately has no array conversion/indexing: an accidental fallback
    to a design-based estimator must fail, never silently use surrogate data.
    """
    def __init__(self,shape):self.shape=tuple(shape)
    def __len__(self):return self.shape[0]

def _write_payload(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(zlib.compress(pickle.dumps(value,protocol=5),3));temporary.replace(path)

def _target_free(stat):
    return dict(gram=stat.gram,feature_sum=stat.feature_sum,rows=stat.rows)

def predict_materialized(pipe,design,contract):
    """Preserve authority predict_contract's full einsum, including broadcast.

    Do not substitute predict_contract_from_design: that API slices amplitude
    columns, whereas the accepted batch predictor broadcasts a linear theta
    across a realized low-cardinality basis. This legacy behavior is frozen.
    """
    if contract['family']=='EXACT_ZERO':return np.zeros(len(design),dtype=np.float64)
    theta=np.asarray(contract['theta'],dtype=np.float64);result=np.empty(len(design),dtype=np.float64)
    for start in range(0,len(design),pipe.u.STREAM_CHUNK_ROWS):
        stop=min(start+pipe.u.STREAM_CHUNK_ROWS,len(design))
        result[start:stop]=np.einsum('tbx,bx->t',design[start:stop],theta)+float(contract['intercept'])
    return result

def predict_cached_score(pipe,prepared,values,contract,score):
    if prepared.basis.metadata()!=contract['basis']:
        raise ProtocolError('Scoring design does not use the fitted contract basis')
    values=np.asarray(values,dtype=np.float64)
    binding=dict(policy=pipe.policy_id,score_input_frame=input_frame_hash(score),
        basis=prepared.basis.metadata(),shape=values.shape,
        values_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    key=digest(binding);cache=getattr(pipe,'_score_design_materializations',None)
    if cache is None:cache={};pipe._score_design_materializations=cache
    if key not in cache:
        cache[key]=pipe.u.tensor_design(values,prepared.basis)
    return predict_materialized(pipe,cache[key],contract)

def prepare(pipe,*,channel_name,profile,mt,fit,values,width,family):
    values=np.asarray(values,dtype=np.float64)
    binding=dict(policy=pipe.policy_id,frame=input_frame_hash(fit),channel=channel_name,
        profile=profile,m_tau=mt,amplitude_width=int(width),input_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        registry=pipe.registration['canonical_protocol_sha256']['K_MODEL_REGISTRY.json'],
        authority=digest(pipe.registration['authority_bindings_sha256_lf']))
    key=digest(binding);root=Path(pipe.root)/'feature_cache'/channel_name
    path=root/(key+'.pickle');meta=path.with_suffix('.json')
    phi=None
    if meta.exists():
        sealed=json.loads(meta.read_text(encoding='utf-8'))
        if sealed['binding']!=json.loads(json.dumps(binding)) or file_sha(path)!=sealed['sha256']:
            raise ProtocolError('Input feature cache binding/hash drift')
        payload=pickle.loads(zlib.decompress(path.read_bytes()))
        basis=payload['basis'];shape=tuple(payload['shape']);hit=True
    else:
        basis=pipe.u.AmplitudeBasis.fit(values,int(width))
        shape=(len(values),values.shape[1],basis.dimension)
        hit=False
    dims=shape[1:];rank=int(family.split('_')[1]) if family.startswith('RANK_') else None
    count=1+(rank*sum(dims) if rank is not None else int(np.prod(dims)))
    # Rows/realized dimensions are checked before target statistics or any solver.
    require_k_support(dict(parameter_count=count,family=family),len(fit),pipe.v2)
    y=fit.y_true.to_numpy(np.float64)
    target_binding=dict(input_key=key,target_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    target_key=digest(target_binding)
    target_path=Path(pipe.root)/'target_statistics_cache'/channel_name/(target_key+'.pickle')
    target_meta=target_path.with_suffix('.json')
    if target_meta.exists():
        receipt=json.loads(target_meta.read_text(encoding='utf-8'))
        if receipt['binding']!=target_binding or receipt['sha256']!=file_sha(target_path):
            raise ProtocolError('Target sufficient-statistics cache binding/hash drift')
        full_stats,linear_stats=pickle.loads(zlib.decompress(target_path.read_bytes()))
    else:
        # Only a different, explicitly hash-bound target requires fresh target
        # statistics. In a frozen run each exact input/target problem is unique.
        if phi is None:phi=pipe.u.tensor_design(values,basis)
        linear=phi[:,:,0] if basis.dimension>0 else phi.reshape(len(phi),0)
        stats=pipe.u.DesignStatistics
        full_stats=stats.from_design(phi.reshape(len(phi),-1),y)
        linear_stats=stats.from_design(linear,y)
        _write_payload(target_path,(full_stats,linear_stats))
        dump(target_meta,dict(binding=target_binding,sha256=file_sha(target_path),
            target_independent=False,contains_winner=False,role='EXACT_AUTHORITY_BATCH_SUFFICIENT_STATISTICS'))
    if not hit:
        payload=dict(basis=basis,shape=shape,full_input_statistics=_target_free(full_stats),
            linear_input_statistics=_target_free(linear_stats))
        _write_payload(path,payload)
        dump(meta,dict(binding=binding,sha256=file_sha(path),target_independent=True,
            contains_winner=False,format='BASIS_SHAPE_EXACT_GRAM_FEATURE_SUM'))
    else:
        for name,stat in (('full_input_statistics',full_stats),('linear_input_statistics',linear_stats)):
            cached=payload[name]
            if cached['rows']!=stat.rows or not np.array_equal(cached['gram'],stat.gram) or not np.array_equal(cached['feature_sum'],stat.feature_sum):
                raise ProtocolError('Target-independent sufficient statistics changed')
    pipe.feature_cache_hits=getattr(pipe,'feature_cache_hits',0)+int(hit)
    pipe.feature_cache_misses=getattr(pipe,'feature_cache_misses',0)+int(not hit)
    prepared=pipe.u.PreparedContractFit(values,y,int(width),basis,DesignShape(shape),full_stats,linear_stats)
    prepared._accel_target_statistics_key=target_key;prepared._accel_channel_name=channel_name
    return prepared
