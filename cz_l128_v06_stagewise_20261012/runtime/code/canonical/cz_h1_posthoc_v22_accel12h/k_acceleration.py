"""Persistent per-channel spawn processes; exact registration-order collection."""
import os,signal,time,ctypes,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .invariants import ProtocolError,dump
from .fold_evidence_cache import FoldEvidenceCache

_PIPE=None
_CACHES={}

def _initialize(context,root,mode):
    global _PIPE
    # Only process-level execution is parallel. Every numerical kernel is single-threaded.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ProtocolError('Parallel K worker changed thread environment')
    if os.name=='posix':
        parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent:os.kill(os.getpid(),signal.SIGTERM)
    from .read_guard import DevelopmentReadGuard
    _guard=DevelopmentReadGuard().install()
    from .canonical_engine import CanonicalPipeline
    _PIPE=CanonicalPipeline(dict(context,resume=True,k_worker=True),root,mode=mode)
    _PIPE._development_read_guard=_guard
    from .runtime_seal import verify_release
    verify_release(Path(__file__).parent,context['authority'],Path(context['release_dir'])/'REGISTRATION.json',
        Path(context['release_dir'])/'RELEASE_SEAL.json',context['release_sha'])

def _task(T,destination,channel,replay_identity=None,reporting_only=False):
    if reporting_only:
        from .reporting_exhaustive_channel import channel as run_channel
    else:
        from .k_channel import channel as run_channel
    pipe=_PIPE;pipe.artifacts=[];pipe.fit_calls=0
    from .hierarchical_search import IdentityBinding
    pipe.replay_session=IdentityBinding(replay_identity) if replay_identity is not None else None
    settings=pipe.protocol['execution_acceleration']
    cache_root=Path(pipe.context.get('acceleration_cache_root',pipe.root/'accel_cache/K'))
    if channel not in _CACHES:
        _CACHES[channel]=FoldEvidenceCache(cache_root/(channel+'.sqlite3'),settings['fold_cache_payload_bytes_per_channel'])
    cache=_CACHES[channel];before=cache.snapshot();started=time.monotonic()
    legacy=getattr(pipe,'_legacy_candidates',None)
    old_imports=legacy.hits if legacy else 0
    if legacy:legacy.begin(T,channel)
    try:
        output=run_channel(pipe,T,Path(destination),channel,cache)
        cache.flush()
    except BaseException:
        # Previously sealed entries survive; this task's uncommitted memo entries may be recomputed.
        cache.groups.clear();cache.dirty.clear();raise
    after=cache.snapshot()
    output.update(artifacts=pipe.artifacts,numerical_fit_calls=pipe.fit_calls,
        execution=dict(channel=channel,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
            cache_hits=after['hits']-before['hits'],cache_misses=after['misses']-before['misses'],cache_evictions=after['evictions']-before['evictions']))
    output['execution']['legacy_candidate_imports']=legacy.hits-old_imports if legacy else 0
    output['execution']['read_guard']=pipe._development_read_guard.report()
    output['execution']['fixed_recipe_fit_cache']={name:cache.snapshot() for name,cache in getattr(pipe,'_fixed_recipe_fit_caches',{}).items()}
    return output

def tune_k_parallel(pipe,T,destination,*,reporting_only=False):
    from .canonical_engine import CHANNELS
    from .k_assembly import assemble
    settings=pipe.protocol['execution_acceleration']
    if reporting_only and pipe.mode!='accelerated_benchmark':raise ProtocolError('Exhaustive K is reporting-only development audit')
    if settings['channel_processes']!=4 or settings['multiprocessing_start_method']!='spawn':
        raise ProtocolError('Unregistered K concurrency')
    if not hasattr(pipe,'_k_executor'):
        pipe._k_executor=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize,initargs=(pipe.context,pipe.root,pipe.mode))
    started=time.monotonic()
    futures=[pipe._k_executor.submit(_task,T,str(destination),channel,None if pipe.replay_session is None else pipe.replay_session.identity(),reporting_only) for channel in CHANNELS]
    outputs=[future.result() for future in futures]
    for output in outputs:
        pipe.fit_calls+=output['numerical_fit_calls'];pipe.artifacts.extend(output['artifacts'])
    model=assemble(pipe,T,destination,outputs)
    dump(Path(destination)/'EXECUTION_ACCELERATION.json',dict(mode=pipe.mode,elapsed_seconds=time.monotonic()-started,
        numerical_fit_calls=sum(o['numerical_fit_calls'] for o in outputs),channels=[o['execution'] for o in outputs],
        collection_order=CHANNELS,scientific_protocol_changed=True))
    return model

def close_workers(pipe,*,interrupt=False):
    executor=getattr(pipe,'_k_executor',None)
    if executor is None:return
    if interrupt:
        for process in list((executor._processes or {}).values()):
            if process.is_alive():process.terminate()
        for process in list((executor._processes or {}).values()):process.join(timeout=5)
    executor.shutdown(wait=not interrupt,cancel_futures=interrupt)
    del pipe._k_executor
