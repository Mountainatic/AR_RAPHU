"""Real development-only policy benchmark; never seals a deployment."""
import json,time,resource,threading,os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from .run_canonical import open_case
from .invariants import dump,digest,frame_hash,prediction_hash,ProtocolError
from .data import SAMPLE_RELATIVE
from .dag_plan import plan
from .read_guard import DevelopmentReadGuard
from .k_acceleration import close_workers

class Timings:
    def __init__(self):self.stack=[];self.nodes=[];self.io_seconds=0.;self.serialization_seconds=0.
    def wrap(self,pipe):
        for name in ('tune','_learn'):
            original=getattr(pipe,name)
            def timed(stage,T,*args,_fn=original,_name=name,**kwargs):
                token=[time.monotonic(),0.];self.stack.append(token)
                try:return _fn(stage,T,*args,**kwargs)
                finally:
                    elapsed=time.monotonic()-token[0];self.stack.pop()
                    if self.stack:self.stack[-1][1]+=elapsed
                    self.nodes.append(dict(role=_name,stage=stage,fit_rows=len(T),
                        seconds=elapsed,exclusive_seconds=max(0,elapsed-token[1])))
            setattr(pipe,name,timed)
        # Refit-only work does not go through tune(); record it separately.
        from .replay_policy import ReplaySession
        original_refit=ReplaySession.refit
        self.original_refit=original_refit
        def timed_refit(session,stage,T,*args,**kw):
            token=[time.monotonic(),0.];self.stack.append(token)
            try:return original_refit(session,stage,T,*args,**kw)
            finally:
                elapsed=time.monotonic()-token[0];self.stack.pop()
                if self.stack:self.stack[-1][1]+=elapsed
                self.nodes.append(dict(role='replay_refit',stage=stage,fit_rows=len(T),
                    seconds=elapsed,exclusive_seconds=max(0,elapsed-token[1])))
        ReplaySession.refit=timed_refit
        for name in ('load','save'):
            original=getattr(pipe.checkpoints,name)
            def timed_io(*args,_fn=original,**kw):
                start=time.monotonic()
                try:return _fn(*args,**kw)
                finally:self.io_seconds+=time.monotonic()-start
            setattr(pipe.checkpoints,name,timed_io)
    def report(self):return dict(nodes=self.nodes,checkpoint_IO_seconds=self.io_seconds)
    def close(self):
        from .replay_policy import ReplaySession
        ReplaySession.refit=self.original_refit

class PeakProcessMemory:
    def __init__(self):self.peak=0;self.stop=threading.Event();self.thread=None
    def start(self):
        def poll():
            while not self.stop.is_set():
                pending=[os.getpid()];total=0;seen=set()
                while pending:
                    pid=pending.pop()
                    if pid in seen:continue
                    seen.add(pid)
                    try:
                        fields=Path(f'/proc/{pid}/statm').read_text().split();total+=int(fields[1])*os.sysconf('SC_PAGE_SIZE')
                        pending.extend(int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
                    except (FileNotFoundError,ProcessLookupError):pass
                self.peak=max(self.peak,total);self.stop.wait(.25)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def close(self):self.stop.set();self.thread.join(timeout=2)

def run(*,authority,c1_root,case,release,output,prefix,release_sha):
    guard=DevelopmentReadGuard().install()
    _,pipe,context=open_case(authority=authority,shared=Path(c1_root)/case,case_id=case,
        release_dir=release,release_sha=release_sha,root=output,mode='accelerated_benchmark')
    if prefix not in pipe.protocol['benchmark']['prefix_rows']:raise ProtocolError('Benchmark prefix was not preregistered')
    metadata=pd.read_parquet(Path(context['shared'])/SAMPLE_RELATIVE/'train.parquet').iloc[:prefix].copy()
    metadata['sample_id']=metadata.view_sample_id.astype(str)
    started=time.monotonic()
    dag=plan(metadata,pipe.protocol,bindings=dict(registries=context['registration']['canonical_protocol_sha256'],
        authority=context['registration']['authority_bindings_sha256_lf'],
        canonical_sources=context['registration']['canonical_source_sha256_lf'],C1=context['C1_manifest']))
    dump(Path(output)/'DAG_PLAN.json',dag)
    preflight_seconds=time.monotonic()-started
    T=pipe.load_development().iloc[:prefix].copy()
    timings=Timings();timings.wrap(pipe)
    memory=PeakProcessMemory().start()
    started=time.monotonic()
    try:
        state=pipe.learn('A',T)
        cold=time.monotonic()-started;fits=pipe.fit_calls
        start=time.monotonic();warm=pipe.learn('A',T);warm_seconds=time.monotonic()-start
        if state!=warm or pipe.fit_calls!=fits:raise ProtocolError('Warm DAG reuse recomputed or changed frozen model')
        stages=[];node=state
        while node is not None:
            stages.append(dict(stage=node['stage'],route=node['routing_status'],domain=node['domain'],
                final_fit_rows=node.get('final_fit_rows'),final_replay_coverage=node.get('final_replay_coverage')))
            node=node.get('parent')
        result=dict(status='COMPLETED',scope='DEVELOPMENT_ONLY_BENCHMARK_NO_DEPLOYMENT',case=case,prefix_rows=len(T),
            cold_seconds=cold,warm_seconds=warm_seconds,preflight_seconds=preflight_seconds,numerical_fit_calls=fits,
            worker_count=pipe.protocol['execution_acceleration']['channel_processes'],
            peak_parent_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_total_process_RAM_bytes=memory.peak,
            stages=stages,timings=timings.report(),read_guard=guard.report(),
            warm_model_hash_equal=digest(state)==digest(warm),release_sha256=release_sha,
            formal_development_training=False,source_validation_reporting=False,target_test_inference=False)
        dump(Path(output)/'RUNTIME_BENCHMARK_RAW.json',result)
        return result
    finally:
        timings.close();memory.close();close_workers(pipe,interrupt=True)
