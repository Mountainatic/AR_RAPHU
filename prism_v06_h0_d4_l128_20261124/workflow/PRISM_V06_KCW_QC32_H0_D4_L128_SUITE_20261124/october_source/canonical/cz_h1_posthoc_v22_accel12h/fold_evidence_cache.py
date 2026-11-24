"""Permanent sealed evidence on disk; bounded per-task hot working set."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np
from .invariants import ProtocolError,NumericallyInvalid,digest,frame_hash,prediction_hash
from .risk import RiskEvidence
from .candidate_status import SupportLimited,DomainNotApplicable

ERRORS=(RuntimeError,ValueError,np.linalg.LinAlgError)
KINDS={c.__name__:c for c in (RuntimeError,ValueError,np.linalg.LinAlgError,SupportLimited,DomainNotApplicable,NumericallyInvalid)}

def fold_binding(pipe,channel,profile,mt,fit,score,x,z):
    def array(v):
        v=np.asarray(v)
        if v.dtype!=np.float64:raise ProtocolError('Fold cache requires original FP64 values')
        return dict(shape=list(v.shape),sha256=prediction_hash(v))
    return dict(schema='EXACT_K_FOLD_EVIDENCE_V1',channel=channel,profile=profile,m_tau=mt,
        fit_frame_hash=frame_hash(fit),score_frame_hash=frame_hash(score),
        fit_values=array(x),score_values=array(z),fit_target=array(fit.y_true.to_numpy(float)),
        score_target=array(score.y_true.to_numpy(float)),
        authority=digest(pipe.registration['authority_bindings_sha256_lf']),
        canonical_sources=digest(pipe.registration['canonical_source_sha256_lf']),
        environment=digest(pipe.registration['environment_fingerprint']),
        registry=pipe.K_registry['payload_sha256'],v2_config=digest(pipe.v2),ALS=digest(pipe.v2_k._als_kwargs(pipe.v2)))

class FoldEvidenceCache:
    def __init__(self,path,budget):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS groups (key TEXT PRIMARY KEY,binding TEXT NOT NULL,payload BLOB NOT NULL,sha TEXT NOT NULL,size INTEGER NOT NULL,tick INTEGER NOT NULL)')
        self.budget=int(budget);self.tick=int(self.db.execute('SELECT COALESCE(MAX(tick),0) FROM groups').fetchone()[0])
        self.groups={};self.dirty=set();self.hits=0;self.misses=0;self.evictions=0

    def _group(self,binding):
        key=digest(binding)
        if key not in self.groups:
            row=self.db.execute('SELECT binding,payload,sha FROM groups WHERE key=?',(key,)).fetchone()
            if row:
                encoded,payload,sha=row
                if digest(json.loads(encoded))!=key or hashlib.sha256(payload).hexdigest()!=sha:
                    raise ProtocolError('Fold evidence cache content/binding changed')
                entries=json.loads(zlib.decompress(payload))
            else:entries={}
            self.groups[key]=(json.loads(json.dumps(binding,default=str)),entries)
        return key,self.groups[key][1]

    def evaluate(self,binding,spec,compute):
        key,entries=self._group(binding);recipe=digest(spec)
        if recipe in entries:
            body=entries[recipe];self.hits+=1
        else:
            self.misses+=1
            try:
                value=compute()
                body=dict(kind='RISK',value=value.value,bound=value.bound,rows=value.rows,
                    ordered_score_support_hash=value.ordered_score_support_hash)
            except ProtocolError:raise
            except ERRORS as exc:
                kind=type(exc).__name__
                if kind not in KINDS:raise
                body=dict(kind='ERROR',error_type=kind,error=str(exc))
            entries[recipe]=body;self.dirty.add(key)
        if body['kind']=='ERROR':
            if body['error_type'] not in KINDS:raise ProtocolError('Unknown cached fold failure')
            raise KINDS[body['error_type']](body['error'])
        if body['kind']!='RISK':raise ProtocolError('Unknown cached fold evidence')
        return RiskEvidence(body['value'],body['bound'],body['rows'],body['ordered_score_support_hash'])

    def flush(self):
        with self.db:
            for key,(binding,entries) in self.groups.items():
                self.tick+=1
                if key in self.dirty:
                    encoded=json.dumps(binding,sort_keys=True,separators=(',',':'),default=str)
                    payload=zlib.compress(json.dumps(entries,separators=(',',':')).encode(),3)
                    self.db.execute('INSERT OR REPLACE INTO groups VALUES (?,?,?,?,?,?)',
                        (key,encoded,payload,hashlib.sha256(payload).hexdigest(),len(encoded.encode())+len(payload),self.tick))
                else:self.db.execute('UPDATE groups SET tick=? WHERE key=?',(self.tick,key))
            # Sealed evidence is a DAG artifact, not an evictable recomputation
            # cache. Clearing only the hot groups below preserves compute-once
            # even when the persistent store grows beyond the hot RAM budget.
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.groups.clear();self.dirty.clear()

    def snapshot(self):return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,path=str(self.path),
        persistent_evidence_never_evicted=True,hot_budget_bytes=self.budget)
    def close(self):self.flush();self.db.close()
