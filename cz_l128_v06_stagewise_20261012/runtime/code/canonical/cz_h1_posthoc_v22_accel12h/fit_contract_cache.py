"""Sealed fixed-recipe fits, independent of the downstream scoring support.

This cache is target-dependent and separate from input-only basis/lag caches.
It stores a fixed mathematical fit, never a selected temporal/MODEL winner.
"""
import json,sqlite3,zlib,hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .invariants import digest,ProtocolError
from .fold_evidence_cache import KINDS,ERRORS

class FixedRecipeFits:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS fits (key TEXT PRIMARY KEY,binding TEXT NOT NULL,payload BLOB NOT NULL,sha TEXT NOT NULL)')
        self.hot=OrderedDict();self.hits=0;self.misses=0
    def evaluate(self,binding,compute):
        key=digest(binding)
        if key in self.hot:
            body=self.hot.pop(key);self.hot[key]=body;self.hits+=1
        else:
            row=self.db.execute('SELECT binding,payload,sha FROM fits WHERE key=?',(key,)).fetchone()
            if row:
                encoded,payload,sha=row
                if digest(json.loads(encoded))!=key or hashlib.sha256(payload).hexdigest()!=sha:
                    raise ProtocolError('Fixed-recipe numerical fit cache binding/hash drift')
                body=json.loads(zlib.decompress(payload));self.hits+=1
            else:
                self.misses+=1
                try:body=dict(kind='CONTRACT',contract=compute())
                except ProtocolError:raise
                except ERRORS as exc:
                    kind=type(exc).__name__
                    if kind not in KINDS:raise
                    body=dict(kind='ERROR',error_type=kind,error=str(exc))
                payload=zlib.compress(json.dumps(body,separators=(',',':')).encode(),3)
                with self.db:self.db.execute('INSERT INTO fits VALUES (?,?,?,?)',
                    (key,json.dumps(binding,separators=(',',':'),sort_keys=True),payload,hashlib.sha256(payload).hexdigest()))
            self.hot[key]=body
            while len(self.hot)>256:self.hot.popitem(last=False)
        if body['kind']=='ERROR':raise KINDS[body['error_type']](body['error'])
        # Separate caller ownership prevents mutation of a hot sealed contract.
        return json.loads(json.dumps(body['contract']))
    def snapshot(self):return dict(hits=self.hits,misses=self.misses,target_independent=False,
        contains_MODEL_winner=False,score_support_in_fit_identity=False,persistent_fits_never_evicted=True)

def fit_fixed_recipe(pipe,prepared,family,m_x,lambdas,**kwargs):
    def compute():
        pipe.fit_calls+=1
        return pipe.u.fit_prepared_contract(prepared,family,m_x,lambdas,**kwargs)
    identity=getattr(prepared,'_accel_target_statistics_key',None)
    if identity is None:return compute()  # Explicit unit-test doubles only.
    channel=prepared._accel_channel_name
    caches=getattr(pipe,'_fixed_recipe_fit_caches',None)
    if caches is None:caches={};pipe._fixed_recipe_fit_caches=caches
    if channel not in caches:caches[channel]=FixedRecipeFits(Path(pipe.root)/'fixed_recipe_fit_cache'/(channel+'.sqlite3'))
    binding=dict(target_statistics_key=identity,family=family,m_x=int(m_x),
        lambdas=tuple(map(float,lambdas)),authority_ALS_parameters=kwargs,
        policy=pipe.policy_id,role='FIXED_RECIPE_TARGET_BOUND_AUTHORITY_FIT')
    return caches[channel].evaluate(binding,compute)
