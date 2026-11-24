"""Crash-safe, hash-validated reuse of identical computations only."""
import json
from pathlib import Path
from .invariants import ProtocolError,digest,dump,file_sha


class Checkpoints:
    def __init__(self,root,binding,*,resume=False):
        self.root=Path(root);self.binding=json.loads(json.dumps(binding,default=str))
        self.hits=0;self.misses=0
        manifest=self.root/'RUN_BINDING.json'
        if self.root.exists():
            if not resume:raise ProtocolError('Existing execution root requires explicit resume')
            if not manifest.exists() or json.loads(manifest.read_text())!=self.binding:
                raise ProtocolError('Resume release/data/environment/mode binding changed')
        else:dump(manifest,self.binding)

    def path(self,role,key):return self.root/'checkpoints'/role/(key+'.json')

    def load(self,role,key):
        path=self.path(role,key)
        if not path.exists():self.misses+=1;return None
        sealed=json.loads(path.read_text())
        if sealed['binding']!=self.binding or sealed['role']!=role or sealed['key']!=key:
            raise ProtocolError('Checkpoint identity mismatch')
        if digest(sealed['value'])!=sealed['value_hash']:raise ProtocolError('Checkpoint content changed')
        for name,sha in sealed['artifact_sha256'].items():
            if file_sha(name)!=sha:raise ProtocolError('Checkpoint artifact changed: '+name)
        self.hits+=1
        return sealed['value']

    def save(self,role,key,value,artifacts=()):
        body=dict(binding=self.binding,role=role,key=key,value=value,
            value_hash=digest(value),artifact_sha256={str(Path(p).resolve()):file_sha(p) for p in artifacts})
        dump(self.path(role,key),body)
