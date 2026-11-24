"""Deny held-out parquet bytes through Python and Arrow reader entry points."""
import sys,os
from pathlib import Path
from .invariants import ProtocolError,digest

def forbidden(path):
    if isinstance(path,(str,bytes,os.PathLike)):
        path=os.fsdecode(path).replace('\\','/').lower()
        return path.endswith('/validation.parquet') or path.endswith('/test.parquet')
    return False

class DevelopmentReadGuard:
    def __init__(self):self.denied=[];self.reads=[];self.installed=False
    def check(self,path):
        if forbidden(path):
            self.denied.append(str(path));raise ProtocolError('Held-out validation/test bytes are forbidden in this revision')
        if isinstance(path,(str,bytes,os.PathLike)) and str(path).endswith('.parquet'):self.reads.append(str(path))
    def install(self):
        import pandas as pd
        import pyarrow.parquet as pq
        guard=self
        def audit(event,args):
            if event=='open' and args:guard.check(args[0])
        sys.addaudithook(audit)
        base=pq.ParquetFile
        class GuardedParquetFile(base):
            def __init__(self,source,*args,**kw):
                guard.check(source);super().__init__(source,*args,**kw)
        pq.ParquetFile=GuardedParquetFile
        for module,name in ((pd,'read_parquet'),(pq,'read_table')):
            original=getattr(module,name)
            def wrap(*args,_fn=original,**kw):
                if args:guard.check(args[0])
                elif 'source' in kw:guard.check(kw['source'])
                elif 'path' in kw:guard.check(kw['path'])
                return _fn(*args,**kw)
            setattr(module,name,wrap)
        self.installed=True;return self
    def report(self):
        return dict(installed=self.installed,held_out_bytes_unread=True,blocked_attempts=self.denied,
            development_parquet_reads=len(self.reads),read_paths_hash=digest(self.reads))
