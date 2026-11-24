"""Shared audit primitives without data or model side effects."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


class ProtocolError(RuntimeError):
    pass


class NumericallyInvalid(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def dump(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    temporary.replace(path)


def frame_hash(frame):
    schema=digest([(str(c),str(frame[c].dtype)) for c in frame])
    values=pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype='<u8')
    return hashlib.sha256(schema.encode()+values.tobytes()).hexdigest()


def prediction_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<f8').tobytes()).hexdigest()
