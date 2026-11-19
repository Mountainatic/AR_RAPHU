"""One-sample interface to a compiled frozen model on Linux."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DOUBLE_POINTER = ct.POINTER(ct.c_double)


def pointer(array):
    return array.ctypes.data_as(DOUBLE_POINTER)


class NativePredictor:
    def __init__(self, artifact, profile_dir, model_path, process_history, mature_target_history, first_origin):
        artifact, profile_dir, model_path = map(Path, (artifact, profile_dir, model_path))
        metadata = json.loads((artifact/'MODEL.json').read_text())
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata['prism_source_sha256']:
            raise ValueError('Model does not match the compiled library')
        binary = artifact/'stream.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest() != metadata['library_sha256']:
            raise ValueError('Compiled library does not match the benchmark artifact')
        sys.path.insert(0, str(profile_dir.resolve()))
        try:
            from deploy_history import load_predictor

            reference = load_predictor(model_path, process_history, mature_target_history, first_origin)
        finally:
            sys.path.pop(0)
        L = metadata['history']
        xp = np.ascontiguousarray(process_history[-(L-1):],dtype=np.float64)
        yp = np.ascontiguousarray(mature_target_history[-(L-1):],dtype=np.float64)
        q = np.empty(64, dtype=np.float64)
        for i, value in reference.q.items():
            q[i%64] = value
        if len(reference.q) != 64:
            raise ValueError('Incomplete issued-response warmup')
        lib = ct.CDLL(str(binary.resolve()))
        lib.prism_state_bytes.argtypes = []
        lib.prism_state_bytes.restype = ct.c_size_t
        lib.init_prism.argtypes = [ct.c_void_p,DOUBLE_POINTER,DOUBLE_POINTER,DOUBLE_POINTER]
        lib.init_prism.restype = None
        lib.prism_step.argtypes = [ct.c_void_p,DOUBLE_POINTER,ct.c_double]
        lib.prism_step.restype = ct.c_double
        self._lib = lib
        self._state = (ct.c_double*((lib.prism_state_bytes()+7)//8))()
        lib.init_prism(self._state,pointer(xp),pointer(yp),pointer(q))

    def step(self, x_current, y_newly_mature):
        row = np.ascontiguousarray(x_current,dtype=np.float64)
        if row.shape != (5,) or not np.isfinite(row).all() or not np.isfinite(y_newly_mature):
            raise ValueError('Provide five finite process inputs and one finite mature target')
        value = self._lib.prism_step(self._state,pointer(row),y_newly_mature)
        if not np.isfinite(value):
            raise FloatingPointError('Invalid RLS state or prediction')
        return value
