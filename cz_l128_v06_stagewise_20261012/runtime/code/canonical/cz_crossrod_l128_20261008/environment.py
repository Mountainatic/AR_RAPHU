"""Numerical runtime identity, recorded before any source-label access."""
import contextlib
import importlib.metadata
import io
import os
import platform
import sys
import numpy as np
from .invariants import ProtocolError,digest

THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def fingerprint():
    import scipy.linalg  # Load every authority LAPACK backend before sealing.
    versions={}
    for name in ('numpy','scipy','pandas','pyarrow','threadpoolctl'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    output=io.StringIO()
    with contextlib.redirect_stdout(output):np.show_config()
    from threadpoolctl import threadpool_info
    pools=[{k:p.get(k) for k in ('user_api','internal_api','prefix','version','threading_layer','architecture','num_threads')}
           for p in threadpool_info() if p.get('user_api')=='blas']
    body=dict(schema='CZ_NUMERICAL_ENVIRONMENT_V1',python=sys.version,
        packages=versions,architecture=platform.machine(),platform=platform.platform(),
        float_dtype=np.dtype(np.float64).str,float_bits=64,
        thread_environment={k:os.environ.get(k) for k in THREAD_KEYS},
        blas=pools,numpy_lapack_configuration=output.getvalue())
    return dict(body,fingerprint_id=digest(body))


def verify_environment(expected,*,exact=False):
    actual=fingerprint()
    for key in ('float_dtype','float_bits','thread_environment','blas'):
        if actual[key]!=expected[key]:raise ProtocolError('Numerical environment changed: '+key)
    if any(v!='1' for v in actual['thread_environment'].values()):
        raise ProtocolError('Registered numerical threads must all equal one')
    if any(p['num_threads']!=1 for p in actual['blas']):
        raise ProtocolError('BLAS runtime thread count is not one')
    if exact and actual!=expected:raise ProtocolError('Resume environment fingerprint changed')
    return actual
