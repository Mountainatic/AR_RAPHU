"""Build and replay the public source using only generated toy models and inputs."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import pickle
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from benchmark import FLAGS, POINTER, ROOT, header, sha
import numpy as np

from deploy_history import load_predictor
from native_predictor import NativePredictor
from stable_spline import StableLinearSpline


def fixture(family):
    rng = np.random.default_rng(917)
    sru = family == 'QC32_RLS'
    L, delay = (16, 8) if sru else (128, 4)
    bins = None if sru else list(zip([0,1,2,4,8,16,32,64], [1,2,4,8,16,32,64,128]))
    spline = family != 'EX_RLS'
    dim = {'QC32_RLS': 13, 'QD8_RLS': 7, 'EX_RLS': 23}[family]
    width = 2 + (2 if sru else 8) * (10 if spline else 5)
    parent = dict(
        config=dict(kind='URYSOHN' if spline else 'LINEAR', own=True, memory=True),
        recipe=dict(spline=StableLinearSpline(n_knots=3, degree=1, include_bias=False,
                     extrapolation='linear').fit(rng.normal(size=(200,5)))) if spline else None,
        ridge=dict(mean=rng.normal(size=width), scale=rng.uniform(.5,2,size=width),
                   ym=.1, ys=.2, beta=rng.normal(0,.08,size=width)),
        coordinate=dict(kind='asinh' if sru else 'signed_log1p', scale=.5 if sru else 1.),
    )
    a = dict(selected=dict(family=family,p=1,forget=.995,gain=1.), admitted=True,
             model=dict(theta=rng.normal(0,.01,size=dim).tolist(), P=(np.eye(dim)*.01).tolist(),
                        mean=[0.]*(dim-1), scale=[1.]*(dim-1), ym=0., ys=1., fit_last_target_end=1))
    config = dict(model=dict(upper=dict(parent=parent,W=None),A=a), L=L, delay=delay,
                  parent_fit_last_origin=0, delta_target=family=='EX_RLS', temporal_bins=bins,
                  target_history_mode='endpoints')
    linear = {}
    for name in ('AR','ARX'):
        xlags = list(range(L)) if name=='ARX' else []
        n = L+len(xlags)*5
        mean, scale, coef = rng.normal(size=n), rng.uniform(.5,2,size=n), rng.normal(0,.01,size=n)
        pipeline = (SimpleNamespace(mean_=mean,scale_=scale),SimpleNamespace(coef_=coef,intercept_=.02))
        linear[name] = dict(model=pipeline,ym=.1,ys=.3,selected=dict(config=dict(Ly=L,Xlags=xlags),target='delta'))
    return config, linear, rng.normal(size=(L+63+768,5)), rng.normal(size=L+63+768)


def run_case(folder, family, flags):
    config, linear, x, y = fixture(family)
    path = folder/'synthetic.pkl'
    path.write_bytes(pickle.dumps(config,protocol=5))
    L = config['L']
    start = L+63
    reference = load_predictor(path,x[:start],y[:start],first_origin=1000)
    (folder/'model_constants.h').write_text(header(config,linear,reference.spline),encoding='ascii')
    binary = folder/'stream.so'
    subprocess.run(['gcc',*flags,'-shared','-fPIC','-I',str(folder),str(ROOT/'native_stream.c'),'-lm','-o',str(binary)],check=True)
    (folder/'MODEL.json').write_text(json.dumps(dict(history=L,prism_source_sha256=sha(path),library_sha256=sha(binary))))
    native = NativePredictor(folder,ROOT,path,x[:start],y[:start],first_origin=1000)
    expected = np.array([reference.step(a,b) for a,b in zip(x[start:],y[start:])])
    actual = np.array([native.step(a,b) for a,b in zip(x[start:],y[start:])])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    lib = ct.CDLL(str(binary))
    lib.run.argtypes = [ct.c_int]+[POINTER]*5+[ct.c_size_t,ct.c_size_t,POINTER]
    lib.run.restype = ct.c_double
    result = np.empty(len(actual))
    q = np.zeros(64)
    arrays = [x[start-L+1:start], y[start-L+1:start], q, x[start:], y[start:]]
    ptr = lambda a: a.ctypes.data_as(POINTER)
    lags = np.arange(L)
    for index,name in enumerate(('AR','ARX'),1):
        item = linear[name]
        scaler,ridge = item['model']
        t = np.arange(start,len(x))
        design = y[t[:,None]-lags]
        if index==2:
            design = np.c_[design,x[t[:,None]-lags].reshape(len(t),-1)]
        expected_linear = item['ym'] + item['ys']*((design-scaler.mean_)/scaler.scale_ @ ridge.coef_ + ridge.intercept_) + y[t]
        lib.run(index,*[ptr(a) for a in arrays],len(actual),64,ptr(result))
        np.testing.assert_allclose(result,expected_linear,atol=1e-12,rtol=1e-12)
    print(f'{family}: 768 steps verified, including AR and ARX',flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--portable',action='store_true',help='Build without host-specific SIMD flags')
    args = cli.parse_args()
    flags = [f for f in FLAGS if not f.startswith(('-march=','-mprefer-vector-width='))] if args.portable else FLAGS
    with TemporaryDirectory(prefix='prism_native_smoke_') as temp:
        for family in ('QC32_RLS','QD8_RLS','EX_RLS'):
            folder = Path(temp)/family
            folder.mkdir()
            run_case(folder,family,flags)
