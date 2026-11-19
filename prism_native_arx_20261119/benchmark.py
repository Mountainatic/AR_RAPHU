"""Compile and compare the frozen PRISM / AR / ARX streams on one CPU core."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import warnings

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
POINTER = ct.POINTER(ct.c_double)
FLAGS = ['-O3', '-march=native', '-fopenmp-simd', '-fno-math-errno', '-ffp-contract=off']
if platform.machine() == 'x86_64':
    FLAGS.append('-mprefer-vector-width=512')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant(name, value, integer=False):
    arr = np.asarray(value)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f'Invalid constant {name}')

    def emit(a):
        if a.ndim == 0:
            return str(int(a)) if integer else float(a).hex()
        return '{' + ', '.join(emit(i) for i in a) + '}'

    if arr.ndim == 0:
        return f'#define {name} ({emit(arr)})'
    dimensions = ''.join(f'[{d}]' for d in arr.shape)
    dtype = 'int' if integer else 'double'
    return f'static const {dtype} {name}{dimensions} = {emit(arr)};'


def fold_ridge(mean, scale, coefficients, intercept, output_mean, output_scale):
    weights = np.asarray(coefficients)*output_scale/scale
    return weights, float(output_mean+output_scale*intercept-np.dot(mean, weights))


def header(config, linear, spline):
    model = config['model']
    up = model['upper']
    p, a = up['parent'], model['A']
    L, delay = config['L'], config['delay']
    spec = a['selected']
    assert L in (16,128) and delay in (4,8)
    assert up['W'] is None and config['target_history_mode']=='endpoints'
    assert p['config']['own'] and p['config']['memory']
    assert p['config']['kind'] in ('LINEAR','URYSOHN') and p['coordinate']['kind'] in ('asinh','signed_log1p')
    assert spec['p']==1 and 'innovation_clip' not in spec
    family = spec['family']
    assert family in ('QC32_RLS','QD8_RLS','EX_RLS')
    dim = len(a['model']['theta'])
    assert dim == {'QC32_RLS':13,'QD8_RLS':7,'EX_RLS':23}[family]
    bins = config['temporal_bins'] or [(0,1),(1,L)]
    if family=='EX_RLS':
        assert bins==[(0,1),(1,2),(2,4),(4,8),(8,16),(16,32),(32,64),(64,128)]
    before_bins = spline is not None and config['temporal_bins'] is None
    after_bins = spline is not None and config['temporal_bins'] is not None
    ints = dict(WINDOW=L, DELAY=delay, BIN_COUNT=len(bins), A_DIM=dim,
                BIN_LO=[x[0] for x in bins], BIN_HI=[x[1] for x in bins],
                HISTORY_WIDTH=10 if before_bins else 5, BIN_WIDTH=10 if spline else 5,
                SPLINE_BEFORE_BINS=before_bins, SPLINE_AFTER_BINS=after_bins,
                DRIVER_EX=family=='EX_RLS', CONDITIONAL=family=='QC32_RLS',
                DELTA_TARGET=config['delta_target'], LINK_ASINH=p['coordinate']['kind']=='asinh')
    r = p['ridge']
    weights, intercept = fold_ridge(r['mean'], r['scale'], r['beta'], 0, r['ym'], r['ys'])
    assert len(weights)==2+len(bins)*ints['BIN_WIDTH']
    coeff = a['model']
    numbers = dict(KCW_WEIGHTS=weights, KCW_INTERCEPT=intercept, LINK_SCALE=p['coordinate']['scale'],
                   SPLINE_KNOTS=spline.knots if spline else np.zeros((5,3)),
                   SPLINE_VALUES=spline.values if spline else np.zeros((5,3,2)),
                   SPLINE_SLOPES=spline.slopes if spline else np.zeros((5,2,2)),
                   A_THETA=coeff['theta'], A_COVARIANCE=coeff['P'], A_MEAN=coeff['mean'],
                   A_SCALE=coeff['scale'], A_YM=coeff['ym'], A_YS=coeff['ys'],
                   A_FORGET=spec['forget'], A_GAIN=spec['gain'])
    for name, obj in linear.items():
        scaler, ridge = obj['model']
        spec = obj['selected']
        assert spec['config']['Ly']==L
        assert spec['config']['Xlags']==(list(range(L)) if name=='ARX' else [])
        w, b = fold_ridge(scaler.mean_,scaler.scale_,ridge.coef_,ridge.intercept_,obj['ym'],obj['ys'])
        numbers[name+'_WEIGHTS'], numbers[name+'_INTERCEPT'] = w, b
        ints[name+'_DELTA']=spec['target']=='delta'
    return '\n'.join([constant(k,v,True) for k,v in ints.items()] + [constant(k,v) for k,v in numbers.items()])+'\n'


def measure(args, task):
    entity, delay = task['entity'], task['delay']
    output = args.results/f'{entity}_D{delay}'
    output.mkdir(parents=True, exist_ok=True)
    sru = entity=='SRU_H2S'
    source = args.sru if sru else args.cz/entity
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',InconsistentVersionWarning)
        linear_path = source/'runs'/f'D{delay}'/'linear/linear.pkl'
        assert sha(linear_path)==json.loads((linear_path.parent/'FROZEN.json').read_text())['files']['linear.pkl']
        linear = pickle.loads(linear_path.read_bytes())
    values = np.vstack([np.load(source/'data'/(n+'.npz'))['values'] for n in ('train','development')])
    start, stop = task['origin_start'], task['origin_stop']
    assert len(values)==stop
    model_path = args.profile/f'{entity}_D{delay}.pkl'
    assert sha(model_path)==task['model_sha256']
    sys.path.insert(0,str(args.profile))
    from deploy_history import load_predictor

    config = pickle.loads(model_path.read_bytes())
    L = config['L']
    past_x = np.ascontiguousarray(values[start-L-63:start,1:])
    past_y = np.ascontiguousarray(values[start-delay-L-63:start-delay,0])
    reference = load_predictor(model_path,past_x,past_y,start)
    q = np.empty(64,dtype=np.float64)
    for i, v in reference.q.items(): q[i%64]=v
    init_x = np.ascontiguousarray(past_x[-(L-1):])
    init_y = np.ascontiguousarray(past_y[-(L-1):])
    x = np.ascontiguousarray(values[start:stop,1:])
    y = np.ascontiguousarray(values[start-delay:stop-delay,0])
    header_path = output/'model_constants.h'
    header_path.write_text(header(config,linear,reference.spline),encoding='ascii')
    library_path = output/'stream.so'
    subprocess.run(['gcc',*FLAGS,'-shared','-fPIC','-I',str(output),str(ROOT/'native_stream.c'),'-lm','-o',str(library_path)],check=True)
    lib = ct.CDLL(str(library_path))
    lib.run.argtypes=[ct.c_int]+[POINTER]*5+[ct.c_size_t,ct.c_size_t,POINTER]
    lib.run.restype=ct.c_double
    lib.prism_state_bytes.restype=ct.c_size_t
    predictions = np.empty(len(x),dtype=np.float64)
    ptr = lambda a: a.ctypes.data_as(POINTER)
    inputs = [ptr(a) for a in (init_x,init_y,q,x,y)]

    def run(kind):
        return lib.run(kind,*inputs,len(x),64,ptr(predictions))

    expected = {'PRISM':np.array([reference.step(row,label) for row,label in zip(x,y)])}
    baseline = pd.read_csv(source/'results'/f'point_predictions_D{delay}.csv',float_precision='round_trip')
    np.testing.assert_array_equal(baseline.origin,np.arange(start,stop))
    for name in ('AR','ARX'): expected[name]=baseline[name].to_numpy()
    observed = values[start:stop,0]
    checks = {}
    for kind,name in enumerate(('PRISM','AR','ARX')):
        run(kind)
        np.testing.assert_allclose(predictions,expected[name],atol=1e-12,rtol=1e-12)
        checks[name]=dict(max_abs_error=float(np.max(np.abs(predictions-expected[name]))),
                          RMSE=float(np.sqrt(np.mean((observed-predictions)**2))))
    assert abs(checks['PRISM']['RMSE']-task['RMSE'])<1e-11
    times = [[],[],[]]
    for repeat in range(args.repeats):
        for offset in range(3):
            kind=(offset+repeat)%3
            times[kind].append(float(run(kind)))
    for name,t in zip(('PRISM','AR','ARX'),times):
        checks[name].update(median_ns_per_sample=statistics.median(t),p10_ns=float(np.quantile(t,.1)),
                            p90_ns=float(np.quantile(t,.9)),repeats=len(t),ns_by_replay=t)
    result = dict(entity=entity,delay=delay,history=L,step_count=len(x),warmup=64,measurements=checks,
                A_parameters=len(config['model']['A']['model']['theta']),prism_state_bytes=lib.prism_state_bytes(),
                prism_source_sha256=sha(model_path),linear_source_sha256=sha(linear_path),constants_sha256=sha(header_path),
                PRISM_over_AR=checks['PRISM']['median_ns_per_sample']/checks['AR']['median_ns_per_sample'],
                PRISM_over_ARX=checks['PRISM']['median_ns_per_sample']/checks['ARX']['median_ns_per_sample'])
    metadata = dict(entity=entity, delay=delay, history=L, library_sha256=sha(library_path),
                    prism_source_sha256=sha(model_path))
    (output/'MODEL.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    from native_predictor import NativePredictor

    single = NativePredictor(output,args.profile,model_path,past_x,past_y,start)
    replay = np.array([single.step(row,label) for row,label in zip(x,y)])
    np.testing.assert_allclose(replay,expected['PRISM'],atol=1e-12,rtol=1e-12)
    result['single_step_api_verified'] = True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','sru','cz','results'):
        parser.add_argument('--'+name,required=True,type=lambda s: Path(s).resolve())
    parser.add_argument('--repeats',type=int,default=51)
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    cpus=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{args.cpu if args.cpu is not None else min(cpus)})
    args.results.mkdir(parents=True,exist_ok=True)
    profile=json.loads((args.profile/'PROFILE.json').read_text())
    rows=[]
    for task in profile['rows']:
        row=measure(args,task)
        rows.append(row)
        print(task['entity'],[(n,round(r['median_ns_per_sample'],1)) for n,r in row['measurements'].items()],flush=True)
    cpu=next(s.partition(':')[2].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
    report=dict(cpu=cpu,core=min(os.sched_getaffinity(0)),platform=platform.platform(),flags=FLAGS,
                compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
                dtype='float64',rows=rows,source_sha256=sha(ROOT/'native_stream.c'))
    (args.results/'BENCHMARK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
