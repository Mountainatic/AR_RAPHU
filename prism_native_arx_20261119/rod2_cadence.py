"""Train-only selection of lower-frequency RLS updates for the strict E128 Rod2 head."""
import argparse
import itertools
import json
import os
from pathlib import Path
import pickle
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd

ENTITY = 'CZ_Rod2'
DELAY = 4


def predict(m, frame, history, model, spec):
    joined = pd.concat([history, frame], ignore_index=True)
    joined['split_id'] = 'CONTINUOUS'
    x, native, audit = m.mature_a.dynamic_features(joined, spec['p'], spec['family'])
    assert native[len(history):].all()
    assert (audit['latest_feature_target_end'][len(history):] <= frame.latest_available_target_index.to_numpy()).all()
    assert model['fit_last_target_end'] <= frame.latest_available_target_index.iloc[0]
    assert (np.diff(frame.origin.to_numpy()) == 1).all()
    z = np.c_[np.ones(len(frame)), (x[len(history):] - model['mean']) / model['scale']]
    theta, cov = np.array(model['theta']), np.array(model['P'])
    residual = frame.residual.to_numpy()
    result = np.empty(len(frame))
    updates = 0
    for t in range(len(frame)):
        issue = t-DELAY
        if issue >= 0 and issue % spec['stride'] == 0:
            issued = z[issue]
            p = cov @ issued
            den = spec['forget'] + issued @ p
            if not np.isfinite(den) or den <= 0:
                raise FloatingPointError('Invalid RLS denominator')
            gain = p / den
            theta += gain * ((residual[issue]-model['ym'])/model['ys'] - issued @ theta)
            cov = (cov - np.outer(gain,p))/spec['forget']
            cov = 0.5*(cov+cov.T)
            updates += 1
            if updates % 256 == 0:
                np.linalg.cholesky(cov)
        result[t] = spec['gain']*(model['ym']+model['ys']*(z[t] @ theta))
    if not np.isfinite(result).all() or not np.isfinite(cov).all() or np.linalg.norm(theta)>1e6:
        raise FloatingPointError('Invalid RLS state')
    return result, updates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ('source','experiment','profile','output'):
        cli.add_argument('--'+flag, required=True, type=lambda s: Path(s).resolve())
    args = cli.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path[:0] = [str(args.experiment/'runtimes/delta_CZ'), str(args.experiment)]
    import model as m

    m.L = 128
    m.configure(DELAY)
    archived = args.experiment/'runs/delta/CZ_Rod2/D4/full'
    obj = pickle.loads((archived/'model.pkl').read_bytes())
    current = obj['A']['selected']
    assert current['family']=='EX_RLS' and obj['upper']['parent']['config']['memory']
    g = json.loads((args.source/'GEOMETRY.json').read_text())
    train = np.load(args.source/'data/train.npz')['values']
    start = g['train_end']
    assert len(train)==start

    def issued(values, lo, hi):
        data = {ENTITY: values}
        window = m.frame(data, lo, hi)
        q = m.predict_kcw(data,window,obj['upper'])
        return m.attach_drivers(data,window,m.source(window,q,'CONTINUOUS')), q

    cal, _ = issued(train,g['parent_end']+128+2*DELAY,start-DELAY+1)
    designs = m.mature_a.all_designs(cal)
    grid = [dict(current,stride=s,forget=f) for s,f in itertools.product((1,2,4,8),(.995,.998,.999,.9995,1.))]
    folds = []
    for lo,hi in ((.5,.75),(.75,1.)):
        ix = m.mature_a.split(cal,lo,hi)
        fit_rows = m.mature_a.earlier(cal,ix)
        initializer = m.mature_a.initialize(cal,fit_rows,current,designs)
        folds.append((cal.iloc[ix],cal.iloc[:ix[0]].tail(64),initializer))
    validation = []
    for spec in grid:
        errors = []
        for frame,history,initializer in folds:
            corr,_ = predict(m,frame,history,initializer,spec)
            errors.append((frame.residual.to_numpy()-corr)**2)
        validation.append(dict(spec=spec, MSE=float(np.concatenate(errors).mean())))
    champion = min(validation,key=lambda row:row['MSE'])
    sparse = min((r for r in validation if r['spec']['stride']>1),key=lambda row:row['MSE'])
    original = next(r for r in validation if r['spec']==dict(current,stride=1))
    frozen = dict(best=champion, best_sparse=sparse, incumbent=original, grid=validation)
    (args.output/'FROZEN_SELECTION.json').write_text(json.dumps(frozen,indent=2))
    # Only these choices are replayed, after all selection results are frozen.
    values = np.vstack([train,np.load(args.source/'data/development.npz')['values']])
    assert len(values)==g['development_end']
    f,q = issued(values,start,len(values))
    h,_ = issued(values,start-64,start)
    results = []
    for name, row in [('incumbent',original),('selected',champion),('selected_sparse',sparse)]:
        spec = row['spec']
        corr, updates = predict(m,f,h,obj['A']['model'],spec)
        pred = q+corr
        if name=='incumbent':
            ref = pd.read_parquet(archived/'predictions.parquet').prediction
            np.testing.assert_allclose(pred,ref,rtol=1e-12,atol=1e-12)
        for n in (1,DELAY+1,65,257):
            cut = start+n-1
            changed = values.copy()
            changed[cut-DELAY+1:,0] += 1234
            changed[cut+1:,1:] -= 1234
            prefix,pq = issued(changed,start,cut+1)
            ph,_ = issued(changed,start-64,start)
            pc,_ = predict(m,prefix,ph,obj['A']['model'],spec)
            np.testing.assert_allclose(pq+pc,pred[:n],rtol=1e-12,atol=1e-12)
        rmse = float(np.sqrt(np.mean((pred-values[start:,0])**2)))
        results.append(dict(name=name,RMSE=rmse,updates=updates,**row))
        np.save(args.output/(name+'.npy'),pred)
    (args.output/'RESULTS.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
