"""Aggregate-only frozen-model diagnostics; no raw workbook or reserved tail reads.

Run in the original scientific environment on the authorized cache host:
  python -B replay_diagnostics.py --base /root/autodl-tmp > diagnostics.json
Writes only JSON to stdout; reconstructed candidate coefficients remain in RAM.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime(path):
    for name in ('model', 'mature_a', 'legacy', 'rls_core', 'stable_spline'):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    import model
    return model


def effect(y, full, other, skip=0):
    a = (y[skip:] - full[skip:]) ** 2
    b = (y[skip:] - other[skip:]) ** 2
    n = len(a)
    rng = np.random.default_rng(20261026)
    samples = []
    for _ in range(2500):
        starts = rng.integers(0, n, size=(n + 191) // 192)
        ix = ((starts[:, None] + np.arange(192)) % n).ravel()[:n]
        samples.append(100 * (np.sqrt(b[ix].mean() / a[ix].mean()) - 1))
    return dict(skip=skip, increase_percent=float(100 * (np.sqrt(b.mean() / a.mean()) - 1)),
                CI95=np.quantile(samples, [.025, .975]).tolist(), block=192, bootstrap=2500,
                role='reused-development descriptive interval; not adjusted for search')


def main(base):
    jobs = [('SRU_H2S', 8)] + [(e, d) for e in ('CZ_Rod1', 'CZ_Rod2') for d in (4, 8, 16)]
    attributions, ablations, checks = [], [], []
    for entity, delay in jobs:
        is_cz = entity.startswith('CZ_')
        root = base / ('PRISM_CZ_E128_20261026' if is_cz else 'PRISM_PAPER_SPLINE_REPAIR_20261026')
        mode = 'E128' if is_cz else 'repaired'
        m = runtime(root / ('E128' if is_cz else 'robust'))
        m.L = 128 if is_cz else 16
        m.configure(delay)
        m.SUPPORT = 127 + delay
        m.old.L = m.SUPPORT
        m.mature_a.configure(delay, m.SUPPORT)
        cache = base / 'cz_sru_matched_20261024' / entity if is_cz else base / 'PRISM_SRU_WINDOW_SELECTION_20261024'
        a = np.load(cache / 'data/train.npz')['values']
        key = f'{entity}_D{delay}'
        parent_end = 6636 if entity == 'CZ_Rod1' else 6810 if entity == 'CZ_Rod2' else 3328
        full_path = root / 'runs' / mode / entity / f'D{delay}' / 'full/model.pkl'
        full = pickle.loads(full_path.read_bytes())
        prepared = {'full': full}
        # Rod1 ONLY: recover constrained choices from the pre-existing ledger.
        # No new grid, A fit, or development-based candidate selection.
        if entity == 'CZ_Rod1':
            T = m.frame({key: a}, 192, parent_end)
            for variant in ('no_KC', 'no_W'):
                choices = [c for c in full['joint_candidates'] if
                           (c['kind'] == 'OWN' if variant == 'no_KC' else c['KCW_selected']['mode'] == 'identity')]
                selected = min(choices, key=lambda c: (c['selection_MSE'], c['kind'] != 'OWN'))
                prepared[variant] = dict(upper=m.fixed_kcw({key: a}, T, selected['KCW_selected']),
                                         A=copy.deepcopy(selected['A']))
            no_a_path = root / 'runs' / mode / entity / f'D{delay}' / 'no_A/model.pkl'
            prepared['no_A'] = pickle.loads(no_a_path.read_bytes())
        fits_before_eval = m.FITS
        b = np.load(cache / 'data/development.npz')['values']
        D = {key: np.vstack([a, b])}
        V = m.frame(D, len(a), len(a) + len(b))
        y = V.y.to_numpy()
        predictions = {}
        for variant, obj in prepared.items():
            q = m.predict_kcw(D, V, obj['upper'])
            correction = np.zeros(len(V)) if obj['A'] is None else m.a_predict(m.source(V, q, 'publication_replay'), obj['A'])[0]
            predictions[variant] = q + correction
        pred = predictions['full']
        saved_path = root / 'results' / mode / entity / f'D{delay}_predictions.parquet'
        saved = pd.read_parquet(saved_path)
        np.testing.assert_allclose(pred, saved.full, rtol=1e-12, atol=1e-12)
        if entity == 'CZ_Rod1':
            for variant, p in predictions.items():
                u = prepared[variant]['upper']
                A = prepared[variant]['A']
                ablations.append(dict(entity=entity, delay=delay, variant=variant, window=128, n=len(y),
                    RMSE=float(np.sqrt(np.mean((y-p)**2))), MAE=float(np.mean(np.abs(y-p))),
                    KC=u['parent']['config'], W=u['selected']['mode'],
                    A=None if A is None else dict(selected=A['selected'], admitted=A['admitted']),
                    provenance='persisted model' if variant in ('full', 'no_A') else 'frozen candidate-restricted reconstruction',
                    effects=[] if variant == 'full' else [effect(y, pred, p, skip) for skip in (0, 32+delay-1)]))
        # Exact additive allocation through the scalar output path.
        u = full['upper']
        parent = u['parent']
        r = parent['ridge']
        x, _ = m.design(D, V, parent['config'], parent['recipe'])
        latent = (x-r['mean'])/r['scale'] * (r['ys']*r['beta'])
        z = r['ym'] + latent.sum(1)
        def output(v):
            return m.w_predict(m.old.inverse(v, parent['coordinate']), u['W'])
        q = output(z)
        baseline = float(output(np.array([r['ym']]))[0])
        dz = z-r['ym']
        secant = np.divide(q-baseline, dz, out=np.zeros_like(dz), where=np.abs(dz)>1e-12)
        close = np.abs(dz)<=1e-12
        if close.any():
            h=1e-6*max(1.,abs(r['ym']))
            secant[close]=(output(np.array([r['ym']+h]))[0]-output(np.array([r['ym']-h]))[0])/(2*h)
        kind = parent['config']['kind']
        own_width = m.L if kind == 'OWN' else 2 if parent['config'].get('own', True) else 0
        names = ['u1','u2','u3','u4','u5'] if not is_cz else ['main_heater_power','crystal_lift_speed','crucible_lift_speed','crystal_rotation','crucible_rotation']
        groups = {'history_y': np.arange(own_width)}
        if kind != 'OWN':
            width = 2 if kind == 'URYSOHN' else 1
            process = np.arange(own_width, x.shape[1]).reshape(-1,5,width)
            groups.update({name: process[:,j,:].ravel() for j,name in enumerate(names)})
        else:
            groups.update({name:np.array([],dtype=int) for name in names})
        group_latent = {name:latent[:,ix].sum(1) for name,ix in groups.items()}
        components = {name:g*secant for name,g in group_latent.items()}
        components['A'] = pred-q
        np.testing.assert_allclose(baseline+sum(components.values()), pred, rtol=1e-12, atol=1e-12)
        yc=y-y.mean();pc=pred-pred.mean();variance=float(np.mean(yc*yc));rmse=float(np.sqrt(np.mean((y-pred)**2)))
        allocations={name:float(100*np.mean((2*yc-pc)*(g-g.mean()))/variance) for name,g in components.items()}
        bias_penalty=float(-100*(y.mean()-pred.mean())**2/variance)
        R2=float(1-np.mean((y-pred)**2)/variance)
        assert abs(sum(allocations.values())+bias_penalty-100*R2)<1e-8
        masking={}
        for name,g in group_latent.items():
            qm=output(z-g)
            am=np.zeros(len(V)) if full['A'] is None else m.a_predict(m.source(V,qm,'publication_center_mask'),full['A'])[0]
            masked_rmse=float(np.sqrt(np.mean((y-qm-am)**2)))
            masking[name]=dict(RMSE=masked_rmse,RMSE_increase_pct=100*(masked_rmse/rmse-1))
        attributions.append(dict(entity=entity,delay=delay,window=m.L,n=len(y),RMSE=rmse,R2=R2,
            model_sha256=digest(full_path),saved_prediction_sha256=digest(saved_path),
            output_path_R2_points=allocations,mean_bias_penalty_R2_points=bias_penalty,
            centered_feature_mask_with_A_replay=masking,
            additive_closure_max_abs=float(np.max(np.abs(baseline+sum(components.values())-pred)))))
        assert m.FITS==fits_before_eval
        checks.append(dict(entity=entity,delay=delay,frozen_full_prediction_match=True,
                           fits_during_development_replay=0,raw_workbook_read=False,reserved_tail_read=False))
    return dict(status='COMPLETED',attribution_method='scalar-secant allocation; R2_j=[2Cov(y,g_j)-Cov(pred,g_j)]/Var(y); add mean-bias penalty for exact R2 closure; predictive/noncausal/nonunique',
                masking_method='set a feature group to its parent ridge training center before scalar output link, replay frozen A; no refit',
                rod1_ablation_method='saved candidate-ledger restriction; recover parent using frozen recipe on original training rows, retain frozen A; not a fresh nested ablation search',
                rod2_module_ablation='DEFERRED_BY_USER; no Rod2 ablation evaluated or exported by this script',
                attribution=attributions,rod1_ablations=ablations,checks=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path('/root/autodl-tmp'))
    args=parser.parse_args()
    print(json.dumps(main(args.base),indent=2,allow_nan=False))
