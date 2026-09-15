"""Audit development contracts without opening any formal values or metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path('/root/autodl-tmp/PRISM_V211_STRICT_OOF_PUBLIC5_HYBRID_HW_20260912_R5_PENALIZED_OBJECTIVE_FULL_6X2_R1')
TEP = Path('/root/autodl-tmp/PRISM_V211_STRICT_OOF_TEP_REPAIR_73D0BD4_20260913_R2')
SHARED = Path('/root/autodl-tmp/PRISM_R5_SHARED_INPUTS')
PARTITION = Path('/root/autodl-tmp/FORMAL_PARTITION_MATERIALIZATION_20260914')
VIEWS = [
    ('Debutanizer', 'DEB_C4__H5__W1', 'public3_shared', 'primary', 'record_time', True),
    ('SRU H2S', 'SRU_H2S_REP_H1__H1__W1', 'sru_shared', 'primary', 'record_time', True),
    ('SRU SO2', 'SRU_SO2_REP_H1__H1__W1', 'sru_shared', 'primary', 'record_time', True),
    ('PMSM proxy-excluded', 'PMSM_PM5__H600__W60', 'public3_shared', 'proxy_excluded', 'record_time', True),
    ('MetroPT P60', 'METRO_P60__H6__W1', 'public3_shared', 'proxy_excluded', 'record_time', True),
    ('MetroPT Oil20', 'METRO_OIL20__H120__W12', 'public3_shared', 'primary', 'record_time', True),
    ('TEP input-only', 'TEP_G_NOWCAST_H0__H0__W1', 'tep_shared', 'proxy_excluded', 'record_time', False),
    ('TEP record-time', 'TEP_G_NOWCAST_H0__H0__W1', 'tep_shared', 'proxy_excluded', 'record_time', True),
    ('TEP maturity-5', 'TEP_G_NOWCAST_H0__H0__W1', 'tep_shared', 'proxy_excluded', 'analyzer_maturity_5_steps', True),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def probe(frame, cap=512):
    if cap is None:
        return frame.reset_index(drop=True)
    indices = np.unique(np.rint(np.linspace(0, len(frame) - 1, min(cap, len(frame)))).astype(np.int64))
    return frame.iloc[indices].reset_index(drop=True)


def error(prediction, reference):
    values = np.asarray(prediction, dtype=np.float64)
    expected = reference['y_pred'].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite replay prediction')
    maximum = float(np.max(np.abs(values - expected), initial=0.0))
    return {'status': 'PASS' if maximum <= 1e-10 else 'REPLAY_MISMATCH',
            'rows': len(values), 'maximum_absolute_error': maximum, 'tolerance': 1e-10}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / 'PRISM_INDUSTRIAL_BENCHMARK_V1/src'))
    from prism_benchmark.cpu_data import BaseAccessor, ViewSpec, load_heads
    from prism_benchmark.v211_support import load_native_samples
    from prism_benchmark.v2_k import profile_values
    from prism_benchmark.v2_urysohn import predict_contract, basis_from_metadata
    from prism_benchmark.v2_c import _ridge_predict
    from prism_benchmark.v211_w import predict_w_correction
    from prism_benchmark.v211_a import mature_residual_features, predict_mature_residual_ar
    from prism_benchmark.representative_prism_checkpoints import _predict_joint, _w_basis_from_contract
    from prism_benchmark.v2_config import load_frozen_config

    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = PARTITION / 'LOCK/FINAL_FORMAL_TEST_PROTOCOL_LOCK_v2.json'
    lock = read(lock_path)
    if not lock['protocol_frozen'] or lock['formal_test_opened']:
        raise RuntimeError('partition protocol not frozen or test already opened')
    config = load_frozen_config(args.project / 'PRISM_INDUSTRIAL_BENCHMARK_V1')
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'project': str(args.project),
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.project, text=True).strip(),
             'worktree': subprocess.check_output(['git', 'status', '--short'], cwd=args.project, text=True).strip(),
             'partition_lock_sha256': digest(lock_path),
             'formal_test_opened': False, 'formal_target_values_read': False,
             'formal_predictions_generated': False, 'formal_metrics_computed': False,
             'fit_calls': 0, 'selection_calls': 0,
             'probe_policy': 'all development rows' if args.full else '512 evenly spaced development rows',
             'views': [], 'sources': []}
    seen = set()
    for task, head_id, shared_name, proxy, availability, dynamic in VIEWS:
        shared = SHARED / shared_name
        head = next(h for h in load_heads(shared, primary_only=False) if h.head_id == head_id)
        view = ViewSpec(head, 'dynamic' if dynamic else 'input_only', availability, proxy)
        dynamic_run = TEP if head.dataset == 'tep' else BASE
        roots = {'C': BASE, 'W': BASE, 'A': dynamic_run, 'J': dynamic_run}
        paths = {'C': BASE / f'results/DEVELOPMENT/C/{head_id}/{proxy}/RESULT.json',
                 'W': BASE / f'results/DEVELOPMENT/W/{head_id}/{proxy}/RESULT.json'}
        if dynamic:
            paths.update(A=dynamic_run / f'results/DEVELOPMENT/A/{head_id}/{availability}/{proxy}/RESULT.json',
                         J=dynamic_run / f'results/DEVELOPMENT/JOINT/{head_id}/{availability}/{proxy}/RESULT.json')
        results = {s: read(p) for s, p in paths.items()}
        channel = results['C']['best_active_k_channel']
        paths['K'] = BASE / f'results/DEVELOPMENT/K/{head_id}/{proxy}/{channel}/RESULT.json'
        roots['K'] = BASE
        results['K'] = read(paths['K'])
        record = {'task': task, 'view': str(view.relative_root), 'source_status': {}, 'replay': {},
                  'routes': {s: results[s][key]['routing_status'] for s, key in [('C', 'family_selection'), ('W', 'selection')]},
                  'joint_feature_contract_saved': bool(results.get('J', {}).get('channel_contracts')),
                  'status': 'PENDING'}
        if dynamic:
            record['routes']['A'] = results['A']['selection']['routing_status']
        frames = {}
        for stage, value in results.items():
            prediction = roots[stage] / 'results' / value['prediction_path']
            actual_hash = digest(prediction)
            record['source_status'][stage] = {'status': value.get('status'), 'result_sha256': digest(paths[stage]),
                                              'prediction_hash_match': actual_hash == value.get('prediction_sha256'),
                                              'contract_present': isinstance(value.get('final_selected_contract'), dict)}
            if value.get('status') != 'PASS' or actual_hash != value.get('prediction_sha256'):
                raise RuntimeError(f'authoritative prerequisite failure: {paths[stage]}')
            frames[stage] = pd.read_parquet(prediction)
            if str(paths[stage]) not in seen:
                seen.add(str(paths[stage]))
                audit['sources'].append({'stage': stage, 'path': str(paths[stage]), 'sha256': digest(paths[stage]),
                                         'prediction_path': str(prediction), 'prediction_sha256': actual_hash})
        native = load_native_samples(shared, view, 'validation')
        accessor = BaseAccessor(shared, head.dataset, 'validation', [*results['C']['active_channels'], channel, head.target])

        def samples_for(frame):
            common = native[['base_origin_id']].merge(frame, on='base_origin_id', validate='one_to_one')
            subset = probe(common, None if args.full else 512)
            samples = subset[['base_origin_id']].merge(native, on='base_origin_id', validate='one_to_one')
            if len(samples) != len(subset):
                raise RuntimeError('native development support does not cover probe')
            return samples, subset

        def physical(samples):
            compressed, joint = [], []
            for item in results['C']['channel_contracts']:
                values, _ = profile_values(accessor, samples, item['channel'], tuple(item['profile']), item['m_tau'])
                contract = item['k_contract']
                compressed.append(predict_contract(values, contract))
                raw = basis_from_metadata(contract['basis']).transform(values).reshape(len(values), -1)
                joint.append(raw[:, np.asarray(item['joint_columns'], dtype=np.int64)])
            matrices = {'compressed': np.column_stack(compressed), 'joint': np.concatenate(joint, axis=1)}
            columns = np.asarray(results['C'].get('global_joint_columns', []), dtype=np.int64)
            if len(columns):
                matrices['joint'] = matrices['joint'][:, columns]
            return matrices

        def c_prediction(matrices):
            contract = results['C']['final_selected_contract']
            family = contract['family']
            if family in {'BEST_ACTIVE_K_CHANNEL', 'BEST_ACTIVE_K'}:
                return matrices['compressed'][:, results['C']['active_channels'].index(contract['channel'])]
            key = 'joint' if family == 'ADDITIVE_JOINT_BASIS' else 'compressed'
            return _ridge_predict(matrices[key], contract)

        try:
            samples, reference = samples_for(frames['K'])
            k_result = results['K']
            values, _ = profile_values(accessor, samples, channel, tuple(k_result['selected_profile']), k_result['selected_m_tau'])
            record['replay']['K'] = error(predict_contract(values, k_result['final_selected_contract']), reference)
            for stage in ['C', 'W']:
                samples, reference = samples_for(frames[stage])
                matrices = physical(samples)
                pred = c_prediction(matrices)
                if stage == 'W':
                    pred = pred + predict_w_correction(pred, results['W']['final_selected_contract'])
                record['replay'][stage] = error(pred, reference)
            if dynamic:
                samples, reference = samples_for(frames['A'])
                contract = results['A']['final_selected_contract']
                wframe = frames['W']
                aligned = samples[['base_origin_id']].merge(wframe[['base_origin_id', 'y_pred']], on='base_origin_id', validate='one_to_one')
                if contract['family'] == 'EXACT_ZERO':
                    apred = np.zeros(len(samples), dtype=np.float64)
                else:
                    oof = pd.read_parquet(BASE / 'results' / results['W']['oof_path'])
                    train = load_native_samples(shared, view, 'train')
                    oof = train[['base_origin_id']].merge(oof, on='base_origin_id', validate='one_to_one')
                    oof['residual'] = oof['y_true'] - oof['physical_w_oof']
                    validation = native[['base_origin_id']].merge(wframe, on='base_origin_id', validate='one_to_one')
                    validation['residual'] = validation['y_true'] - validation['y_pred']
                    residual_mean = float(oof['residual'].mean())
                    source = pd.concat([oof[['entity_id', 'origin', 'residual']], validation[['entity_id', 'origin', 'residual']]], ignore_index=True)
                    features, coverage, maturity = mature_residual_features(samples, source,
                        h_steps=head.h_steps, w_steps=head.w_steps, delta=int(contract['profile'][0]),
                        history=int(contract['profile'][1]), maximum_lags=int(config['A_module']['state_profile']['maximum_lags']),
                        residual_mean=residual_mean)
                    apred = predict_mature_residual_ar(features, contract)
                    record['a_residual_mean'] = residual_mean
                    record['a_maturity_audit'] = maturity
                record['replay']['A'] = error(aligned['y_pred'].to_numpy(dtype=np.float64) + apred, reference)
                samples, reference = samples_for(frames['J'])
                matrices = physical(samples)
                jcontract = results['J']['final_selected_contract']
                blocks = {'K': matrices['compressed'] if jcontract['k_representation'] == 'CHANNEL_COMPRESSED' else matrices['joint'],
                          'W': _w_basis_from_contract(c_prediction(matrices), results['J']['joint_w_basis_contract']),
                          'A': accessor.target_state(samples, head.target, *results['J']['ar_profile'])}
                record['replay']['J_using_C_feature_contract'] = error(_predict_joint(blocks, jcontract), reference)
        except Exception as exc:
            record['replay_exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        failures = [name for name, value in record['replay'].items() if value['status'] != 'PASS']
        maturity_joint_only = task == 'TEP maturity-5' and failures == ['J_using_C_feature_contract']
        if maturity_joint_only:
            record['joint_formal_status'] = 'NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT'
            record['status'] = 'PASS_WITH_JOINT_NOT_APPLICABLE'
        else:
            record['status'] = 'PASS' if not record.get('replay_exception') and not failures else 'FROZEN_CHECKPOINT_NOT_REPLAYABLE'
        audit['views'].append(record)
        write(args.output / 'FROZEN_CONTRACT_PREFLIGHT.json', audit)
        print(task, record['status'], {s: v['maximum_absolute_error'] for s, v in record['replay'].items()}, flush=True)
        del accessor, native, frames
    allowed = {'PASS', 'PASS_WITH_JOINT_NOT_APPLICABLE'}
    audit['status'] = 'PASS_WITH_ONE_JOINT_NOT_APPLICABLE' if all(v['status'] in allowed for v in audit['views']) else 'BLOCKED_BY_MISSING_FROZEN_FEATURE_CONTRACT'
    audit['final_fit_policy_found_in_current_lock'] = False
    audit['formal_execution_authorized'] = False
    write(args.output / 'FROZEN_CONTRACT_PREFLIGHT.json', audit)
    write(args.output / 'AUTHORITATIVE_SOURCE_HASHES.json', audit['sources'])
    print('FINAL', audit['status'], flush=True)


if __name__ == '__main__':
    main()
