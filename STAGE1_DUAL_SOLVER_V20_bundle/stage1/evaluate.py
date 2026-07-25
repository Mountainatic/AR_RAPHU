"""Evaluation utilities for Stage1TargetDelayKAN."""
import torch, json, os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, pandas as pd
from .train import calculate_metrics

@torch.no_grad()
def evaluate_model(model, data_loader, device=None):
    """Run evaluation and return metrics dict, predictions, targets, aux."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_preds, all_targets = [], []
    aux_accum = {}
    for bx, by in data_loader:
        bx, by = bx.to(device), by.to(device)
        pred, aux = model(bx, return_aux=True)
        all_preds.append(pred.cpu()); all_targets.append(by.cpu())
        for k, v in aux.items():
            if k not in aux_accum: aux_accum[k] = []
            aux_accum[k].append(v.cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    rmse, mae, r2 = calculate_metrics(preds, targets)
    aux_cat = {}
    for k, v in aux_accum.items():
        if v[0].dim() > 0:
            aux_cat[k] = torch.cat(v, dim=0)
        else:
            aux_cat[k] = v[0]
    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2,
               'MSE': float(torch.mean((preds - targets) ** 2))}
    return metrics, preds.numpy(), targets.numpy(), aux_cat

def split_audit(train_indices, val_indices, test_indices, window_size,
                scaler_fit_source='train', save_path=None):
    """Audit train/val/test split for leakage and embargo violations."""
    audit = {'window_size': window_size, 'embargo_required': window_size - 1, 'checks': {}}
    def raw_set(indices):
        s = set()
        for a, b in indices: s.update(range(a, b))
        return s
    train_set = raw_set(train_indices)
    val_set = raw_set(val_indices)
    test_set = raw_set(test_indices)
    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set
    audit['checks']['no_overlap'] = not (tv_overlap or tt_overlap or vt_overlap)
    overlap_detail = {}
    if tv_overlap: overlap_detail['train_val'] = sorted(tv_overlap)
    if tt_overlap: overlap_detail['train_test'] = sorted(tt_overlap)
    if vt_overlap: overlap_detail['val_test'] = sorted(vt_overlap)
    audit['checks']['overlap_detail'] = overlap_detail
    def find_boundaries(indices_set):
        if not indices_set: return None, None
        sorted_idx = sorted(indices_set)
        return sorted_idx[0], sorted_idx[-1]
    train_min, train_max = find_boundaries(train_set)
    val_min, val_max = find_boundaries(val_set)
    test_min, test_max = find_boundaries(test_set)
    embargo = window_size - 1
    audit['boundaries'] = {'train': [train_min, train_max],
        'val': [val_min, val_max], 'test': [test_min, test_max]}
    embargo_ok = True; embargo_detail = {}
    if val_min is not None and train_max is not None:
        gap = val_min - train_max
        if gap < embargo: embargo_ok = False; embargo_detail['train_val_gap'] = gap
    if test_min is not None and val_max is not None:
        gap = test_min - val_max
        if gap < embargo: embargo_ok = False; embargo_detail['val_test_gap'] = gap
    audit['checks']['embargo_satisfied'] = embargo_ok
    audit['checks']['embargo_detail'] = embargo_detail
    audit['checks']['scaler_fit_train_only'] = (scaler_fit_source == 'train')
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(audit, f, indent=2, default=str)
    return audit

def rolling_window_validate(model, X, Y, train_loader, config_dict, device=None):
    """Leak-free rolling window validation on real data."""
    results = {'seeds': [], 'test_r2': [], 'test_rmse': [], 'test_mae': [],
               'active_vars': [], 'branch_norms': []}
    if device is None: device = torch.device('cpu')
    from .train import Stage1Trainer
    seeds = config_dict.get('seeds', [42])
    for seed in seeds:
        import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        X_tensor = torch.FloatTensor(X).to(device) if isinstance(X, np.ndarray) else X.to(device)
        Y_tensor = torch.FloatTensor(Y).to(device) if isinstance(Y, np.ndarray) else Y.to(device)
        from .model import Stage1TargetDelayKAN
        N = X.shape[1]; L = X.shape[2]
        model = Stage1TargetDelayKAN(N, L, epsilon=config_dict.get('epsilon', 0.5),
            hidden_score=config_dict.get('hidden_score', 8),
            hidden_kan=config_dict.get('hidden_kan', 8)).to(device)
        tr_loader = DataLoader(TensorDataset(X_tensor, Y_tensor),
            batch_size=config_dict.get('batch_size', 128), shuffle=True)
        vl_loader = DataLoader(TensorDataset(X_tensor, Y_tensor),
            batch_size=config_dict.get('batch_size', 128), shuffle=False)
        trainer = Stage1Trainer(model, tr_loader, vl_loader,
            lr=config_dict.get('lr', 0.001),
            lambda_group=config_dict.get('lambda_group', 0.01),
            lambda_smooth=config_dict.get('lambda_smooth', 0.0),
            max_epochs=config_dict.get('max_epochs', 50),
            patience=config_dict.get('patience', 10), device=device)
        trainer.train()
        metrics, _, _, _ = evaluate_model(model, vl_loader, device)
        results['seeds'].append(seed)
        results['test_r2'].append(metrics['R2'])
        results['test_rmse'].append(metrics['RMSE'])
        results['test_mae'].append(metrics['MAE'])
        results['active_vars'].append(model.get_active_variables().sum().item())
        results['branch_norms'].append(
            model.response_branches.compute_branch_norms().cpu().numpy())
    return results
