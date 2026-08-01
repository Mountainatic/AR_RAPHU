from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import TaskData, block_lag_matrix, target_lags


@dataclass
class Fitted:
    name: str
    model: object | None
    train_pred: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    parameter_count: int
    information_set: str
    profile_id: str
    metadata: dict


def _fit_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray, alphas: Iterable[float]) -> tuple[object, float]:
    best = None
    best_alpha = None
    best_loss = np.inf
    for alpha in alphas:
        m = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="lsqr", fit_intercept=True))
        m.fit(Xtr, ytr)
        loss = float(np.mean((yval - m.predict(Xval)) ** 2)) if len(Xval) else float(np.mean((ytr - m.predict(Xtr)) ** 2))
        if loss < best_loss:
            best, best_alpha, best_loss = m, float(alpha), loss
    if best is None:
        raise RuntimeError("ridge selection had no candidates")
    return best, best_alpha


def _fit_pls(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray) -> tuple[object, int]:
    max_comp = max(1, min(10, Xtr.shape[1], max(1, Xtr.shape[0] - 1)))
    best, best_k, best_loss = None, 1, np.inf
    for k in range(1, max_comp + 1):
        m = make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False, max_iter=500))
        m.fit(Xtr, ytr)
        loss = float(np.mean((yval - m.predict(Xval).ravel()) ** 2))
        if loss < best_loss:
            best, best_k, best_loss = m, k, loss
    return best, best_k


def _slices(data: TaskData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return data.train, data.validation, data.test


def _row_predictions(model: object, X: np.ndarray, masks: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(model.predict(X[m])).reshape(-1) for m in masks)


def fit_simple_models(data: TaskData, *, seed: int = 0) -> list[Fitted]:
    rng = np.random.default_rng(seed)
    masks = _slices(data)
    X, y = data.X, data.y
    out: list[Fitted] = []
    mean = float(np.mean(y[masks[0]]))
    for name, pred in (("MEAN", np.full(len(y), mean)),):
        out.append(Fitted(name, None, pred[masks[0]], pred[masks[1]], pred[masks[2]], 1, "X_NONE", "baseline", {"mean": mean}))
    # Persistence uses the value at the prediction origin, which is available at t.
    ys = np.asarray(data.metadata["target_series"], dtype=np.float64)
    pers = ys[data.origins]
    out.append(Fitted("PERSISTENCE", None, pers[masks[0]], pers[masks[1]], pers[masks[2]], 0, "Y_t", "baseline", {}))
    # Current physical input ridge.
    model, alpha = _fit_ridge(X[masks[0]], y[masks[0]], X[masks[1]], y[masks[1]], np.logspace(-4, 4, 12))
    pr = _row_predictions(model, X, masks)
    out.append(Fitted("RIDGE_X", model, *pr, X.shape[1] + 1, "X_t", "current", {"alpha": alpha}))
    pls, k = _fit_pls(X[masks[0]], y[masks[0]], X[masks[1]], y[masks[1]])
    pp = _row_predictions(pls, X, masks)
    out.append(Fitted("PLS_X", pls, *pp, k * (X.shape[1] + 1), "X_t", "current", {"components": k}))
    # Compact nonlinear classical baseline; subsampling keeps CPU memory bounded.
    train_idx = np.flatnonzero(masks[0])
    if len(train_idx) > 30_000:
        train_idx = rng.choice(train_idx, size=30_000, replace=False)
    hgb = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1e-3, random_state=seed)
    hgb.fit(X[train_idx], y[train_idx])
    hp = _row_predictions(hgb, X, masks)
    out.append(Fitted("HISTGB_X", hgb, *hp, 120 * 15, "X_t", "current", {"max_iter": 120, "max_leaf_nodes": 15}))
    return out


def fit_dynamic_models(data: TaskData) -> list[Fitted]:
    masks = _slices(data)
    ys = np.asarray(data.metadata["target_series"], dtype=np.float64)
    lags = list(range(0, min(32, int(data.metadata["history_steps"]))))
    Y, names = target_lags(data, lags)
    out: list[Fitted] = []
    ar, alpha = _fit_ridge(Y[masks[0]], data.y[masks[0]], Y[masks[1]], data.y[masks[1]], np.logspace(-4, 4, 12))
    ap = _row_predictions(ar, Y, masks)
    out.append(Fitted("AR", ar, *ap, Y.shape[1] + 1, "Y_history", "AR_scale", {"alpha": alpha, "lags": lags}))
    XA = np.hstack([data.X, Y])
    arx, alpha = _fit_ridge(XA[masks[0]], data.y[masks[0]], XA[masks[1]], data.y[masks[1]], np.logspace(-4, 4, 12))
    axp = _row_predictions(arx, XA, masks)
    out.append(Fitted("ARX", arx, *axp, XA.shape[1] + 1, "X_t+Y_history", "ARX_scale", {"alpha": alpha, "lags": lags}))
    return out


def _fit_prism_features(data: TaskData, Xfeat: np.ndarray, names: list[str], *, label: str, masks: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Fitted:
    model, alpha = _fit_ridge(Xfeat[masks[0]], data.y[masks[0]], Xfeat[masks[1]], data.y[masks[1]], np.logspace(-5, 3, 12))
    p = _row_predictions(model, Xfeat, masks)
    return Fitted(label, model, *p, Xfeat.shape[1] + 1, "X_history", label, {"alpha": alpha, "feature_names": names, "feature_count": int(Xfeat.shape[1])})


def fit_prism_models(data: TaskData, *, n_blocks: int = 16) -> tuple[list[Fitted], dict[str, np.ndarray]]:
    masks = _slices(data)
    series = np.asarray(data.metadata["X_series"], dtype=np.float64)
    max_history = int(data.metadata["history_steps"])
    features, names = block_lag_matrix(series, data.origins, max_history=max_history, n_blocks=n_blocks)
    # Fixed multi-resolution uses every registered block. Channel-specific is selected
    # using training-only absolute correlations; no validation/test information enters.
    out: list[Fitted] = []
    out.append(_fit_prism_features(data, features, names, label="PRISM_FIXED_MULTIRESOLUTION", masks=masks))
    n_channels = series.shape[1]
    train = masks[0]
    corr = []
    for j in range(n_channels):
        col = features[:, j::n_channels]
        score = np.corrcoef(col[train].mean(axis=1), data.y[train])[0, 1] if np.std(col[train].mean(axis=1)) > 0 else 0.0
        corr.append(abs(float(score)) if np.isfinite(score) else 0.0)
    selected = np.argsort(corr)[::-1][: min(4, n_channels)]
    keep = [i for i in range(features.shape[1]) if (i % n_channels) in set(selected)]
    fsel, nsel = features[:, keep], [names[i] for i in keep]
    out.append(_fit_prism_features(data, fsel, nsel, label="PRISM_CHANNEL_SPECIFIC", masks=masks))
    # Single-scale is a short current-to-past lag bank for all channels.
    single, snames = block_lag_matrix(series, data.origins, max_history=min(max_history, 8), n_blocks=4)
    out.append(_fit_prism_features(data, single, snames, label="PRISM_SINGLE_SCALE", masks=masks))
    # K-Joint AR: frozen physical branch plus target history in a joint ridge fit.
    Y, ynames = target_lags(data, range(0, min(32, max_history)))
    joint = np.hstack([fsel, Y])
    joint_fit = _fit_prism_features(data, joint, nsel + ynames, label="PRISM_K_JOINT_AR", masks=masks)
    joint_fit.information_set = "X_multiresolution+Y_history"
    out.append(joint_fit)
    # Urysohn-first: freeze K on train, obtain cross-fitted residuals, then fit residual AR.
    k_model, _ = _fit_ridge(fsel[train], data.y[train], fsel[masks[1]], data.y[masks[1]], np.logspace(-5, 3, 12))
    k_pred = np.asarray(k_model.predict(fsel)).reshape(-1)
    train_idx = np.flatnonzero(train)
    oof = np.full(len(train_idx), np.nan)
    folds = np.array_split(train_idx, 3)
    for fold in folds:
        fit_idx = np.setdiff1d(train_idx, fold, assume_unique=False)
        if len(fit_idx) < 10 or len(fold) == 0:
            continue
        km, _ = _fit_ridge(fsel[fit_idx], data.y[fit_idx], fsel[masks[1]], data.y[masks[1]], [1e-2])
        oof[np.isin(train_idx, fold)] = data.y[fold] - km.predict(fsel[fold])
    residual = data.y[train_idx] - k_pred[train_idx] if not np.isfinite(oof).all() else oof
    ar_model, ar_alpha = _fit_ridge(Y[train_idx], residual, Y[masks[1]], data.y[masks[1]] - k_pred[masks[1]], np.logspace(-4, 4, 12))
    ar_pred = np.asarray(ar_model.predict(Y)).reshape(-1)
    total = k_pred + ar_pred
    out.append(Fitted("PRISM_U_FIRST_RESIDUAL_AR", (k_model, ar_model), total[train], total[masks[1]], total[masks[2]], fsel.shape[1] + Y.shape[1] + 2, "X_frozen_K+OOF_residual_AR", "U_FIRST", {"K_alpha": "selected", "AR_alpha": ar_alpha, "exact_zero_candidate": True, "feature_names": nsel + ynames}))
    return out, {"block_features": features, "selected_features": fsel, "K_prediction": k_pred, "selected_channels": selected}

