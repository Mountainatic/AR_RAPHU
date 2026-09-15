"""Preregistered strict nested-OOF external W validation on EMPS."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


HISTORIES = (4, 8, 16, 32)
BLOCKS = (1, 2, 4, 8)
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0)
W_GRID = tuple((knots, smooth, mu) for knots in (4, 6, 8, 12) for smooth in (0.0, 1e-4, 1e-2, 1.0) for mu in (0.0, 0.03, 0.3))
SEED = 20260915


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale <= 64 * np.finfo(np.float64).eps] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(), "alpha": alpha}


def ridge_predict(x: np.ndarray, contract: dict) -> np.ndarray:
    mean = np.asarray(contract["mean"])
    scale = np.asarray(contract["scale"])
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return design @ np.asarray(contract["coefficient"])


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2, dtype=np.float64))


def metrics(delta: np.ndarray, pred: np.ndarray, current: np.ndarray) -> dict:
    error = pred - delta
    denominator = float(np.sum((delta - delta.mean()) ** 2))
    future = current + delta
    level = current + pred
    level_den = float(np.sum((future - future.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "delta_r2": float(1 - np.sum(error * error) / denominator),
        "level_rmse": float(np.sqrt(np.mean((level - future) ** 2))),
        "level_mae": float(np.mean(np.abs(level - future))),
        "level_r2": float(1 - np.sum((level - future) ** 2) / level_den),
    }


def edges(history: int, blocks: int) -> np.ndarray:
    values = np.unique(np.rint(np.geomspace(1, history, blocks)).astype(int))
    if values[-1] != history:
        values = np.r_[values, history]
    return values


def channel_features(values: np.ndarray, origins: np.ndarray, history: int, blocks: int) -> np.ndarray:
    boundaries = edges(history, blocks)
    columns = []
    for near, far in zip(np.r_[0, boundaries[:-1]], boundaries):
        offsets = np.arange(int(near), int(far), dtype=np.int64)
        columns.append(values[origins[:, None] - offsets[None, :]].mean(axis=1))
    return np.column_stack(columns)


def arrays(mat: dict) -> dict:
    qm = np.asarray(mat["qm"][:, 0], dtype=np.float64)
    origins = np.arange(max(HISTORIES) - 1, len(qm) - 1, dtype=np.int64)
    return {
        "origins": origins,
        "target": qm[origins + 1] - qm[origins],
        "current": qm[origins],
        "vir": np.asarray(mat["vir"][:, 0], dtype=np.float64),
        "qg": np.asarray(mat["qg"][:, 0], dtype=np.float64),
        "sampling_time": float(np.asarray(mat["t"][:, 0])[1] - np.asarray(mat["t"][:, 0])[0]),
    }


def folds(rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    boundaries = [int(rows * value) for value in (0.50, 0.625, 0.75, 0.875, 1.0)]
    return [(np.arange(boundaries[i]), np.arange(boundaries[i], boundaries[i + 1])) for i in range(4)]


def inner_folds(rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    boundaries = [int(rows * value) for value in (0.40, 0.60, 0.75, 0.875, 1.0)]
    return [(np.arange(boundaries[i]), np.arange(boundaries[i], boundaries[i + 1])) for i in range(4) if boundaries[i] > 100]


def all_features(data: dict) -> dict:
    result = {}
    for channel in ("vir", "qg"):
        for history in HISTORIES:
            for blocks in BLOCKS:
                if blocks <= history:
                    result[(channel, history, blocks)] = channel_features(data[channel], data["origins"], history, blocks)
    return result


def select_k(features: dict, y: np.ndarray, subset: np.ndarray) -> tuple[dict, dict]:
    selected = {}
    scores = {}
    for channel in ("vir", "qg"):
        candidates = []
        for history in HISTORIES:
            for blocks in BLOCKS:
                if blocks > history:
                    continue
                x = features[(channel, history, blocks)][subset]
                for alpha in ALPHAS:
                    fold_scores = []
                    for fit, validation in inner_folds(len(subset)):
                        contract = ridge_fit(x[fit], y[subset][fit], alpha)
                        fold_scores.append(mse(y[subset][validation], ridge_predict(x[validation], contract)))
                    candidates.append((float(np.mean(fold_scores)), history, blocks, alpha))
        score, history, blocks, alpha = min(candidates)
        selected[channel] = {"history": history, "blocks": blocks, "alpha": alpha}
        scores[channel] = score
    return selected, scores


def fit_k(selected: dict, features: dict, y: np.ndarray, fit: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    fit_columns, eval_columns, contracts = [], [], {}
    for channel in ("vir", "qg"):
        config = selected[channel]
        x = features[(channel, config["history"], config["blocks"])]
        contract = ridge_fit(x[fit], y[fit], config["alpha"])
        contracts[channel] = {"config": config, "ridge": contract}
        fit_columns.append(ridge_predict(x[fit], contract))
        eval_columns.append(ridge_predict(x[evaluation], contract))
    return np.column_stack(fit_columns), np.column_stack(eval_columns), contracts


def select_c_alpha(selected_k: dict, features: dict, y: np.ndarray, subset: np.ndarray) -> float:
    scores = {alpha: [] for alpha in ALPHAS}
    for fit_local, val_local in inner_folds(len(subset)):
        fit, validation = subset[fit_local], subset[val_local]
        k_fit, k_val, _ = fit_k(selected_k, features, y, fit, validation)
        for alpha in ALPHAS:
            contract = ridge_fit(k_fit, y[fit], alpha)
            scores[alpha].append(mse(y[validation], ridge_predict(k_val, contract)))
    return min(ALPHAS, key=lambda alpha: (float(np.mean(scores[alpha])), alpha))


def chain_fit(selected_k: dict, best_channel: str, c_alpha: float, c_route: str, features: dict, y: np.ndarray, fit: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    k_fit, k_eval, k_contracts = fit_k(selected_k, features, y, fit, evaluation)
    channel_index = ("vir", "qg").index(best_channel)
    parent_fit, parent_eval = k_fit[:, channel_index], k_eval[:, channel_index]
    c_contract = ridge_fit(k_fit, y[fit], c_alpha)
    c_fit, c_eval = ridge_predict(k_fit, c_contract), ridge_predict(k_eval, c_contract)
    if c_route == "ZERO_IDENTITY":
        return parent_fit, parent_eval, {"K": k_contracts, "best_channel": best_channel, "C": None}
    return c_fit, c_eval, {"K": k_contracts, "best_channel": best_channel, "C": c_contract}


def select_w(chain_fit_values: np.ndarray, y_fit: np.ndarray, fit_w_correction) -> tuple[tuple, dict]:
    split = max(100, int(len(y_fit) * 0.75))
    candidates = []
    for knots, smooth, mu in W_GRID:
        correction, contract = fit_w_correction(chain_fit_values[:split], y_fit[:split] - chain_fit_values[:split], chain_fit_values[split:], family="NATURAL_CUBIC_CORRECTION", knot_count=knots, smoothness=smooth, mu=mu, upstream_predictions=chain_fit_values[:split, None], direction=1)
        candidates.append((mse(y_fit[split:], chain_fit_values[split:] + correction), knots, smooth, mu, contract))
    _, knots, smooth, mu, contract = min(candidates, key=lambda value: value[:4])
    return (knots, smooth, mu), contract


def epsilon(parent_mse: float, child_mse: float) -> float:
    return float(1000 * np.finfo(np.float64).eps * max(1.0, abs(parent_mse), abs(child_mse)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    sys_path = args.project / "PRISM_INDUSTRIAL_BENCHMARK_V1/src"
    import sys
    sys.path.insert(0, str(sys_path))
    from prism_benchmark.v211_w import fit_w_correction, predict_w_correction

    prereg = root / "PRE_REGISTRATION.json"
    if not prereg.is_file() or (root / "TEST_OPENING.json").exists():
        raise RuntimeError("pre-registration missing or test already opened")
    dev_path = root / "DATA/DATA_EMPS.mat"
    test_path = root / "DATA/DATA_EMPS_PULSES.mat"
    dev = arrays(loadmat(dev_path))
    y, features = dev["target"], all_features(dev)
    outer_records, k_oof, c_oof = [], np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_contracts = []
    for fold_index, (fit, validation) in enumerate(folds(len(y))):
        selected_k, k_scores = select_k(features, y, fit)
        best_channel = min(k_scores, key=k_scores.get)
        c_alpha = select_c_alpha(selected_k, features, y, fit)
        k_fit, k_val, k_contracts = fit_k(selected_k, features, y, fit, validation)
        best_index = ("vir", "qg").index(best_channel)
        c_contract = ridge_fit(k_fit, y[fit], c_alpha)
        pk, pc = k_val[:, best_index], ridge_predict(k_val, c_contract)
        k_oof[validation], c_oof[validation] = pk, pc
        fold_contracts.append({"fold": fold_index, "selected_k": selected_k, "best_channel": best_channel, "c_alpha": c_alpha})
        outer_records.append({"fold": fold_index, "fit_rows": len(fit), "validation_rows": len(validation), "k_mse": mse(y[validation], pk), "c_mse": mse(y[validation], pc)})
    support = np.isfinite(k_oof) & np.isfinite(c_oof)
    k_mse, c_mse = mse(y[support], k_oof[support]), mse(y[support], c_oof[support])
    c_route = "ACTIVE" if k_mse - c_mse > epsilon(k_mse, c_mse) else "ZERO_IDENTITY"

    parent_oof = c_oof.copy() if c_route == "ACTIVE" else k_oof.copy()
    w_oof = np.full(len(y), np.nan)
    w_configs = []
    for fold_index, (fit, validation) in enumerate(folds(len(y))):
        info = fold_contracts[fold_index]
        chain_fit_values, chain_eval, _ = chain_fit(info["selected_k"], info["best_channel"], info["c_alpha"], c_route, features, y, fit, validation)
        w_config, _ = select_w(chain_fit_values, y[fit], fit_w_correction)
        correction, w_contract = fit_w_correction(chain_fit_values, y[fit] - chain_fit_values, chain_eval, family="NATURAL_CUBIC_CORRECTION", knot_count=w_config[0], smoothness=w_config[1], mu=w_config[2], upstream_predictions=chain_fit_values[:, None], direction=1)
        w_oof[validation] = chain_eval + correction
        w_configs.append({"fold": fold_index, "config": list(w_config), "contract": w_contract})
    w_support = np.isfinite(parent_oof) & np.isfinite(w_oof)
    parent_mse, child_mse = mse(y[w_support], parent_oof[w_support]), mse(y[w_support], w_oof[w_support])
    w_route = "ACTIVE" if parent_mse - child_mse > epsilon(parent_mse, child_mse) else "ZERO_IDENTITY"

    all_rows = np.arange(len(y))
    selected_k, k_scores = select_k(features, y, all_rows)
    best_channel = min(k_scores, key=k_scores.get)
    c_alpha = select_c_alpha(selected_k, features, y, all_rows)
    chain_dev, _, chain_contract = chain_fit(selected_k, best_channel, c_alpha, c_route, features, y, all_rows, all_rows)
    w_config, _ = select_w(chain_dev, y, fit_w_correction)
    _, final_w_contract = fit_w_correction(chain_dev, y - chain_dev, chain_dev, family="NATURAL_CUBIC_CORRECTION", knot_count=w_config[0], smoothness=w_config[1], mu=w_config[2], upstream_predictions=chain_dev[:, None], direction=1)
    freeze = {
        "status": "FROZEN_BEFORE_TEST_ACCESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration_sha256": sha256(prereg),
        "development_file_sha256": sha256(dev_path),
        "test_file_bytes_without_content_access": test_path.stat().st_size,
        "sampling_time": dev["sampling_time"],
        "development_rows": len(y),
        "outer_folds": outer_records,
        "C": {"route": c_route, "parent_mse": k_mse, "child_mse": c_mse, "gain": (k_mse - c_mse) / k_mse, "epsilon_num": epsilon(k_mse, c_mse)},
        "W": {"route": w_route, "parent_mse": parent_mse, "child_mse": child_mse, "gain": (parent_mse - child_mse) / parent_mse, "epsilon_num": epsilon(parent_mse, child_mse)},
        "final": {"selected_k": selected_k, "best_channel": best_channel, "c_alpha": c_alpha, "chain_contract": chain_contract, "w_config": list(w_config), "w_contract": final_w_contract},
        "fit_calls_scope": "DEVELOPMENT_ONLY",
        "selection_calls_scope": "DEVELOPMENT_ONLY",
        "test_target_accessed": False,
    }
    freeze_path = root / "DEVELOPMENT_FREEZE.json"
    write_json(freeze_path, freeze)
    opening = {"EXTERNAL_TEST_OPENED": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "development_freeze_sha256": sha256(freeze_path), "pre_registration_sha256": sha256(prereg), "first_test_target_access_follows_this_event": True}
    write_json(root / "TEST_OPENING.json", opening)

    test = arrays(loadmat(test_path))
    test_features = all_features(test)
    test_rows = np.arange(len(test["target"]))
    test_k_columns = []
    for channel in ("vir", "qg"):
        cfg = selected_k[channel]
        contract = chain_contract["K"][channel]["ridge"]
        x_test = test_features[(channel, cfg["history"], cfg["blocks"])]
        test_k_columns.append(ridge_predict(x_test, contract))
    test_k = np.column_stack(test_k_columns)
    best_index = ("vir", "qg").index(best_channel)
    pk = test_k[:, best_index]
    if c_route == "ACTIVE":
        pc = ridge_predict(test_k, chain_contract["C"])
    else:
        pc = pk.copy()
    pw_candidate = pc + predict_w_correction(pc, final_w_contract)
    pw = pw_candidate if w_route == "ACTIVE" else pc.copy()
    result = {
        "status": "COMPLETED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "development_freeze_sha256": sha256(freeze_path),
        "test_file_sha256_after_open": sha256(test_path),
        "test_rows": len(test_rows),
        "sampling_time": test["sampling_time"],
        "C_route": c_route,
        "W_route": w_route,
        "development_W_margin": (parent_mse - child_mse) / parent_mse,
        "KC": metrics(test["target"], pc, test["current"]),
        "KCW": metrics(test["target"], pw, test["current"]),
        "W_gain": (np.sqrt(mse(test["target"], pc)) - np.sqrt(mse(test["target"], pw))) / np.sqrt(mse(test["target"], pc)),
        "residual_skill": 1 - mse(test["target"], pw) / mse(test["target"], pc),
        "fit_calls_after_test_open": 0,
        "selection_calls_after_test_open": 0,
    }
    if w_route == "ACTIVE" and result["W_gain"] > 0:
        result["verdict"] = "REAL_WORLD_W_INCREMENTAL_VALUE_SUPPORTED"
    elif w_route == "ZERO_IDENTITY":
        result["verdict"] = "W_CORRECTLY_REJECTED_ON_THIS_TASK"
    else:
        result["verdict"] = "W_ADMISSION_DID_NOT_GENERALIZE_ON_THIS_TASK"
    write_json(root / "RESULT.json", result)
    pd.DataFrame({"sample_id": [f"EMPS_PULSES:{v}" for v in test["origins"]], "origin": test["origins"], "y_true": test["target"], "current_level": test["current"], "KC": pc, "KCW": pw}).to_parquet(root / "TEST_PREDICTIONS.parquet", index=False, compression="zstd")
    (root / "REPORT.md").write_text(f"""# External W Validation: EMPS\n\nStatus: **COMPLETED**\n\nThe official file split was used: `DATA_EMPS.mat` for development and `DATA_EMPS_PULSES.mat` for one-shot test. The pre-registration and development freeze were written before test target access.\n\n- C route: `{c_route}`\n- W route: `{w_route}`\n- KC RMSE: {result['KC']['rmse']:.12g}\n- KCW RMSE: {result['KCW']['rmse']:.12g}\n- Development W margin: {result['development_W_margin']:.8%}\n- Test W gain: {result['W_gain']:.8%}\n- Residual skill: {result['residual_skill']:.8%}\n- Verdict: `{result['verdict']}`\n\nNo fit or selection call occurred after external test opening.\n""", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
