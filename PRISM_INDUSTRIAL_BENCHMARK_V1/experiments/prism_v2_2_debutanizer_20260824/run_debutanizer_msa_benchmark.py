from __future__ import annotations

import csv
import json
import math
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np

# This benchmark is intentionally standalone and lives on a benchmark-only branch.
# It does not modify the frozen PRISM v2.2(beta) theory/code line.

DATA_URL = (
    "https://raw.githubusercontent.com/Ujjwal-1267/industrial-debutanizer-soft-sensor/"
    "main/data/debutanizer_data.txt"
)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Debutanizer sampling period reported by Patane et al. (Electronics 2024): 6 min.
DT_SECONDS = 360.0
HORIZONS = (2, 5, 10, 20)  # 12, 30, 60, 120 min
# Pre-registered, dataset-agnostic dyadic time-scale rule: 1,2,4,8,16,32 samples.
TAU_STEPS = (1, 2, 4, 8, 16, 32)
TAUS_SECONDS = tuple(DT_SECONDS * step for step in TAU_STEPS)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CONDITION_LIMIT = 1e8
ASSEMBLY_RIDGE = 1e-3

PUBLISHED = {
    "ARX": {2: 0.983, 5: 0.854, 10: 0.347, 20: -1.20},
    "FIR": {2: -0.99, 5: -0.99, 10: -0.99, 20: -0.99},
    "MSA-HDMDc": {2: 0.998, 5: 0.974, 10: 0.890, 20: 0.661},
}


def load_data() -> np.ndarray:
    destination = OUT / "debutanizer_data.txt"
    urllib.request.urlretrieve(DATA_URL, destination)
    rows = []
    for raw in destination.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().replace(",", " ").split()
        if len(parts) != 8:
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            rows.append(values)
    array = np.asarray(rows, dtype=np.float64)
    if array.shape != (2394, 8):
        raise RuntimeError(f"expected (2394,8), got {array.shape}")
    return array


def stable_states_segment(x: np.ndarray) -> np.ndarray:
    taus = np.asarray(TAUS_SECONDS, dtype=np.float64)
    a = np.exp(-DT_SECONDS / taus)
    z = np.empty((len(x), x.shape[1], len(taus)), dtype=np.float64)
    z[0] = x[0, :, None]
    for t in range(1, len(x)):
        z[t] = a[None, :] * z[t - 1] + (1.0 - a)[None, :] * x[t, :, None]
    return z


def build_segment_features(segment: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    # State bank uses the 7 easy-to-measure inputs plus causally available past y.
    x = segment
    z = stable_states_segment(x)
    max_lag = max(TAU_STEPS)
    indices = np.arange(max_lag, len(x), dtype=np.int64)

    delay_blocks = [x[indices]]
    for lag in TAU_STEPS:
        delay_blocks.append(x[indices - lag])
    delay = np.concatenate(delay_blocks, axis=1)

    absolute = np.concatenate([z[indices, :, r] for r in range(z.shape[2])], axis=1)
    multires_blocks = [x[indices] - z[indices, :, 0]]
    multires_blocks.extend(
        z[indices, :, r] - z[indices, :, r + 1]
        for r in range(z.shape[2] - 1)
    )
    multires = np.concatenate(multires_blocks, axis=1)
    return {
        "delay": delay,
        "ct_multires": multires,
        "ct_absolute": absolute,
    }, indices


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def condition_number(x: np.ndarray) -> tuple[float, int]:
    mean, scale = standardize_fit(x)
    xs = (x - mean) / scale
    s = np.linalg.svd(xs, compute_uv=False)
    if not len(s):
        return math.inf, 0
    tol = np.finfo(np.float64).eps * max(xs.shape) * s[0]
    rank = int(np.sum(s > tol))
    cond = math.inf if rank < x.shape[1] or s[-1] <= tol else float(s[0] / s[-1])
    return cond, rank


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float):
    mean, scale = standardize_fit(x)
    xs = (x - mean) / scale
    intercept = float(y.mean())
    yc = y - intercept
    alpha = lam * len(x)
    coef = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ yc)
    return mean, scale, coef, intercept


def ridge_predict(model, x: np.ndarray) -> np.ndarray:
    mean, scale, coef, intercept = model
    return ((x - mean) / scale) @ coef + intercept


def r2(y: np.ndarray, p: np.ndarray) -> float:
    den = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / den) if den > 0 else math.nan


def metrics(y: np.ndarray, p: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y - p) ** 2))
    p_mse = float(np.mean((y - persistence) ** 2))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(y - p))),
        "mape_percent": float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-12))) * 100.0),
        "r2": r2(y, p),
        "persistence_r2": r2(y, persistence),
        "persistence_skill_mse": float(1.0 - mse / p_mse) if p_mse > 0 else math.nan,
    }


def project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.flatnonzero(u * (np.arange(len(u)) + 1) > (cssv - 1.0))
    if not len(rho_candidates):
        return np.full_like(v, 1.0 / len(v))
    rho = int(rho_candidates[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def fit_assembly(predictions: dict[str, np.ndarray], y: np.ndarray):
    names = tuple(sorted(predictions))
    # Final zero column is persistence in delta space. Here predictions are y levels,
    # so the persistence level is supplied explicitly by caller as a named branch.
    matrix = np.column_stack([predictions[name] for name in names])
    dim = matrix.shape[1]
    penalty = ASSEMBLY_RIDGE * np.eye(dim)
    hessian = (matrix.T @ matrix) / len(y) + penalty
    linear = (matrix.T @ y) / len(y)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    step = 1.0 / max(2.0 * lipschitz, 1e-12)
    w = np.full(dim, 1.0 / dim)
    for _ in range(50000):
        new = project_simplex(w - step * 2.0 * (hessian @ w - linear))
        if np.linalg.norm(new - w) < 1e-12:
            w = new
            break
        w = new
    return names, w


def select_lambda(x_train, y_train, x_val, y_val):
    records = []
    for lam in RIDGE_GRID:
        model = ridge_fit(x_train, y_train, lam)
        pred = ridge_predict(model, x_val)
        records.append((float(np.mean((y_val - pred) ** 2)), lam, model))
    return min(records, key=lambda item: (item[0], -item[1]))


def run() -> list[dict]:
    data = load_data()
    u = data[:, :7]
    y = data[:, 7]
    state = np.column_stack([u, y])
    n = len(data)
    train_end = n // 2  # 1197
    val_end = int(round(0.75 * n))  # 1796; final 598 samples are test
    boundaries = (0, train_end, val_end, n)

    # Reset all CT states and lag support at month/split boundaries; no history is carried
    # from March/May training into July validation or September test.
    segment_features = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        features, local_indices = build_segment_features(state[a:b])
        segment_features.append((a, b, features, local_indices))

    results = []
    for horizon in HORIZONS:
        prepared = []
        for a, b, features, local_indices in segment_features:
            valid = local_indices + horizon < (b - a)
            idx = local_indices[valid]
            block = {name: values[valid] for name, values in features.items()}
            target = y[a + idx + horizon]
            persistence = y[a + idx]
            prepared.append((block, target, persistence, a + idx))

        train_block, y_train, p_train, ids_train = prepared[0]
        val_block, y_val, p_val, ids_val = prepared[1]
        test_block, y_test, p_test, ids_test = prepared[2]

        branch_models = {}
        branch_val_preds = {}
        branch_test_preds = {}
        audits = {}
        for name in ("delay", "ct_multires", "ct_absolute"):
            cond, rank = condition_number(train_block[name])
            audits[name] = {
                "condition_number": cond,
                "rank": rank,
                "n_features": int(train_block[name].shape[1]),
                "admissible": bool(math.isfinite(cond) and cond <= CONDITION_LIMIT and rank == train_block[name].shape[1]),
            }
            if not audits[name]["admissible"]:
                continue
            val_mse, lam, model = select_lambda(train_block[name], y_train, val_block[name], y_val)
            # Refit branch on train only. Validation is reserved for Gamma_CT weights,
            # so branch parameters never see validation targets after lambda selection.
            # The lambda choice itself is a validation hyperparameter, as in published protocol.
            branch_models[name] = {"lambda": lam, "model": model, "validation_mse": val_mse}
            branch_val_preds[name] = ridge_predict(model, val_block[name])
            branch_test_preds[name] = ridge_predict(model, test_block[name])

        # Persistence is a transparent fallback branch and uses no learned parameter.
        gamma_val = {**branch_val_preds, "persistence": p_val}
        gamma_test = {**branch_test_preds, "persistence": p_test}
        names, weights = fit_assembly(gamma_val, y_val)
        final_test = np.column_stack([gamma_test[name] for name in names]) @ weights
        final_val = np.column_stack([gamma_val[name] for name in names]) @ weights

        record = {
            "horizon_steps": horizon,
            "horizon_minutes": horizon * 6,
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "tau_steps": list(TAU_STEPS),
            "tau_minutes": [step * 6 for step in TAU_STEPS],
            "audits": audits,
            "selected_lambdas": {name: float(item["lambda"]) for name, item in branch_models.items()},
            "gamma_names": list(names),
            "gamma_weights": [float(v) for v in weights],
            "validation": metrics(y_val, final_val, p_val),
            "test": metrics(y_test, final_test, p_test),
            "published_r2": {name: values[horizon] for name, values in PUBLISHED.items()},
            "r2_gap_vs_msa_hdmdc": float(metrics(y_test, final_test, p_test)["r2"] - PUBLISHED["MSA-HDMDc"][horizon]),
        }
        # Descriptive branch tests: never used for branch selection.
        record["branch_test_descriptive"] = {
            name: metrics(y_test, pred, p_test) for name, pred in gamma_test.items()
        }
        results.append(record)

    return results


def main() -> None:
    results = run()
    payload = {
        "benchmark": "Fortuna Debutanizer / Patane et al. 2024 MSA protocol approximation",
        "protocol": {
            "sampling_minutes": 6,
            "split": "chronological 50% train / 25% validation / 25% test, CT state reset at boundaries",
            "future_process_inputs_used": False,
            "target_history_used": True,
            "target_file_note": "public dataset y is already translated by 8 samples to compensate analyzer delay",
            "tau_rule": "fixed dyadic 1,2,4,8,16,32 sample time scales; no learned tau",
            "horizons_steps": list(HORIZONS),
            "condition_limit": CONDITION_LIMIT,
            "ridge_grid": list(RIDGE_GRID),
        },
        "published_reference": {
            "paper": "Patane & Sapuppo, Soft Sensors for Industrial Processes Using Multi-Step-Ahead Hankel Dynamic Mode Decomposition with Control, Electronics 2024, 13, 3047",
            "r2": PUBLISHED,
        },
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "horizon_steps", "horizon_minutes", "prism_v22_r2", "prism_rmse", "prism_mape_percent",
            "persistence_skill_mse", "msa_hdmdc_r2", "arx_r2", "r2_gap_vs_msa_hdmdc", "gamma_weights"
        ])
        for item in results:
            writer.writerow([
                item["horizon_steps"], item["horizon_minutes"], item["test"]["r2"], item["test"]["rmse"],
                item["test"]["mape_percent"], item["test"]["persistence_skill_mse"],
                item["published_r2"]["MSA-HDMDc"], item["published_r2"]["ARX"],
                item["r2_gap_vs_msa_hdmdc"],
                json.dumps(dict(zip(item["gamma_names"], item["gamma_weights"])), ensure_ascii=False),
            ])

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
