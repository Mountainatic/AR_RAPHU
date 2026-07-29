#!/usr/bin/env python3
"""Run the frozen CZ development baseline matrix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch
from torch import nn

from ar_raphu.cz_real.linear import (
    TrainScaler,
    fit_ols,
    regression_metrics,
    target_indices,
    window_designs,
)
from ar_raphu.cz_real.protocol import build_development_folds, load_furnace_a
from ar_raphu.orss.diagnostics import write_json


class MLPBaseline(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class GRUBaseline(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(values)
        return self.readout(sequence[:, -1]).squeeze(-1)


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _compressed(
    x: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    taps: int,
) -> np.ndarray:
    origins = targets - horizon
    x_taps = np.unique(
        np.rint(np.linspace(0, L_x - 1, taps)).astype(np.int64)
    )
    y_taps = np.unique(
        np.rint(np.linspace(0, L_y - 1, taps)).astype(np.int64)
    )
    x_values = x[origins[:, None] - x_taps[None, :]]
    x_values = x_values.transpose(0, 2, 1).reshape(len(targets), -1)
    y_values = y[origins[:, None] - y_taps[None, :]]
    return np.column_stack((x_values, y_values))


def _sequence(
    x: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    taps: int,
) -> np.ndarray:
    origins = targets - horizon
    fraction = np.linspace(1.0, 0.0, taps)
    x_offsets = np.rint(fraction * (L_x - 1)).astype(np.int64)
    y_offsets = np.rint(fraction * (L_y - 1)).astype(np.int64)
    x_values = x[origins[:, None] - x_offsets[None, :]]
    y_values = y[origins[:, None] - y_offsets[None, :], None]
    return np.concatenate((x_values, y_values), axis=2)


def _standardize(
    train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    scale = train.std(axis=axes, keepdims=True)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, (validation - mean) / scale


def _polynomial_degree_two(values: np.ndarray) -> np.ndarray:
    columns = [values]
    left, right = np.triu_indices(values.shape[1])
    columns.append(values[:, left] * values[:, right])
    return np.column_stack(columns)


def _ridge_fit(
    train: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(train)), train))
    val_design = np.column_stack((np.ones(len(validation)), validation))
    gram = design.T @ design / len(design)
    rhs = design.T @ target / len(design)
    penalty = ridge * np.eye(len(gram))
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return val_design @ coefficients


def _train_torch(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    config: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    y_mean = float(np.mean(train_y))
    y_scale = float(np.std(train_y)) or 1.0
    train_features = torch.as_tensor(
        train_x, device=device, dtype=torch.float32
    )
    train_target = torch.as_tensor(
        (train_y - y_mean) / y_scale,
        device=device,
        dtype=torch.float32,
    )
    validation_features = torch.as_tensor(
        validation_x, device=device, dtype=torch.float32
    )
    validation_target = torch.as_tensor(
        (validation_y - y_mean) / y_scale,
        device=device,
        dtype=torch.float32,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    batch_size = int(config["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs_run = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        order = torch.randperm(
            len(train_features), generator=generator
        ).to(device)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_features[indices])
            loss = torch.mean((prediction - train_target[indices]) ** 2)
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        if epoch % int(config["validation_interval"]) == 0:
            model.eval()
            with torch.no_grad():
                loss = float(
                    torch.mean(
                        (
                            model(validation_features)
                            - validation_target
                        )
                        ** 2
                    ).item()
                )
            if loss < best_loss - 1.0e-10:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += int(config["validation_interval"])
            if stale >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = (
            model(validation_features).detach().cpu().numpy() * y_scale
            + y_mean
        )
    return prediction, {
        "epochs_run": epochs_run,
        "best_validation_scaled_MSE": best_loss,
        "seed": seed,
    }


def _fold(
    data,
    *,
    fold,
    horizon: int,
    L_x: int,
    L_y: int,
    config: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    train_indices = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    validation_indices = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    scaler = TrainScaler.fit(
        data.inputs, data.target, fold.effective_train_stop
    )
    train_x, train_ar = window_designs(
        data.inputs,
        data.target,
        targets=train_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        scaler=scaler,
    )
    val_x, val_ar = window_designs(
        data.inputs,
        data.target,
        targets=validation_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        scaler=scaler,
    )
    train_target_scaled = (
        data.target[train_indices] - scaler.y_mean
    ) / scaler.y_scale
    validation_target = data.target[validation_indices]
    ar = fit_ols(train_ar, train_target_scaled)
    x_only = fit_ols(train_x, train_target_scaled)
    arx = fit_ols(
        np.column_stack((train_x, train_ar)), train_target_scaled
    )
    predictions = {
        "Mean": np.full(len(validation_indices), scaler.y_mean),
        "Persistence": data.target[validation_indices - horizon],
        "AR": scaler.y_mean + scaler.y_scale * ar.predict(val_ar),
        "X_only_FIR": scaler.y_mean
        + scaler.y_scale * x_only.predict(val_x),
        "ARX": scaler.y_mean
        + scaler.y_scale
        * arx.predict(np.column_stack((val_x, val_ar))),
    }
    train_compressed = _compressed(
        data.inputs,
        data.target,
        train_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        taps=int(config["compressed_lag_taps"]),
    )
    val_compressed = _compressed(
        data.inputs,
        data.target,
        validation_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        taps=int(config["compressed_lag_taps"]),
    )
    train_compressed, val_compressed = _standardize(
        train_compressed, val_compressed
    )
    train_poly = _polynomial_degree_two(train_compressed)
    val_poly = _polynomial_degree_two(val_compressed)
    predictions["pNARX"] = _ridge_fit(
        train_poly,
        data.target[train_indices],
        val_poly,
        ridge=float(config["pNARX"]["ridge_weight"]),
    )
    predictions["MLP_NARX"], mlp_diagnostics = _train_torch(
        MLPBaseline(
            train_compressed.shape[1],
            int(config["MLP_NARX"]["hidden"]),
        ),
        train_compressed,
        data.target[train_indices],
        val_compressed,
        validation_target,
        config=config["MLP_NARX"],
        device=device,
    )
    train_sequence = _sequence(
        data.inputs,
        data.target,
        train_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        taps=int(config["sequence_taps"]),
    )
    val_sequence = _sequence(
        data.inputs,
        data.target,
        validation_indices,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        taps=int(config["sequence_taps"]),
    )
    train_sequence, val_sequence = _standardize(
        train_sequence, val_sequence
    )
    predictions["GRU"], gru_diagnostics = _train_torch(
        GRUBaseline(
            train_sequence.shape[2],
            int(config["GRU"]["hidden"]),
        ),
        train_sequence,
        data.target[train_indices],
        val_sequence,
        validation_target,
        config=config["GRU"],
        device=device,
    )
    return {
        "fold": fold.fold,
        "metrics": {
            name: regression_metrics(validation_target, prediction)
            for name, prediction in predictions.items()
        },
        "neural_diagnostics": {
            "MLP_NARX": mlp_diagnostics,
            "GRU": gru_diagnostics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--furnace-a-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--results-root",
        default="results/cz_real_data/complete_5090",
    )
    args = parser.parse_args()
    if not args.furnace_a_only:
        raise RuntimeError("FURNACE_B_ACCESSED_BEFORE_FREEZE")
    root = Path(args.results_root)
    output = root / "R4"
    status_path = output / "R4_STATUS.json"
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("BASELINE_MATRIX_COMPLETE"):
            print("BASELINE_MATRIX_RESUMED")
            return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    continuation = json.loads(
        (root / "R3C" / "continuation_selection.json").read_text(
            encoding="utf-8"
        )
    )
    data = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    device = torch.device(config["runtime"]["device"])
    started = time.perf_counter()
    results: dict[str, object] = {}
    for horizon in config["horizons"]:
        frozen = continuation["selections"][str(horizon)]
        L_x = int(frozen["history"]["L_x"])
        L_y = int(frozen["history"]["L_y"])
        folds = [
            _fold(
                data,
                fold=fold,
                horizon=int(horizon),
                L_x=L_x,
                L_y=L_y,
                config=config,
                device=device,
            )
            for fold in build_development_folds(L_x=L_x, L_y=L_y)
        ]
        models = tuple(folds[0]["metrics"])
        aggregate = {
            model: {
                metric: {
                    "mean_across_folds": float(
                        np.mean(
                            [
                                row["metrics"][model][metric]
                                for row in folds
                            ]
                        )
                    ),
                    "SE_across_folds": float(
                        np.std(
                            [
                                row["metrics"][model][metric]
                                for row in folds
                            ],
                            ddof=1,
                        )
                        / np.sqrt(len(folds))
                    ),
                }
                for metric in folds[0]["metrics"][model]
            }
            for model in models
        }
        results[str(horizon)] = {
            "history": {"L_x": L_x, "L_y": L_y},
            "folds": folds,
            "aggregate": aggregate,
        }
        print(
            f"R4 h={horizon} AR={aggregate['AR']['RMSE_mm']['mean_across_folds']:.8g} "
            f"ARX={aggregate['ARX']['RMSE_mm']['mean_across_folds']:.8g} "
            f"GRU={aggregate['GRU']['RMSE_mm']['mean_across_folds']:.8g}",
            flush=True,
        )
        torch.cuda.empty_cache()
    payload = {
        "schema": "CZ_R4_BASELINE_MATRIX_V1",
        "status": "COMPLETED",
        "models": [
            "Mean",
            "Persistence",
            "AR",
            "X_only_FIR",
            "ARX",
            "pNARX",
            "MLP_NARX",
            "GRU",
        ],
        "optional_references": {
            "AKGNN": config["AKGNN"],
            "Stage1TargetDelayKAN": config["Stage1TargetDelayKAN"],
        },
        "results": results,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "baseline_matrix.json", payload)
    write_json(
        status_path,
        {
            "status": "COMPLETED",
            "BASELINE_MATRIX_COMPLETE": True,
            "next_stage": "FREEZE_AND_CONFIRM_FURNACE_A",
        },
    )


if __name__ == "__main__":
    main()
