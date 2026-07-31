#!/usr/bin/env python3
"""Evaluate finalist input-noise and missing-control ablations without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpu_common import atomic_npz, regression_metrics, write_csv
from src.gpu_data import load_direction, resolve_shared_root
from src.gpu_models import build_model
from src.gpu_runner import TaskSpec, _task_arrays
from src.gpu_training import predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cpu-results", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    shared = resolve_shared_root(args.shared)
    results = Path(args.results).expanduser().resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    noise_fraction = float(config["ablation_protocol"]["input_noise_std_fraction"])
    device = torch.device(args.device)
    rows = []
    for raw in config["models"]:
        task = TaskSpec(
            model_id=raw["id"],
            architecture=raw["architecture"],
            mode=raw["mode"],
            stage="finalists",
            label=raw["label"],
            parameters=dict(raw.get("parameters", {})),
            batch_size=int(raw.get("batch_size", config["training"]["batch_size"])),
            max_parameters=int(raw.get("max_parameters", config["max_parameters"])),
        )
        for direction_name in ("sheet1_to_sheet2", "sheet2_to_sheet1"):
            direction = load_direction(shared, direction_name, validate=False)
            arrays = _task_arrays(
                direction,
                task,
                cpu_results_root=Path(args.cpu_results),
                residual_history_rows=int(config["residual_history_rows"]),
            )
            for seed_root in sorted(
                (results / "tasks" / "finalists" / direction_name / task.model_id).glob("seed_*")
            ):
                seed = int(seed_root.name.split("_")[-1])
                record = json.loads((seed_root / "result.json").read_text(encoding="utf-8"))
                if record.get("status") != "PASS":
                    continue
                mean = np.asarray(record["x_scaler_mean"], dtype=np.float32)[None, None, :]
                scale = np.asarray(record["x_scaler_scale"], dtype=np.float32)[None, None, :]
                x_scaled = ((arrays["x_test"] - mean) / scale).astype(np.float32)
                model = build_model(
                    task.architecture,
                    sequence_length=x_scaled.shape[1],
                    input_dim=x_scaled.shape[2],
                    parameters=task.parameters,
                ).to(device)
                checkpoint = torch.load(
                    seed_root / "best_model.pt", map_location=device, weights_only=True
                )
                model.load_state_dict(checkpoint["state_dict"])
                perturbations: dict[str, np.ndarray] = {}
                if task.mode != "residual":
                    token = f"{direction_name}:{task.model_id}:{seed}".encode()
                    rng_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
                    rng = np.random.default_rng(rng_seed)
                    noisy = x_scaled.copy()
                    noisy[..., :4] += rng.normal(
                        0.0, noise_fraction, size=noisy[..., :4].shape
                    ).astype(np.float32)
                    perturbations[f"input_noise_{noise_fraction:g}"] = noisy
                    for channel in range(4):
                        missing = x_scaled.copy()
                        missing[..., channel] = 0.0
                        perturbations[f"missing_control_{channel}"] = missing
                for ablation, values in perturbations.items():
                    predicted_scaled, inference_ms = predict(
                        model,
                        values,
                        device=device,
                        batch_size=max(256, task.batch_size),
                        num_workers=0,
                    )
                    target_scale = float(record["target_scaler"]["scale"])
                    target_mean = float(record["target_scaler"]["mean"])
                    correction = predicted_scaled.astype(np.float64) * target_scale + target_mean
                    prediction = arrays["base_test"] + correction
                    mask = direction.test.evaluation_mask & arrays["test_available"]
                    metrics = regression_metrics(
                        direction.test.target_z[mask], prediction[mask]
                    )
                    output = (
                        results
                        / "ABLATIONS"
                        / "inference"
                        / direction_name
                        / task.model_id
                        / seed_root.name
                        / f"{ablation}.npz"
                    )
                    atomic_npz(
                        output,
                        sample_id=direction.test.sample_id,
                        y_true=direction.test.target_z,
                        y_pred=prediction.astype(np.float32),
                        evaluation_mask=mask,
                    )
                    rows.append(
                        {
                            "direction": direction_name,
                            "model_id": task.model_id,
                            "mode": task.mode,
                            "seed": seed,
                            "ablation": ablation,
                            "evaluation_rows": int(mask.sum()),
                            "MSE": metrics["MSE"],
                            "RMSE": metrics["RMSE"],
                            "MAE": metrics["MAE"],
                            "R2": metrics["R2"],
                            "infer_ms_per_1000": inference_ms,
                        }
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    write_csv(results / "ABLATIONS" / "INFERENCE_ABLATIONS.csv", rows)
    print(
        "GPU_INFERENCE_ABLATIONS="
        + json.dumps({"status": "PASS", "rows": len(rows)}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
