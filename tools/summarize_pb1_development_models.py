#!/usr/bin/env python3
"""Create matched one-step and baseline-faithful PB1 development comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.baselines.arx_champneys2024 import simulate_arx
from ar_raphu.baselines.mlp_narx_champneys2024 import (
    MLPWeights,
    MinMaxScaling,
    history_design,
    predict_mlp,
    simulate_mlp_narx,
)
from ar_raphu.baselines.pnarx_champneys2024 import (
    legendre_monomial_design,
    simulate_pnarx,
)
from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)


LOADERS = {"pwh": load_pwh, "whpn": load_whpn}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(name: str, raw_root: Path):
    freeze = load_pb1_protocol_freeze(
        ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    )
    audit = (
        _json(
            ROOT
            / "results/public_benchmarks/pb1/protocol_audit"
            / "whpn_realization_audit.json"
        )
        if name == "whpn"
        else None
    )
    return apply_pb1_development_partition(
        LOADERS[name](raw_root, include_test=False),
        freeze,
        whpn_audit=audit,
    )


def _validation_records(dataset):
    records = []
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        if str(dataset.split[indices[0]]) == "validation":
            records.append(
                (
                    str(sequence),
                    np.asarray(dataset.x[indices, 0], dtype=np.float64),
                    np.asarray(dataset.y[indices, 0], dtype=np.float64),
                )
            )
    return records


def _load_weights(path: Path) -> MLPWeights:
    values = np.load(path)
    return MLPWeights(
        hidden_weight=values["hidden_weight"],
        hidden_bias=values["hidden_bias"],
        output_weight=values["output_weight"],
        output_bias=float(values["output_bias"]),
    )


def _metric(scores: list[float], counts: list[int]) -> dict[str, object]:
    pooled = float(
        np.sum(np.asarray(scores) * np.asarray(counts)) / np.sum(counts)
    )
    return {
        "mse_by_record": scores,
        "mse_record_mean": float(np.mean(scores)),
        "mse_pooled": pooled,
        "rmse_pooled": float(np.sqrt(pooled)),
        "record_count": len(scores),
        "sample_count": int(np.sum(counts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(LOADERS), required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument("--force-development", action="store_true")
    args = parser.parse_args()

    base = (
        ROOT / "results/public_benchmarks/pb1" / args.dataset / "development"
    )
    arx = _json(
        base / "H1_ARX_NO_FUTURE_X/history_selection.json"
    )
    pnarx = _json(
        base / "PNARX_CHAMPNEYS2024/order_selection.json"
    )
    mlp = _json(
        base / "MLPNARX_CHAMPNEYS2024/selection.json"
    )
    spectral = _json(
        base
        / "H3_SHARED_HISTORY/SPECTRAL_PILOT_H1/full_spectral.json"
    )
    dataset = _dataset(args.dataset, args.raw_root)
    records = _validation_records(dataset)
    nx, ny = int(arx["selected"]["nx"]), int(arx["selected"]["ny"])
    burn = max(nx, ny)

    x_mean = float(arx["selected"]["x_mean"])
    x_scale = float(arx["selected"]["x_scale"])
    y_mean = float(arx["selected"]["y_mean"])
    y_scale = float(arx["selected"]["y_scale"])
    arx_y = np.asarray(
        arx["selected"]["coefficients_y_standardized"], dtype=np.float64
    )
    arx_x = np.asarray(
        arx["selected"]["coefficients_x_standardized"], dtype=np.float64
    )
    pnarx_order = int(pnarx["selected"]["order"])
    pnarx_coefficients = np.asarray(
        pnarx["selected"]["coefficients_standardized"], dtype=np.float64
    )
    mlp_scaling = MinMaxScaling(
        **{
            key: float(value)
            for key, value in _json(
                ROOT / mlp["selected"]["metadata_file"]
            )["scaling"].items()
        }
    )
    mlp_weights = _load_weights(ROOT / mlp["selected"]["weights_file"])

    direct: dict[str, list[float]] = {
        name: [] for name in ("persistence", "arx", "pnarx", "mlpnarx")
    }
    free: dict[str, list[float]] = {
        name: [] for name in ("arx", "pnarx", "mlpnarx")
    }
    counts: list[int] = []
    record_ids: list[str] = []
    for sequence, x_raw, y_raw in records:
        record_ids.append(sequence)
        x_z = (x_raw - x_mean) / x_scale
        y_z = (y_raw - y_mean) / y_scale
        histories_z, target_z = history_design(x_z, y_z, nx=nx, ny=ny)
        counts.append(len(target_z))
        persistence = y_z[burn - 1 : -1]
        arx_direct = histories_z[:, :ny] @ arx_y + histories_z[:, ny:] @ arx_x
        pnarx_direct = (
            legendre_monomial_design(histories_z, pnarx_order)
            @ pnarx_coefficients
        )
        x_mlp = mlp_scaling.x_transform(x_raw)
        y_mlp = mlp_scaling.y_transform(y_raw)
        histories_mlp, _ = history_design(x_mlp, y_mlp, nx=nx, ny=ny)
        mlp_direct_scaled = np.asarray(
            [predict_mlp(row, mlp_weights) for row in histories_mlp]
        )
        mlp_direct_raw = (
            (mlp_direct_scaled + 1.0)
            * 0.5
            * (mlp_scaling.y_max - mlp_scaling.y_min)
            + mlp_scaling.y_min
        )
        mlp_direct = (mlp_direct_raw - y_mean) / y_scale
        for name, prediction in (
            ("persistence", persistence),
            ("arx", arx_direct),
            ("pnarx", pnarx_direct),
            ("mlpnarx", mlp_direct),
        ):
            direct[name].append(float(np.mean((target_z - prediction) ** 2)))

        arx_free, _ = simulate_arx(
            x_z,
            y_z,
            coefficients_y=arx_y,
            coefficients_x=arx_x,
        )
        pnarx_free, _ = simulate_pnarx(
            x_z,
            y_z,
            nx=nx,
            ny=ny,
            order=pnarx_order,
            coefficients=pnarx_coefficients,
        )
        mlp_free_scaled, _ = simulate_mlp_narx(
            x_mlp,
            y_mlp,
            nx=nx,
            ny=ny,
            weights=mlp_weights,
        )
        mlp_free_raw = (
            (mlp_free_scaled + 1.0)
            * 0.5
            * (mlp_scaling.y_max - mlp_scaling.y_min)
            + mlp_scaling.y_min
        )
        mlp_free = (mlp_free_raw - y_mean) / y_scale
        for name, prediction in (
            ("arx", arx_free),
            ("pnarx", pnarx_free),
            ("mlpnarx", mlp_free),
        ):
            free[name].append(
                float(np.mean((target_z - prediction[burn:]) ** 2))
            )

    direct_metrics = {
        name: _metric(scores, counts) for name, scores in direct.items()
    }
    free_metrics = {
        name: _metric(scores, counts) for name, scores in free.items()
    }
    persistence_mse = direct_metrics["persistence"]["mse_pooled"]
    spectral_rank_mse = spectral["rank_audit_after_penalty_freeze"][
        "rank_validation_mse"
    ]
    spectral_metrics = {
        "full": {
            "mse_pooled": spectral["rank_audit_after_penalty_freeze"][
                "full_validation_mse"
            ]
        },
        "rank1": {"mse_pooled": spectral_rank_mse[0]},
        "rank2": {"mse_pooled": spectral_rank_mse[1]},
        "adaptive_primary_5pct": {
            "rank": spectral["rank_audit_after_penalty_freeze"][
                "predictive_effective_ranks"
            ]["0.05"],
        },
    }
    for metrics in direct_metrics.values():
        metrics["skill_vs_persistence"] = float(
            1.0 - metrics["mse_pooled"] / persistence_mse
        )
    for metrics in spectral_metrics.values():
        if "mse_pooled" in metrics:
            metrics["rmse_pooled"] = float(np.sqrt(metrics["mse_pooled"]))
            metrics["skill_vs_persistence"] = float(
                1.0 - metrics["mse_pooled"] / persistence_mse
            )
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "dataset": args.dataset,
        "stage": "development",
        "horizon": 1,
        "history": {"L_x": nx, "L_y": ny, "source": "H1_ARX_AIC"},
        "matched_record_ids": record_ids,
        "matched_information_set": "X_THROUGH_T_AND_Y_THROUGH_T_TO_TARGET_Y_T_PLUS_1",
        "target_scale": "TRAIN_ONLY_Z_SCORE",
        "direct_protocol_metrics": direct_metrics,
        "baseline_faithful_free_running_metrics": free_metrics,
        "spectral_h3_pilot_metrics": spectral_metrics,
        "selection_boundaries": {
            "arx_pnarx_mlpnarx": "BASELINE_FAITHFUL_VALIDATION_AIC",
            "spectral": "GROUPED_VALIDATION_ONE_SE_AFTER_NORMALIZED_PENALTY_SEARCH",
            "official_test_used": False,
        },
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "status": "COMPLETED",
    }
    output = base / "DEVELOPMENT_MODEL_COMPARISON_H1.json"
    if output.exists() and not args.force_development:
        raise FileExistsError(f"{output} already exists.")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    for name, metrics in direct_metrics.items():
        print(name, metrics["rmse_pooled"], metrics["skill_vs_persistence"])
    for name, metrics in spectral_metrics.items():
        if "rmse_pooled" in metrics:
            print(name, metrics["rmse_pooled"], metrics["skill_vs_persistence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
