#!/usr/bin/env python3
"""Audit train-fitted PB1 spline domains without loading official test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.datasets.loaders import (
    load_cascaded_tanks,
    load_pwh,
    load_silverbox,
    load_whpn,
)
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    apply_pb1_repair_v2_partition,
    load_pb1_protocol_freeze,
)
from ar_raphu.datasets.scaling import TrainOnlyStandardizer
from ar_raphu.spectral.amplitude_domain import AmplitudeDomain


LOADERS = {
    "pwh": load_pwh,
    "whpn": load_whpn,
    "cascaded_tanks": load_cascaded_tanks,
    "silverbox": load_silverbox,
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition(name: str, raw_root: Path, config: dict) -> object:
    raw = LOADERS[name](raw_root, include_test=False)
    if name in {"cascaded_tanks", "silverbox"}:
        return apply_pb1_repair_v2_partition(raw, config)
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
    return apply_pb1_development_partition(raw, freeze, whpn_audit=audit)


def _channel(values: np.ndarray, train: np.ndarray, validation: np.ndarray) -> dict:
    fit = np.asarray(values[train], dtype=np.float64)
    evaluate = np.asarray(values[validation], dtype=np.float64)
    domain = AmplitudeDomain.fit(
        fit,
        padding_fraction=0.10,
        core_quantiles=(0.01, 0.99),
    )
    mask = domain.in_domain_mask(evaluate)
    return {
        "train_min": float(np.min(fit)),
        "train_max": float(np.max(fit)),
        "fitted_lower": domain.fit_lower,
        "fitted_upper": domain.fit_upper,
        "validation_min": float(np.min(evaluate)),
        "validation_max": float(np.max(evaluate)),
        "validation_count": int(len(evaluate)),
        "validation_ood_count": int(np.count_nonzero(~mask)),
        "validation_ood_rate": domain.ood_rate(evaluate),
        "coverage_gate_passed": bool(np.all(mask)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    args = parser.parse_args()
    datasets = {}
    for name in LOADERS:
        config = _json(
            ROOT / f"configs/public_benchmarks/pb1_{name}.yaml"
        )
        dataset = _partition(name, args.raw_root, config)
        scaled = TrainOnlyStandardizer.fit(dataset).transform(dataset)
        train = scaled.split == "train"
        validation = scaled.split == "validation"
        datasets[name] = {
            "x": _channel(scaled.x[:, 0], train, validation),
            "y": _channel(scaled.y[:, 0], train, validation),
            "official_test_rows_loaded": 0,
            "official_test_access_count": 0,
        }
    payload = {
        "schema_version": "PB1_REPAIR_AMPLITUDE_COVERAGE_V2",
        "stage": "development_repair",
        "basis_domain": {
            "fit_partition": "train_only",
            "padding_fraction": 0.10,
            "core_quantiles": [0.01, 0.99],
            "silent_clipping": False,
        },
        "datasets": datasets,
        "official_test_access_count": 0,
        "status": "COMPLETED",
    }
    output = (
        ROOT
        / "results/public_benchmarks/pb1_repair_v2"
        / "amplitude_coverage_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(datasets, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
