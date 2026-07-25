"""Shared utilities for the V20 dual-solver experiments.

The functions in this module deliberately keep selection and truth-based
post-hoc evaluation separate.  Configuration selection only sees validation
metrics and support stability.  Truth is consumed after a configuration has
been fixed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    if fieldnames is None:
        ordered: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def input_ranges_from_train(x: np.ndarray, train_indices: np.ndarray, pad_fraction: float = 0.05):
    values = x[train_indices].transpose(1, 0, 2).reshape(x.shape[1], -1)
    lo = values.min(axis=1)
    hi = values.max(axis=1)
    pad = np.maximum((hi - lo) * pad_fraction, 0.05)
    return [(float(a - p), float(b + p)) for a, b, p in zip(lo, hi, pad)]


def pairwise_jaccard(supports: Iterable[Iterable[int]]) -> float:
    sets = [set(map(int, s)) for s in supports]
    if len(sets) < 2:
        return 1.0
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            values.append(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])))
    return float(np.mean(values))


def cross_seed_one_se_select(rows: Sequence[dict], *, config_key: str = "config_id") -> dict:
    """Select a configuration using validation metrics only.

    The one-standard-error tolerance is the standard error *across seeds for
    the minimum-mean configuration*.  It is never computed across different
    regularization points from one seed.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[config_key]), []).append(row)
    if not groups:
        raise ValueError("no rows supplied")
    stats = []
    for config_id, records in groups.items():
        vals = np.asarray([float(r["refit_val_rmse"]) for r in records], dtype=float)
        supports = [r["terminal_support"] for r in records]
        counts = np.asarray([len(s) for s in supports], dtype=float)
        stats.append({
            "config_id": config_id,
            "mean_val_rmse": float(vals.mean()),
            "std_val_rmse": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "se_val_rmse": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "mean_active_count": float(counts.mean()),
            "pairwise_jaccard": pairwise_jaccard(supports),
            "stable_fraction": float(np.mean([bool(r.get("stable", False)) for r in records])),
            "records": records,
        })
    minimum = min(stats, key=lambda item: item["mean_val_rmse"])
    threshold = minimum["mean_val_rmse"] + minimum["se_val_rmse"]
    eligible = [item for item in stats if item["mean_val_rmse"] <= threshold]
    selected = min(
        eligible,
        key=lambda item: (
            item["mean_active_count"],
            -item["pairwise_jaccard"],
            -item["stable_fraction"],
            item["mean_val_rmse"],
        ),
    )
    result = {k: v for k, v in selected.items() if k != "records"}
    result["one_se_reference_config"] = minimum["config_id"]
    result["one_se_threshold"] = threshold
    result["all_config_stats"] = [{k: v for k, v in item.items() if k != "records"} for item in stats]
    return result


def support_truth_metrics(support: Iterable[int], truth=(0, 1, 2)) -> dict:
    chosen, target = set(map(int, support)), set(map(int, truth))
    tp = len(chosen & target)
    precision = tp / max(1, len(chosen))
    recall = tp / max(1, len(target))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_support": chosen == target,
        "false_positives": sorted(chosen - target),
        "false_negatives": sorted(target - chosen),
    }


def torch_environment() -> dict:
    payload = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        payload.update({
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        })
    return payload
