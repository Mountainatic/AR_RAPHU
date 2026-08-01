from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from prism_cpu.config import TASKS, dump_protocol, task_registry
from prism_cpu.features import build_task_data, attach_series
from prism_cpu.io import audit_frame, load_dataset, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-cap", type=int, default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_views").mkdir(exist_ok=True)
    dump_protocol(args.output / "PROTOCOL.json", sample_cap=args.sample_cap)
    registry = {"datasets": {}, "tasks": task_registry(), "sample_cap": args.sample_cap}
    split_registry = {}
    hashes = {}
    for name in ("TEP", "Debutanizer", "SRU", "PMSM", "MetroPT-3"):
        ddir = args.output / "DATASET_REGISTRY" / name.replace("-", "_")
        ddir.mkdir(parents=True, exist_ok=True)
        try:
            ds = load_dataset(name, args.raw_root)
            audit = audit_frame(ds)
            registry["datasets"][name] = {"status": "COMPLETED", "audit": audit}
            for p in ds.source_files:
                hashes[str(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            (ddir / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
            (ddir / "RAW_FILE_HASHES.json").write_text(json.dumps([hashes[str(p)] | {"path": str(p)} for p in ds.source_files], indent=2), encoding="utf-8")
        except Exception as exc:
            registry["datasets"][name] = {"status": "BLOCKED_BY_MISSING_DATA", "error": f"{type(exc).__name__}: {exc}"}
            (ddir / "FREEZE_DECISION.md").write_text(f"# {name}\n\n`BLOCKED_BY_MISSING_DATA`\n\n{exc}\n", encoding="utf-8")
    for task in TASKS:
        try:
            ds = load_dataset(task.dataset, args.raw_root)
            data = build_task_data(ds, task, sample_cap=args.sample_cap)
            metadata = {k: v for k, v in data.metadata.items() if not isinstance(v, np.ndarray)}
            split_registry[task.task_id] = {"status": "COMPLETED", "dataset": task.dataset, "n_samples": int(len(data.y)), "counts": {s: int(np.sum(data.split == s)) for s in ("train", "validation", "test")}, "metadata": metadata}
        except Exception as exc:
            split_registry[task.task_id] = {"status": "BLOCKED_BY_MISSING_DATA", "dataset": task.dataset, "error": f"{type(exc).__name__}: {exc}"}
    (args.output / "DATASET_HASHES.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (args.output / "TASK_REGISTRY.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "SPLIT_REGISTRY.json").write_text(json.dumps(split_registry, indent=2, ensure_ascii=False), encoding="utf-8")
    blocked = [k for k, v in registry["datasets"].items() if v["status"].startswith("BLOCKED")]
    report = ["# PRISM shared dataset validation", "", f"Datasets registered: {len(registry['datasets'])}", f"Blocked datasets: {', '.join(blocked) if blocked else 'none'}", "", "The registry contains metadata and hashes only; raw files remain outside the shared return package."]
    (args.output / "VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": registry["datasets"], "blocked": blocked}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
