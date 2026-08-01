from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from prism_cpu.config import TASKS
from prism_cpu.io import audit_frame, load_dataset, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    args = ap.parse_args()
    args.registry.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(args.dataset, args.raw_root)
    except Exception as exc:
        payload = {"dataset": args.dataset, "status": "BLOCKED_BY_MISSING_DATA", "error": f"{type(exc).__name__}: {exc}"}
        (args.registry / "FREEZE_DECISION.md").write_text(f"# {args.dataset}\n\n`BLOCKED_BY_MISSING_DATA`\n\n{exc}\n", encoding="utf-8")
        (args.registry / "AUDIT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0
    audit = audit_frame(ds)
    audit["status"] = "COMPLETED" if audit["missing_cells"] == 0 else "COMPLETED_WITH_MISSING_REPORT"
    hashes = [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in ds.source_files]
    (args.registry / "CADENCE_AUDIT.json").write_text(json.dumps(audit["cadence"], indent=2), encoding="utf-8")
    (args.registry / "MISSING_AND_DUPLICATE_AUDIT.json").write_text(json.dumps({k: audit[k] for k in ("missing_cells", "duplicate_rows", "constant_numeric_columns")}, indent=2), encoding="utf-8")
    (args.registry / "RAW_FILE_HASHES.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (args.registry / "VARIABLE_DICTIONARY.json").write_text(json.dumps({"targets": ds.target_columns, "inputs": ds.input_columns, "metadata": ds.metadata}, indent=2), encoding="utf-8")
    (args.registry / "RUN_BOUNDARIES.json").write_text(json.dumps({"run_count": audit["run_count"], "run_id_min": int(ds.run_id.min()) if len(ds.run_id) else None, "run_id_max": int(ds.run_id.max()) if len(ds.run_id) else None}, indent=2), encoding="utf-8")
    (args.registry / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.registry / "FREEZE_DECISION.md").write_text(f"# {args.dataset}\n\nStatus: `{audit['status']}`\n\nRows: {audit['rows']}\n\nSource files are kept outside return artifacts.\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

