"""Run the exact E2 logic frozen in the authoritative strict-OOF branch.

E2 in that branch is dataset-independent IDENTIFIABLE_E2_V2.  It does not
read private CZ data and it does not refit the industrial CZ K/C/W/A chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


AUTHORITY_COMMIT = "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
E2_MODULE_SHA256 = "f0d9fcc3786a265742395a2c21e07ba7cf839f703f99d7520ee5150aca8692ec"
SELECTOR_SHA256 = "cac126f10ae015d8b4771816671257a637de668fdaafec66a8bebad926cccfa9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output_root.resolve()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    module_path = project / "src" / "prism_benchmark" / "e1e6_revalidation.py"
    selector_path = project / "src" / "prism_benchmark" / "strict_oof_selection.py"
    hashes = {
        "e1e6_revalidation.py": _sha256(module_path),
        "strict_oof_selection.py": _sha256(selector_path),
    }
    expected = {
        "e1e6_revalidation.py": E2_MODULE_SHA256,
        "strict_oof_selection.py": SELECTOR_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"STOP_AUTHORITY_E2_BLOB_MISMATCH:{hashes}")
    if (output / "E2_SYNTHETIC").exists():
        raise RuntimeError("STOP_AUTHORITY_E2_REFUSE_EXISTING_OUTPUT")

    sys.path.insert(0, str(project / "src"))
    from prism_benchmark.e1e6_revalidation import (
        SYNTHETIC_ID,
        audit_before_gate,
        phase_a_gate,
        run_e2,
    )

    if SYNTHETIC_ID != "IDENTIFIABLE_E2_V2":
        raise RuntimeError("STOP_AUTHORITY_E2_GENERATOR_MISMATCH")
    output.mkdir(parents=True, exist_ok=True)
    formal, null = run_e2(output, workers=int(args.workers))
    audit = audit_before_gate(output, formal, null)
    verdict = phase_a_gate(output, formal, null, audit)
    summary = {
        regime: {
            "seeds": int(len(group)),
            "C_active_rate": float(group["C_active"].mean()),
            "W_active_rate": float(group["W_active"].mean()),
            "A_active_rate": float(group["A_active"].mean()),
            "channel_f1_mean": float(group["channel_f1"].mean()),
            "rmse_mean": float(group["rmse"].mean()),
        }
        for regime, group in formal.groupby("regime", sort=True)
    }
    status = {
        "status": "COMPLETED" if verdict == "GO" else verdict,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_blob_sha256": hashes,
        "generator": SYNTHETIC_ID,
        "scope": "DATASET_INDEPENDENT_AUTHORITY_E2_NOT_CZ_RAW_DATA",
        "private_cz_data_accessed": False,
        "formal_target_or_ood_accessed": False,
        "workers": int(args.workers),
        "phase_a_verdict": verdict,
        "formal_summary": summary,
        "cz_e3_status_under_authority_branch": "PROTOCOL_BLOCKED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "AUTHORITY_BRANCH_E2_STATUS.json", status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
