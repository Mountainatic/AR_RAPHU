"""Aggregate completed E6 task/rod outputs without inventing missing coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_mode(root: Path, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    aggregate = []
    provenance = []
    mode_root = root / mode
    if not mode_root.is_dir():
        return per_seed, aggregate, provenance
    for path in sorted(mode_root.glob("*/per_seed.csv")):
        per_seed.extend(_read_csv(path))
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    for path in sorted(mode_root.glob("*/aggregate.csv")):
        aggregate.extend(_read_csv(path))
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    return per_seed, aggregate, provenance


def build_report(root: Path) -> dict[str, Any]:
    n1_rows, n1_aggregate, n1_provenance = _collect_mode(root, "N1")
    n2_rows, n2_aggregate, n2_provenance = _collect_mode(root, "N2")
    _write_csv(root / "N1" / "per_seed.csv", n1_rows)
    _write_csv(root / "N1" / "aggregate.csv", n1_aggregate)
    _write_csv(root / "N2" / "per_seed.csv", n2_rows)
    _write_csv(root / "N2" / "aggregate.csv", n2_aggregate)
    structural: dict[str, list[dict[str, Any]]] = {
        "channel_stability.csv": [],
        "scale_stability.csv": [],
        "stage_stability.csv": [],
    }
    structural_provenance = []
    for name in structural:
        for path in sorted((root / "N2").glob(f"*/{name}")):
            structural[name].extend(_read_csv(path))
            structural_provenance.append({"path": str(path), "sha256": _sha256(path)})
        _write_csv(root / "N2" / name, structural[name])

    n1_tasks = sorted({str(row.get("task")) for row in n1_rows})
    n2_tasks = sorted({str(row.get("task")) for row in n2_rows})
    n1_perturbations = {str(row.get("perturbation")) for row in n1_rows}
    n2_perturbations = {str(row.get("perturbation")) for row in n2_rows}
    n1_process_only = any(
        row.get("perturbation") == "gaussian_process_only"
        and row.get("measurement_scope", "process_only") == "process_only"
        for row in n1_rows
    )
    n2_process_only = any(
        row.get("perturbation") == "gaussian_process_only"
        and row.get("measurement_scope", "process_only") == "process_only"
        for row in n2_rows
    )
    n1_realistic = any(
        row.get("perturbation") == "gaussian_process_only"
        and row.get("measurement_scope", "process_only") == "realistic_dynamic"
        for row in n1_rows
    )
    n2_realistic = any(
        row.get("perturbation") == "gaussian_process_only"
        and row.get("measurement_scope", "process_only") == "realistic_dynamic"
        for row in n2_rows
    )
    coverage = {
        "gaussian_process_only": {
            "N1": "COMPLETED" if n1_process_only else "NOT_RUN",
            "N2": "COMPLETED" if n2_process_only else "NOT_RUN",
            "N1_tasks": n1_tasks,
            "N2_tasks": n2_tasks,
        },
        "gaussian_realistic_dynamic": {
            "N1": "COMPLETED" if n1_realistic else "NOT_RUN",
            "N2": "COMPLETED" if n2_realistic else "NOT_RUN",
        },
        "bias": {"N1": "COMPLETED" if "bias" in n1_perturbations else "NOT_RUN", "N2": "NOT_RUN"},
        "linear_drift": {"N1": "COMPLETED" if "linear_drift" in n1_perturbations else "NOT_RUN", "N2": "NOT_RUN"},
        "random_walk_drift": {"N1": "COMPLETED" if "random_walk_drift" in n1_perturbations else "NOT_RUN", "N2": "NOT_RUN"},
        "quantization": {"N1": "COMPLETED" if "quantization" in n1_perturbations else "NOT_RUN", "N2": "NOT_RUN"},
        "dropout_optional": {"N1": "NOT_RUN", "N2": "NOT_RUN"},
    }
    required_tasks = {"CZ_H4", "TEP_H0"}
    status = (
        "COMPLETED"
        if required_tasks.issubset(n1_tasks) and required_tasks.issubset(n2_tasks)
        else "PARTIAL"
    )
    manifest = {
        "status": status,
        "coverage": coverage,
        "N1_case_count": len(n1_rows),
        "N2_case_count": len(n2_rows),
        "N1_structure_evaluated": False,
        "N2_structure_evaluated": bool(n2_rows),
        "source_files": [*n1_provenance, *n2_provenance, *structural_provenance],
    }
    _write_json(root / "perturbation_manifest.json", manifest)
    interpretation = (
        "# E6 Measurement Perturbation Robustness\n\n"
        f"Status: `{status}`.\n\n"
        "N1 is frozen-model inference and is used only for prediction robustness. "
        "N2 repeats full structure identification and reports prediction plus "
        "channel/scale/stage robustness. Gaussian process-only noise is injected at "
        "the raw aligned measurement level, uses outer-train sigma, preserves missing "
        "values, keeps the target/reference clean, and uses nested alpha realizations.\n\n"
        "Coverage not present in the machine-readable manifest is `NOT_RUN`; claims "
        "about realistic noisy target history/anchor, bias, drift, quantization, or "
        "dropout are made only when the corresponding rows are present.\n"
    )
    (root / "README.md").write_text(interpretation, encoding="utf-8")
    (root / "E6_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    _write_json(root / "E6_REPORT_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
