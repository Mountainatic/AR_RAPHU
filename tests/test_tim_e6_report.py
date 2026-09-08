from __future__ import annotations

import csv

from experiments.tim_validation.e6_robustness.report import build_report


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_partial_report_does_not_upgrade_missing_perturbations(tmp_path) -> None:
    _csv(tmp_path / "N1" / "CZ" / "per_seed.csv", [{"task": "CZ_H4", "seed": 1}])
    _csv(tmp_path / "N1" / "CZ" / "aggregate.csv", [{"task": "CZ_H4", "metric": "RMSE"}])
    result = build_report(tmp_path)
    assert result["status"] == "PARTIAL"
    assert result["coverage"]["bias"]["N1"] == "NOT_RUN"
    assert result["N1_structure_evaluated"] is False
