from __future__ import annotations

import csv
import json

from experiments.tim_validation.paper_outputs import _e6_table, _manifest_status


def _csv(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_status_does_not_infer_completion_from_files(tmp_path) -> None:
    assert _manifest_status(tmp_path, ("MANIFEST.json",)) == "NOT_RUN"
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"status": "PARTIAL"}), encoding="utf-8"
    )
    assert _manifest_status(tmp_path, ("MANIFEST.json",)) == "PARTIAL"


def test_e6_paper_table_joins_n2_scale_agreement(tmp_path) -> None:
    base = tmp_path / "E6_MEASUREMENT_ROBUSTNESS" / "N2"
    common = {
        "task": "CZ_H4", "rod": "Rod_1_to_Rod_2", "mode": "N2",
        "perturbation": "gaussian_process_only", "measurement_scope": "process_only",
        "information_set": "dynamic", "availability_scenario": "record_time",
        "alpha": 0.1,
    }
    _csv(
        base / "aggregate.csv",
        [{**common, "metric": "channel_jaccard", "mean": 1.0}],
    )
    structural = {key: value for key, value in common.items() if key != "mode"}
    _csv(base / "scale_stability.csv", [{**structural, "scale_agreement": 0.75}])
    rows = _e6_table(tmp_path)
    assert rows[0]["channel_jaccard"] == "1.0"
    assert rows[0]["scale_agreement"] == 0.75
