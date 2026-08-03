from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.c6_full_final import (
    _build_report,
    _metrics_and_statistics,
    _prediction_inventory,
    _repair_freeze,
)
from prism_benchmark.cpu_data import sha256_file
from prism_benchmark.stage0 import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild C6 metrics and paired statistics from frozen predictions.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=20)
    arguments = parser.parse_args()

    config = _repair_freeze(arguments.project)
    failures = json.loads((arguments.output / "C6_FAILURES.json").read_text(encoding="utf-8"))
    metrics, bootstrap, ranks, entities = _metrics_and_statistics(arguments.output, config, arguments.n_jobs)
    metrics.to_csv(arguments.output / "CPU_FINAL_METRICS.csv", index=False)
    bootstrap.to_csv(arguments.output / "BOOTSTRAP_ALL_PAIRS.csv", index=False)
    ranks.to_csv(arguments.output / "CROSS_TASK_RANKS.csv", index=False)
    entities.to_csv(arguments.output / "ENTITY_METRICS.csv", index=False)
    _build_report(arguments.output, metrics, bootstrap, ranks, failures, config)

    manifest = arguments.output / "FULL_FINAL_FREEZE_MANIFEST.json"
    registry_rows = sum(1 for _ in (arguments.output / "FINAL_FIT_REGISTRY.csv").open(encoding="utf-8")) - 1
    decision = {
        "status": "PASS" if not failures else "PASS_WITH_RETAINED_FAILURES",
        "stage": "C6_FULL_FINAL_V2",
        "test_accessed": True,
        "successful_prediction_files": len(_prediction_inventory(arguments.output)),
        "fit_registry_rows": registry_rows,
        "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "entity_metric_rows": len(entities),
        "failure_and_not_applicable_records": len(failures),
        "full_final_freeze_sha256": sha256_file(manifest),
        "repair_config_sha256": sha256_file(arguments.project / "configs/c6_full_final_v2.json"),
        "n_jobs": int(arguments.n_jobs),
        "statistics_rebuilt_from_frozen_predictions": True,
    }
    write_json(arguments.output / "CPU_FULL_FINAL_DECISION.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
