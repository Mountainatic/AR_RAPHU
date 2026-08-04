from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_many(root: Path, pattern: str) -> list[dict]:
    values = []
    for path in sorted(root.glob(pattern)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "PASS":
            values.append(value)
    return values


def key(value: dict) -> tuple[str, str, str, str]:
    return (
        str(value["target_head"]),
        str(value["information_set"]),
        str(value["availability_scenario"]),
        str(value["proxy_policy"]),
    )


def comparison(candidate: dict, reference: dict) -> dict:
    improvements = []
    rows = []
    for item_key in sorted(set(candidate) & set(reference)):
        candidate_mse = float(candidate[item_key]["mse"])
        reference_mse = float(reference[item_key]["mse"])
        if not np.isfinite(candidate_mse) or not np.isfinite(reference_mse) or reference_mse <= 0:
            continue
        improvement = (reference_mse - candidate_mse) / reference_mse
        improvements.append(improvement)
        rows.append({"key": list(item_key), "relative_mse_improvement": improvement})
    array = np.asarray(improvements, dtype=np.float64)
    return {
        "pairs": len(array),
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(np.abs(array) <= 1e-12)),
        "losses": int(np.sum(array < 0)),
        "median_relative_mse_improvement": float(np.median(array)) if len(array) else None,
        "mean_relative_mse_improvement": float(np.mean(array)) if len(array) else None,
        "q25_relative_mse_improvement": float(np.quantile(array, 0.25)) if len(array) else None,
        "q75_relative_mse_improvement": float(np.quantile(array, 0.75)) if len(array) else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    cards = read_many(output, "ASSEMBLY_CARDS/**/ASSEMBLY_CARD.json")
    selected = {
        key(card): {
            **card["selected_prediction"],
            "dataset": card["dataset"],
            "selected_assembly": card["selected_assembly"],
        }
        for card in cards
    }
    c2 = read_many(output, "BASELINE_DEVELOPMENT/C2/PREDICTIONS/**/RESULT.json")
    c3 = read_many(output, "BASELINE_DEVELOPMENT/C3/PREDICTIONS/**/RESULT.json")
    baselines = {}
    for value in [*c2, *c3]:
        baselines.setdefault(str(value["model"]), {})[key(value)] = value
    v7_values = read_many(output, "DEVELOPMENT/JOINT_PREDICTIVE/**/RESULT.json")
    v7 = {
        (
            str(value["target_head"]), "dynamic",
            str(value["availability_scenario"]), str(value["proxy_policy"]),
        ): value
        for value in v7_values
    }
    input_selected = {item_key: value for item_key, value in selected.items() if item_key[1] == "input_only"}
    dynamic_selected = {item_key: value for item_key, value in selected.items() if item_key[1] == "dynamic"}
    best_c2 = {}
    for item_key in set(baselines.get("DPLS", {})) | set(baselines.get("XGBOOST", {})):
        candidates = [
            source[item_key]
            for source in (baselines.get("DPLS", {}), baselines.get("XGBOOST", {}))
            if item_key in source
        ]
        if candidates:
            best_c2[item_key] = min(candidates, key=lambda value: float(value["mse"]))
    selections = {}
    for card in cards:
        selections[card["selected_assembly"]] = selections.get(card["selected_assembly"], 0) + 1
    report = {
        "status": "PARTIAL_VALIDATION_ONLY",
        "test_accessed": False,
        "assembly_cards": len(cards),
        "assembly_selection_counts": selections,
        "completed_baseline_results": len(c2) + len(c3),
        "input_selected_vs_best_completed_dpls_xgboost": comparison(input_selected, best_c2),
        "dynamic_selected_vs_ar": comparison(dynamic_selected, baselines.get("AR", {})),
        "joint_predictive_v7_vs_ar": comparison(v7, baselines.get("AR", {})),
        "joint_predictive_v7_vs_selected_assembly": comparison(v7, dynamic_selected),
        "positive_r2": {
            "selected_assembly": sum(float(value.get("r2", -np.inf)) > 0 for value in selected.values()),
            "selected_assembly_total": len(selected),
            "joint_predictive_v7": sum(float(value.get("r2", -np.inf)) > 0 for value in v7.values()),
            "joint_predictive_v7_total": len(v7),
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
