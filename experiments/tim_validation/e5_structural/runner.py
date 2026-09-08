"""Build E5 from comparable, already-frozen structure signatures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REQUIRED = {
    "task", "head", "H", "W", "outer_fold", "run", "seed",
    "candidate_universe", "model_variant", "admitted_channels",
    "selected_scale_class_by_channel", "K_admitted", "C_admitted",
    "W_admitted", "A_admitted", "RMSE", "support_id",
}
STAGES = ("K_admitted", "C_admitted", "W_admitted", "A_admitted")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _signature_paths(roots: Iterable[Path]) -> list[Path]:
    found: dict[Path, Path] = {}
    for root in roots:
        for path in root.rglob("STRUCTURE_SIGNATURE*.json"):
            parent = path.parent
            current = found.get(parent)
            if current is None or "COMMON_SUPPORT" in path.name:
                found[parent] = path
    return sorted(found.values())


def _view_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("information_set", "UNRECORDED")),
        str(value.get("availability_scenario", "UNRECORDED")),
        str(value.get("proxy_policy", "UNRECORDED")),
    )


def _comparable_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        value["task"], value["head"], int(value["H"]), int(value["W"]),
        *_view_key(value), value["model_variant"],
    )


def _performance_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (*_comparable_key(value), value["support_id"])


def _key_fields(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(
        zip(
            (
                "task", "head", "H", "W", "information_set",
                "availability_scenario", "proxy_policy", "model_variant",
            ),
            key,
            strict=True,
        )
    )


def _variation(value: dict[str, Any]) -> str:
    if value.get("rod") not in (None, ""):
        return "rod"
    if str(value.get("outer_fold")) not in {"registered_outer_test", "None"}:
        return "fold"
    if int(value.get("seed", 0)) != 0:
        return "seed"
    if value.get("candidate_universe") in {"coarse", "expanded"}:
        return "candidate_universe"
    return "run"


def _jaccard(left: set[str], right: set[str]) -> tuple[float, bool]:
    if not left and not right:
        return 1.0, True
    return len(left & right) / len(left | right), False


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _entropy(values: list[str]) -> tuple[float, float, int]:
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    k = len(counts)
    normalized = 0.0 if k <= 1 else entropy / math.log(k)
    return entropy, normalized, k


def build_report(input_roots: list[Path], output: Path) -> dict[str, Any]:
    paths = _signature_paths(input_roots)
    signatures: list[dict[str, Any]] = []
    reused = []
    for path in paths:
        value = _read(path)
        missing = sorted(REQUIRED - set(value))
        if missing:
            reused.append(
                {"path": str(path), "sha256": _sha256(path), "status": "INVALID", "missing": missing}
            )
            continue
        value = dict(value)
        value["_path"] = str(path)
        value["_sha256"] = _sha256(path)
        value["_variation"] = _variation(value)
        signatures.append(value)
        reused.append(
            {
                "path": str(path), "sha256": value["_sha256"],
                "status": "COMPLETED", "variation": value["_variation"],
                "comparison_key": list(_comparable_key(value)),
                "performance_comparison_key": list(_performance_key(value)),
            }
        )
    if not signatures:
        raise RuntimeError("no valid structure signatures")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for value in signatures:
        groups[_comparable_key(value)].append(value)

    channel_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    rashomon_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []

    for key, values in groups.items():
        common = _key_fields(key)
        channels = sorted(set().union(*(set(value["admitted_channels"]) for value in values)))
        for channel in channels:
            admitted = sum(channel in set(value["admitted_channels"]) for value in values)
            channel_rows.append(
                {
                    **common, "channel": channel, "runs": len(values),
                    "admitted_runs": admitted, "admission_probability": admitted / len(values),
                    "sources_of_variation": ";".join(sorted({value["_variation"] for value in values})),
                }
            )

        pair_scores = []
        for left, right in itertools.combinations(values, 2):
            left_set = set(left["admitted_channels"])
            right_set = set(right["admitted_channels"])
            score, empty_empty = _jaccard(left_set, right_set)
            pair_scores.append(score)
            jaccard_rows.append(
                {
                    **common, "left_run": left["run"], "right_run": right["run"],
                    "left_path": left["_path"], "right_path": right["_path"],
                    "jaccard": score, "agreement_type": (
                        "EMPTY_EMPTY_AGREEMENT" if empty_empty else "NONEMPTY_COMPARISON"
                    ),
                }
            )
            admitted_both = sorted(left_set & right_set)
            agreements = [
                left["selected_scale_class_by_channel"].get(channel)
                == right["selected_scale_class_by_channel"].get(channel)
                for channel in admitted_both
            ]
            scale_rows.append(
                {
                    **common, "left_run": left["run"], "right_run": right["run"],
                    "common_admitted_channels": len(admitted_both),
                    "conditional_scale_agreement": (
                        None if not agreements else sum(agreements) / len(agreements)
                    ),
                    "status": "NOT_RUN" if not agreements else "COMPLETED",
                    "reason": "NO_COMMON_ADMITTED_CHANNELS" if not agreements else "",
                }
            )
            left_vector = tuple(bool(left[name]) for name in STAGES)
            right_vector = tuple(bool(right[name]) for name in STAGES)
            stage_rows.append(
                {
                    **common, "left_run": left["run"], "right_run": right["run"],
                    "left_vector": json.dumps(left_vector), "right_vector": json.dumps(right_vector),
                    "stage_agreement": sum(a == b for a, b in zip(left_vector, right_vector)) / 4,
                    "full_vector_agreement": left_vector == right_vector,
                }
            )
        if pair_scores:
            jaccard_rows.append(
                {
                    **common, "left_run": "AGGREGATE", "right_run": "AGGREGATE",
                    "jaccard": mean(pair_scores), "median": median(pair_scores),
                    "Q1": _quantile(pair_scores, 0.25), "Q3": _quantile(pair_scores, 0.75),
                    "minimum": min(pair_scores), "agreement_type": "PAIRWISE_SUMMARY",
                }
            )

        for stage in STAGES:
            selected = [str(bool(value[stage])) for value in values]
            entropy, normalized, k = _entropy(selected)
            entropy_rows.append(
                {
                    **common, "selection": stage, "entropy": entropy,
                    "normalized_entropy": normalized, "K_observed_categories": k,
                    "K_definition": "number of observed discrete categories within comparable runs",
                }
            )
        scale_channels = sorted(
            set().union(*(set(value["selected_scale_class_by_channel"]) for value in values))
        )
        for channel in scale_channels:
            selected = [
                str(value["selected_scale_class_by_channel"][channel])
                for value in values
                if channel in set(value["admitted_channels"])
                and channel in value["selected_scale_class_by_channel"]
            ]
            if selected:
                entropy, normalized, k = _entropy(selected)
                entropy_rows.append(
                    {
                        **common, "selection": f"scale:{channel}", "entropy": entropy,
                        "normalized_entropy": normalized, "K_observed_categories": k,
                        "K_definition": "observed scales conditional on channel admission",
                    }
                )

    performance_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for value in signatures:
        performance_groups[_performance_key(value)].append(value)
    for key, values in performance_groups.items():
        common = _key_fields(key[:-1])
        common["support_id"] = key[-1]
        rmses = [float(value["RMSE"]) for value in values]
        best = min(rmses)
        performance_rows.append(
            {
                **common, "runs": len(values), "RMSE_mean": mean(rmses),
                "RMSE_median": median(rmses), "RMSE_min": best,
                "RMSE_max": max(rmses), "RMSE_Q1": _quantile(rmses, 0.25),
                "RMSE_Q3": _quantile(rmses, 0.75),
                "performance_stability_separate_from_structure": True,
            }
        )
        near = [value for value in values if float(value["RMSE"]) <= 1.01 * best]
        for value in values:
            rashomon_rows.append(
                {
                    **common, "run": value["run"], "path": value["_path"],
                    "RMSE": value["RMSE"], "RMSE_best": best,
                    "within_1_percent_of_best": value in near,
                    "statistically_indistinguishable_from_best": "NOT_RUN",
                    "statistical_reason": "PAIRED_BLOCK_RESIDUALS_UNAVAILABLE_IN_STRUCTURE_SIGNATURE",
                    "active_channel_count": value.get("active_channel_count"),
                    "active_stage_count": value.get("active_stage_count"),
                    "parameter_count": value.get("parameter_count"),
                    "stage_vector": json.dumps([bool(value[name]) for name in STAGES]),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "reused_run_manifest.json",
        {
            "status": "COMPLETED", "input_roots": [str(path) for path in input_roots],
            "signatures": reused, "valid_signature_count": len(signatures),
            "structural_comparison_rule": "same task/head/H/W/information_set/availability/proxy/model",
            "performance_comparison_rule": "structural rule plus identical support_id",
        },
    )
    for name, rows in (
        ("channel_admission.csv", channel_rows),
        ("pairwise_jaccard.csv", jaccard_rows),
        ("scale_agreement.csv", scale_rows),
        ("stage_agreement.csv", stage_rows),
        ("selection_entropy.csv", entropy_rows),
        ("rashomon_models.csv", rashomon_rows),
        ("performance_stability.csv", performance_rows),
    ):
        _write_csv(output / name, rows)

    comparable_groups = sum(len(values) >= 2 for values in groups.values())
    empty_empty = sum(row.get("agreement_type") == "EMPTY_EMPTY_AGREEMENT" for row in jaccard_rows)
    rashomon_groups = 0
    for key, values in performance_groups.items():
        if len(values) < 2:
            continue
        best = min(float(value["RMSE"]) for value in values)
        near = [value for value in values if float(value["RMSE"]) <= 1.01 * best]
        structures = {
            (tuple(sorted(value["admitted_channels"])), tuple(bool(value[name]) for name in STAGES))
            for value in near
        }
        rashomon_groups += len(near) >= 2 and len(structures) >= 2
    overall = "COMPLETED" if comparable_groups else "PARTIAL"
    interpretation = (
        "# E5 Structural Stability Audit\n\n"
        f"Status: `{overall}`.\n\n"
        f"Valid frozen structure signatures: {len(signatures)}; comparable groups: "
        f"{comparable_groups}. Empty-empty channel agreements are explicitly marked "
        f"({empty_empty} pairs) and are not interpreted as successful structure recovery.\n\n"
        f"Rashomon groups with near-equal prediction but different admitted structure: "
        f"{rashomon_groups}. The <=1.01 x best definition is complete. The secondary "
        "statistical-indistinguishability definition is `NOT_RUN` wherever paired block "
        "residuals are absent; no IID timestamp test is substituted.\n\n"
        "Performance stability, support/channel stability, and stage/scale stability "
        "are reported in separate files.\n"
    )
    (output / "README.md").write_text(interpretation, encoding="utf-8")
    (output / "E5_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    manifest = {
        "status": overall,
        "valid_signature_count": len(signatures),
        "comparable_group_count": comparable_groups,
        "empty_empty_pair_count": empty_empty,
        "rashomon_structure_group_count": rashomon_groups,
        "statistical_indistinguishability_status": "NOT_RUN",
    }
    _write_json(output / "E5_REPORT_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report([path.resolve() for path in args.input_root], args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
