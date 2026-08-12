"""Command-line stage runner for the frozen NeuroBEM PRISM experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .neurobem_data import (
    assert_partition_disjoint,
    development_data_audit,
    extract_processed_archive,
    load_segment,
    parse_flights,
    registry_from_zip,
    registry_json,
    sha256_file,
    write_json,
)
from .neurobem_experiment import (
    aggregate_predictions,
    bootstrap_segment_differences,
    context_from_json,
    context_to_json,
    era_from_json,
    era_to_json,
    final_crossfit_contracts,
    load_partition,
    predict_test_segment,
    records_from_json,
    ridge_from_json,
    ridge_to_json,
    run_a_development,
    run_era_development,
    run_k_development,
    run_w_development,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _runtime(repo_root: Path) -> dict[str, object]:
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(repo_root, "status", "--porcelain"),
        "blas_thread_environment": {
            key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
        },
    }


def stage0(repo_root: Path, config_path: Path, data_root: Path, output_root: Path) -> None:
    config = _load_json(config_path)
    source_docs = data_root / "source_docs"
    archive = data_root / "raw" / "processed_data.zip"
    extracted = data_root / "extracted"
    flights_path = source_docs / "Flights.txt"
    testset_path = source_docs / "testset.txt"
    if len(parse_flights(flights_path)) != int(config["source"]["flight_count_expected"]):
        raise RuntimeError("EXPECTED_96_PARENT_FLIGHTS")
    records = registry_from_zip(archive, flights_path, testset_path, config)
    assert_partition_disjoint(records)
    extract_processed_archive(archive, extracted)
    development = [
        load_segment(extracted, record, allow_locked_test=False)
        for record in records
        if record.partition in {"train", "validation"}
    ]
    audit = development_data_audit(development)
    counts: dict[str, dict[str, int]] = {}
    for partition in ["train", "validation", "test"]:
        subset = [record for record in records if record.partition == partition]
        counts[partition] = {
            "parent_flights": len({record.flight_id for record in subset}),
            "segments": len(subset),
            "parsed_numeric_rows": int(sum(segment.row_count for segment in development if segment.record.partition == partition)),
        }
    result = {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(config_path),
        "canonical_theory_sha256": sha256_file(repo_root / "PRISM_INDUSTRIAL_BENCHMARK_V1" / "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PACKAGE" / "reference" / "PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md"),
        "sources": {
            "processed_data.zip": {"bytes": archive.stat().st_size, "sha256": sha256_file(archive)},
            "Flights.txt": {"bytes": flights_path.stat().st_size, "sha256": sha256_file(flights_path)},
            "testset.txt": {"bytes": testset_path.stat().st_size, "sha256": sha256_file(testset_path)},
            "Readme.md": {"bytes": (source_docs / "Readme.md").stat().st_size, "sha256": sha256_file(source_docs / "Readme.md")},
        },
        "counts": counts,
        "development_data_audit": audit,
        "test_accessed": False,
        "ood_accessed": False,
        "runtime": _runtime(repo_root),
    }
    write_json(output_root / "N0" / "RESULT.json", result)
    write_json(output_root / "N1" / "SEGMENT_REGISTRY.json", registry_json(records))
    write_json(
        output_root / "N1" / "SPLIT_MANIFEST.json",
        {
            "status": "PASS",
            "parent_group_disjoint": True,
            "history_entity": "CONTIGUOUS_PROCESSED_SEGMENT_ID",
            "cross_segment_history": False,
            "counts": counts,
            "test_numeric_values_accessed": False,
        },
    )
    write_json(output_root / "RUN_STATUS.json", {"status": "STAGE0_PASS", "test_accessed": False, "ood_accessed": False})


def development(repo_root: Path, config_path: Path, data_root: Path, output_root: Path) -> None:
    config = _load_json(config_path)
    n0 = _load_json(output_root / "N0" / "RESULT.json")
    if n0["status"] != "PASS" or n0["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("STAGE0_OR_CONFIG_PROVENANCE_FAILURE")
    records = records_from_json(_load_json(output_root / "N1" / "SEGMENT_REGISTRY.json"))
    extracted = data_root / "extracted"
    train = load_partition(records, extracted, "train")
    validation = load_partition(records, extracted, "validation")
    k_result, k_contract, frames, _ = run_k_development(train, validation, config)
    write_json(output_root / "N2_K" / "RESULT.json", k_result)
    if k_result["status"] != "PASS":
        write_json(output_root / "RUN_STATUS.json", {"status": "PHYSICS_ROUTE_NOT_SUPPORTED", "stage": "N2", "test_accessed": False, "ood_accessed": False})
        return
    w_result, _, pf_frames = run_w_development(frames, config)
    write_json(output_root / "N3_W" / "RESULT.json", w_result)
    a_result, _, a_lags = run_a_development(pf_frames, config)
    write_json(output_root / "N4_A" / "RESULT.json", a_result)
    history = int(k_result["selected_history"])
    era_result, era_contract = run_era_development(k_contract, history, validation, config)
    write_json(output_root / "N5_MIMO_ERA" / "RESULT.json", era_result)
    selected_w = str(w_result["selected_W"])
    development_all = list(train) + list(validation)
    final_k, final_w, final_a, _ = final_crossfit_contracts(
        development_all, history, selected_w, a_lags, config
    )
    final_era = None
    if era_contract is not None:
        from .neurobem_linear import era_from_markov, markov_parameters

        final_era = era_from_markov(
            markov_parameters(final_k, history),
            int(era_result["selected_order"]),
            int(era_result["block_rows"]),
        )
        if final_era.spectral_radius >= float(config["mimo_realization"]["maximum_spectral_radius"]):
            final_era = None
    formal_routes = ["K", "KW", "PF_SELECTED"]
    if a_lags is not None:
        formal_routes.insert(2, "KWA")
    if final_era is not None:
        formal_routes.append("ERA_K")
    freeze = {
        "status": "PASS",
        "development_frozen": True,
        "formal_routes": formal_routes,
        "selected_K_history": history,
        "selected_W": selected_w,
        "selected_A": a_result["selected_A"],
        "selected_A_lags": a_lags,
        "MIMO_realization_status": era_result["status"],
        "selected_MIMO_order": era_result.get("selected_order"),
        "K_contract": ridge_to_json(final_k),
        "W_contract": context_to_json(final_w),
        "A_contract": None if final_a is None else ridge_to_json(final_a),
        "ERA_contract": era_to_json(final_era),
        "config_sha256": sha256_file(config_path),
        "source_archive_sha256": n0["sources"]["processed_data.zip"]["sha256"],
        "code_commit": _git(repo_root, "rev-parse", "HEAD"),
        "test_accessed": False,
        "ood_accessed": False,
        "test_access_authorized": True,
        "runtime": _runtime(repo_root),
    }
    write_json(output_root / "N6_FREEZE" / "DEVELOPMENT_FREEZE.json", freeze)
    write_json(
        output_root / "N6_FREEZE" / "DEVELOPMENT_DECISION.json",
        {
            "status": "PASS_PF_FROZEN",
            "K_status": k_result["status"],
            "W_status": selected_w,
            "A_status": a_result["selected_A"],
            "MIMO_status": era_result["status"],
            "formal_routes": formal_routes,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    write_json(output_root / "RUN_STATUS.json", {"status": "DEVELOPMENT_FROZEN", "stage": "N6", "test_accessed": False, "ood_accessed": False})


def final(repo_root: Path, config_path: Path, data_root: Path, output_root: Path) -> None:
    config = _load_json(config_path)
    freeze_path = output_root / "N6_FREEZE" / "DEVELOPMENT_FREEZE.json"
    freeze = _load_json(freeze_path)
    if not freeze.get("development_frozen") or not freeze.get("test_access_authorized"):
        raise RuntimeError("TEST_ACCESS_NOT_AUTHORIZED_BY_DEVELOPMENT_FREEZE")
    if freeze["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("CONFIG_CHANGED_AFTER_FREEZE")
    records = records_from_json(_load_json(output_root / "N1" / "SEGMENT_REGISTRY.json"))
    test = load_partition(records, data_root / "extracted", "test", allow_locked_test=True)
    history = int(freeze["selected_K_history"])
    k_contract = ridge_from_json(freeze["K_contract"])
    w_contract = context_from_json(freeze["W_contract"])
    a_contract = None if freeze["A_contract"] is None else ridge_from_json(freeze["A_contract"])
    a_lags = freeze["selected_A_lags"]
    era_contract = era_from_json(freeze["ERA_contract"])
    predictions = [
        predict_test_segment(segment, history, k_contract, w_contract, a_contract, a_lags, era_contract, config)
        for segment in test
    ]
    main_metrics = aggregate_predictions(predictions)
    high_speed_metrics = aggregate_predictions(
        predictions,
        mask_name="high_speed",
        threshold=float(config["locked_evaluation"]["high_speed_threshold_m_per_s"]),
    )
    bootstrap = bootstrap_segment_differences(
        predictions,
        int(config["locked_evaluation"]["paired_cluster_bootstrap_replicates"]),
    )
    result = {
        "status": "COMPLETED",
        "formal_routes": freeze["formal_routes"],
        "test_parent_flights": len({segment.record.flight_id for segment in test}),
        "test_segments": len(test),
        "test_metrics": main_metrics,
        "high_speed_ood_threshold_m_per_s": config["locked_evaluation"]["high_speed_threshold_m_per_s"],
        "high_speed_ood_metrics": high_speed_metrics,
        "paired_segment_cluster_bootstrap": bootstrap,
        "test_accessed": True,
        "ood_accessed": True,
        "model_selection_after_test": False,
        "runtime": _runtime(repo_root),
    }
    write_json(output_root / "N7_TEST" / "RESULT.json", result)
    write_json(output_root / "N8_FINAL" / "FINAL_RESULT.json", result)
    write_json(output_root / "RUN_STATUS.json", {"status": "COMPLETED", "stage": "N8", "test_accessed": True, "ood_accessed": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage0", "development", "final"])
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "stage0":
        stage0(args.repo_root, args.config, args.data_root, args.output_root)
    elif args.stage == "development":
        development(args.repo_root, args.config, args.data_root, args.output_root)
    else:
        final(args.repo_root, args.config, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
