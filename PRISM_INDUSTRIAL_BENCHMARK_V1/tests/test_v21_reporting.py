import numpy as np
import pandas as pd

from prism_benchmark.v21_config import V21Paths
from prism_benchmark.v21_reporting import (
    _comparison_specs,
    _holm,
    _moving_block_means,
    package_results,
)


def test_moving_block_bootstrap_is_deterministic_and_entity_local():
    paired = pd.DataFrame(
        {
            "entity_id": ["a"] * 5 + ["b"] * 5,
            "origin": list(range(5)) * 2,
            "loss_difference": np.arange(10, dtype=np.float64),
        }
    )
    first = _moving_block_means(paired, block_length=3, replicates=20, seed=7)
    second = _moving_block_means(paired, block_length=3, replicates=20, seed=7)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 20


def test_holm_adjustment_is_monotone_within_family():
    rows = [
        {
            "target_head": "head",
            "information_set": "dynamic",
            "comparison_family": "primary",
            "raw_p_value": value,
        }
        for value in (0.01, 0.03, 0.02)
    ]
    _holm(rows)
    ordered = sorted(rows, key=lambda row: row["raw_p_value"])
    adjusted = [row["holm_p_value"] for row in ordered]
    assert adjusted == sorted(adjusted)
    assert all(row["holm_p_value"] >= row["raw_p_value"] for row in rows)


def test_result_packager_builds_verified_zip(tmp_path, monkeypatch):
    project = tmp_path / "project"
    plan = project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED"
    plan.mkdir(parents=True)
    (plan / "README.md").write_text("plan", encoding="utf-8")
    output = project / "results_prism_v2_1_sru"
    for directory in (
        "FREEZE",
        "DATA_AUDIT",
        "BASELINES",
        "ASSEMBLY_CARDS",
        "FINAL",
        "REPORTS",
    ):
        root = output / directory
        root.mkdir(parents=True)
        (root / "artifact.txt").write_text(directory, encoding="utf-8")
    (output / "FREEZE" / "V21_SRU_FINAL_FREEZE_MANIFEST.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (output / "FINAL" / "V21_SRU_FINAL_METRICS.csv").write_text(
        "model,mse\nmodel,1\n",
        encoding="utf-8",
    )
    (output / "REPORTS" / "V21_SRU_FINAL_REPORT.md").write_text(
        "report",
        encoding="utf-8",
    )
    paths = V21Paths(project, tmp_path / "shared", output)
    monkeypatch.setattr(
        "prism_benchmark.v21_reporting._git",
        lambda _project, *args: "head" if "rev-parse" in args else "",
    )
    result = package_results(paths)
    assert result["status"] == "ZIP_INTEGRITY_PASS"
    assert (project / "PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle.zip").is_file()


def test_comparison_registry_contains_primary_and_mechanism_routes():
    head = "SRU_H2S__H5__W1"
    common = {
        "target_head": head,
        "availability_scenario": "record_time",
        "proxy_policy": "primary",
    }
    input_models = (
        "PRISM_V2_1_K_C",
        "PRISM_V2_1_K_C_W",
        "PRISM_V2_1_K_C_W_MU0_ABLATION",
        "PRISM_CHANNEL_SPECIFIC",
        "HAMMERSTEIN_WIENER",
        "PARALLEL_HAMMERSTEIN",
    )
    dynamic_models = (
        "PRISM_V2_1_K_C_DYNAMIC",
        "PRISM_V2_1_K_C_W_DYNAMIC",
        "PRISM_V2_1_K_C_A_ABLATION",
        "PRISM_V2_1_PHYSICS_FIRST",
        "PRISM_V2_1_PF_A_MU0_ABLATION",
        "PRISM_V2_1_JOINT_KWA",
        "PRISM_V2_1_J_KA",
        "PRISM_V2_1_J_KWA",
        "PRISM_PHYSICS_FIRST",
        "PRISM_K_JOINT_AR",
        "ARX",
        "LINEAR_NARX",
    )
    audits = [
        {**common, "information_set": "input_only", "model": model}
        for model in input_models
    ] + [
        {**common, "information_set": "dynamic", "model": model}
        for model in dynamic_models
    ]
    inventory = {
        "best_by_validation": {
            f"{head}|input_only|record_time|primary": "PRISM_CHANNEL_SPECIFIC",
            f"{head}|dynamic|record_time|primary": "ARX",
        }
    }
    specs = _comparison_specs(inventory, audits)
    pairs = {(row["candidate"], row["comparator"]) for row in specs}
    assert ("PRISM_V2_1_K_C_W", "PRISM_CHANNEL_SPECIFIC") in pairs
    assert ("PRISM_V2_1_K_C_W", "PRISM_V2_1_K_C") in pairs
    assert ("PRISM_V2_1_K_C_A_ABLATION", "PRISM_V2_1_K_C_DYNAMIC") in pairs
    assert ("PRISM_V2_1_PHYSICS_FIRST", "PRISM_V2_1_K_C_W_DYNAMIC") in pairs
    assert ("PRISM_V2_1_J_KWA", "PRISM_V2_1_J_KA") in pairs
