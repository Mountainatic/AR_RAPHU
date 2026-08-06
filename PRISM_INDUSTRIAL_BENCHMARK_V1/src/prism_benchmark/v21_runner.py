from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .stage0 import write_json
from .v21_a import run_e4_a
from .v21_audit import write_post_audit, write_pre_audit
from .v21_baselines import freeze_baseline_inventory
from .v21_c import run_e2_c
from .v21_config import ACTIVE_HEADS, V21Paths, load_v21_config
from .v21_joint import run_e5_joint
from .v21_k import run_e2_k
from .v21_views import assert_only_sru, sru_dynamic_views, sru_input_views
from .v21_w import run_e3_w


E1_TESTS = (
    "tests/test_v21_selection.py",
    "tests/test_v21_maturity.py",
    "tests/test_v21_w.py",
    "tests/test_v21_a.py",
    "tests/test_v21_joint.py",
    "tests/test_v21_assembly.py",
    "tests/test_v21_data_immutability.py",
    "tests/test_v21_prediction_contract.py",
    "tests/test_v21_baselines.py",
    "tests/test_v21_reporting.py",
)


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-c", "core.filemode=false", *args], cwd=project, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def run_e0(paths: V21Paths) -> dict[str, Any]:
    config = load_v21_config(paths.project)
    validation = json.loads((paths.shared / "C1_VALIDATION.json").read_text(encoding="utf-8"))
    input_views = sru_input_views(paths.shared)
    dynamic_views = sru_dynamic_views(paths.shared)
    assert_only_sru(input_views)
    assert_only_sru(dynamic_views)
    pre = write_pre_audit(paths.shared, paths.output)
    observed_heads = {view.head.head_id for view in [*input_views, *dynamic_views]}
    non_sru = pre["summary"].get("non_sru", {}).get("files", 0)
    branch = _git(paths.project, "branch", "--show-current")
    checks = {
        "C1_validation_pass": validation.get("status") == "PASS",
        "active_heads_exact": observed_heads == ACTIVE_HEADS,
        "non_sru_bases_preserved": non_sru > 0,
        "write_shared_data_false": config["write_shared_data"] is False,
        "legacy_v2_freeze_present": (paths.project / "PRISM_V2_MODULAR_NUMERICALLY_FROZEN").is_dir(),
        "branch_is_v21_or_detached_release": branch
        in {"prism-v2-1-sru-stagewise-routed", ""},
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP_INHERITANCE_MISMATCH",
        "stage": "E0_INHERITANCE_AUDIT",
        "checks": checks,
        "git_head": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": branch,
        "git_dirty": bool(_git(paths.project, "status", "--short")),
        "data_pre_audit_sha256": _sha256(paths.output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"),
        "test_accessed": False,
    }
    write_json(paths.output / "FREEZE" / "E0_INHERITANCE_AUDIT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(result["status"])
    return result


def run_e1(paths: V21Paths) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *E1_TESTS, "-q"]
    completed = subprocess.run(command, cwd=paths.project, text=True, capture_output=True)
    result = {"status": "PASS" if completed.returncode == 0 else "FAILED", "stage": "E1_REGRESSION_TESTS", "command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "test_accessed": False}
    write_json(paths.output / "FREEZE" / "E1_REGRESSION_TESTS.json", result)
    if completed.returncode:
        raise RuntimeError("E1 v2.1 regression tests failed")
    return result


def _result_files(output: Path) -> list[Path]:
    return sorted(path for path in (output / "DEVELOPMENT").rglob("RESULT.json") if path.is_file())


def freeze_e6(paths: V21Paths) -> dict[str, Any]:
    from .v21_selection import assert_final_prediction_contract
    load_v21_config(paths.project)
    required = {"E2_K": False, "E2_C": False, "E3_W": False, "E4_A": False, "E5_JOINT": False}
    selections = []
    validation_files = []
    for path in _result_files(paths.output):
        result = json.loads(path.read_text(encoding="utf-8"))
        stage = result.get("stage")
        if stage in required:
            required[stage] = True
        if result.get("status") != "PASS":
            raise RuntimeError(f"cannot freeze retained failure: {path}")
        assert_final_prediction_contract(result)
        prediction = paths.output / result["final_selected_prediction_path"]
        validation_files.append({"path": prediction.relative_to(paths.output).as_posix(), "sha256": _sha256(prediction)})
        selections.append({"stage": stage, "target_head": result.get("target_head"), "proxy_policy": result.get("proxy_policy"), "availability_scenario": result.get("availability_scenario"), "selected": result.get("final_selected_candidate"), "contract_sha256": _sha256(path)})
    if not all(required.values()):
        raise RuntimeError(f"E6 prerequisites missing: {required}")
    baseline_inventory = freeze_baseline_inventory(paths)
    post = write_post_audit(paths.shared, paths.output)
    if post["comparison_to_pre"]["status"] != "PASS":
        raise RuntimeError("STOP_DATA_BASE_MUTATED")
    frozen_config = paths.output / "FREEZE" / "V21_SRU_CONFIG_FROZEN.json"
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_config.write_bytes(paths.config_path.read_bytes())
    manifest = {
        "status": "ASSEMBLY_FROZEN", "stage": "E6_FINAL_FREEZE",
        "theory_sha256": _sha256(paths.plan / "PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md"),
        "plan_sha256": _sha256(paths.plan / "PRISM_V2_1_SRU_EXPERIMENT_AND_IMPLEMENTATION_PLAN.md"),
        "config_sha256": _sha256(paths.config_path), "frozen_config_sha256": _sha256(frozen_config),
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "dirty_status": _git(paths.project, "status", "--short"),
        "data_pre_audit_sha256": _sha256(paths.output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"),
        "data_post_audit_sha256": _sha256(paths.output / "DATA_AUDIT" / "V21_DATA_BASE_POST_AUDIT.json"),
        "baseline_inventory_sha256": _sha256(
            paths.output / "BASELINES" / "FROZEN_BASELINE_INVENTORY.json"
        ),
        "baseline_inclusion": baseline_inventory["entries"],
        "best_frozen_baselines": baseline_inventory["best_by_validation"],
        "selections": selections, "validation_files": validation_files,
        "test_accessed": False,
    }
    write_json(paths.final_freeze_path, manifest)
    return manifest


def run_stage(stage: str, paths: V21Paths) -> dict[str, Any]:
    if paths.output.name != "results_prism_v2_1_sru":
        raise RuntimeError("v2.1 output must use the isolated results_prism_v2_1_sru namespace")
    if stage == "e0": return run_e0(paths)
    if stage == "e1": return run_e1(paths)
    if stage == "e2k": return run_e2_k(paths.shared, paths.project, paths.output)
    if stage == "e2c": return run_e2_c(paths.shared, paths.project, paths.output)
    if stage == "e3": return run_e3_w(paths.shared, paths.project, paths.output)
    if stage == "e4": return run_e4_a(paths.shared, paths.project, paths.output)
    if stage == "e5": return run_e5_joint(paths.shared, paths.project, paths.output)
    if stage == "e6": return freeze_e6(paths)
    if stage in {"e7", "e8"}:
        from .v21_final import run_e7_test, run_e8_report
        return run_e7_test(paths) if stage == "e7" else run_e8_report(paths)
    raise ValueError(f"unknown v2.1 stage: {stage}")
