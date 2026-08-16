from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark import v211_public_all_closure as closure
from prism_benchmark.v211_public_all_baselines import SupportRequirement


def _view(head_id: str, information_set: str) -> ViewSpec:
    return ViewSpec(
        head=HeadSpec(
            head_id=head_id,
            task_id=head_id,
            dataset="synthetic",
            target="y",
            cadence_seconds=600.0,
            h_steps=1,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set=information_set,
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def test_run_public_all_test_materializes_registered_ood_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    dynamic = _view("DYNAMIC_HEAD", "dynamic")
    input_only = _view("INPUT_HEAD", "input_only")
    ood_path = shared / "sample_ids" / dynamic.relative_root / "ood.parquet"
    ood_path.parent.mkdir(parents=True)
    ood_path.touch()

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "generating_commit": "g",
                "canonical_theory_sha256": "theory",
                "shared_development_metadata_sha256": "shared",
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "access.json"
    paths = SimpleNamespace(
        shared=shared,
        project=tmp_path,
        final=tmp_path / "final",
        run_root=tmp_path,
        development_freeze_path=freeze_path,
        test_access_audit_path=audit_path,
    )

    monkeypatch.setattr(closure, "public_all_input_views", lambda _: [input_only])
    monkeypatch.setattr(closure, "public_all_dynamic_views", lambda _: [dynamic])
    monkeypatch.setattr(
        closure,
        "load_public_all_descriptor",
        lambda _: {"config_sha256": "config"},
    )

    prism_calls: list[tuple[str, str]] = []
    baseline_calls: list[tuple[str, str]] = []

    def fake_input(paths, view, *, split):
        del paths
        prism_calls.append((view.head.head_id, split))
        return [{"status": "PASS", "model": "input", "split": split}]

    def fake_dynamic(paths, view, *, split):
        del paths
        prism_calls.append((view.head.head_id, split))
        return [{"status": "PASS", "model": "dynamic", "split": split}]

    def fake_baseline(paths, view, *, split):
        del paths
        baseline_calls.append((view.head.head_id, split))
        return [{"status": "PASS", "model": "baseline", "split": split}]

    import prism_benchmark.v211_public_all_materialization as materialization
    import prism_benchmark.v211_public_all_baseline_materialization as baseline

    monkeypatch.setattr(materialization, "materialize_input_prism_view", fake_input)
    monkeypatch.setattr(materialization, "materialize_dynamic_prism_view", fake_dynamic)
    monkeypatch.setattr(
        materialization,
        "preflight_public_all_materialization",
        lambda *args: {
            "status": "PASS",
            "formal_joint_views": 1,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    monkeypatch.setattr(baseline, "materialize_baseline_view", fake_baseline)

    result = closure.run_public_all_test(paths)

    assert result["status"] == "PASS"
    assert result["registered_ood_views"] == 1
    assert result["test_accessed"] is True
    assert result["ood_accessed"] is True
    assert result["test_y_read"] is True
    assert result["ood_y_read"] is True
    access_audit = json.loads(paths.test_access_audit_path.read_text(encoding="utf-8"))
    assert access_audit["materialization_contract_preflight"]["status"] == "PASS"
    assert sorted(prism_calls) == sorted(
        [
            ("INPUT_HEAD", "test"),
            ("DYNAMIC_HEAD", "test"),
            ("DYNAMIC_HEAD", "ood"),
        ]
    )
    assert sorted(baseline_calls) == sorted(
        [
            ("INPUT_HEAD", "test"),
            ("DYNAMIC_HEAD", "test"),
            ("DYNAMIC_HEAD", "ood"),
        ]
    )
    assert (
        tmp_path / "final" / "baseline_test_predictions" / "BASELINE_TEST_RESULT.json"
    ).is_file()
    assert (
        tmp_path / "final" / "baseline_ood_predictions" / "BASELINE_OOD_RESULT.json"
    ).is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["stage"] == "T1_PUBLIC_ALL_TEST_OOD_ACCESS"


def test_joint_requirements_only_include_formal_joint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view("DYNAMIC_HEAD", "dynamic")
    paths = SimpleNamespace(output=tmp_path)
    monkeypatch.setattr(closure, "_selected_k_histories", lambda *_: [])
    monkeypatch.setattr(closure, "_baseline_result", lambda *_: None)

    def rejected_result(output, stage, selected_view):
        del output, selected_view
        if stage == "JOINT":
            return {
                "status": "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
                "ar_profile": [2, 8],
            }
        if stage == "A":
            return {"status": "PASS", "a_contract": {"family": "EXACT_ZERO"}}
        return None

    monkeypatch.setattr(closure, "_prism_result", rejected_result)
    assert closure.view_support_requirements(paths, view) == (SupportRequirement(),)

    def passed_result(output, stage, selected_view):
        value = rejected_result(output, stage, selected_view)
        if stage == "JOINT":
            return {**value, "status": "PASS"}
        return value

    monkeypatch.setattr(closure, "_prism_result", passed_result)
    assert closure.view_support_requirements(paths, view) == (
        SupportRequirement(target_delta_steps=2, target_history_steps=8),
    )


def test_formal_routes_keep_pf_when_joint_is_not_supported() -> None:
    view = _view("DYNAMIC_HEAD", "dynamic")
    a_result = {"status": "PASS"}
    joint_result = {
        "status": "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT"
    }
    assert closure._formal_routes(view, a_result, joint_result) == [
        "PHYSICS_FIRST"
    ]
    assert closure._formal_routes(
        view, a_result, {"status": "PASS"}
    ) == ["PHYSICS_FIRST", "JOINT"]
