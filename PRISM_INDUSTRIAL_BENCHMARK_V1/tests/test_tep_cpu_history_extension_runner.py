from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT / "scripts" / "run_tep_cpu_history_extension_20260826.py"
LAUNCHER_PATH = PROJECT / "scripts" / "launch_tep_cpu_history_extension_20260826.py"
CONFIG_PATH = PROJECT / "configs" / "tep_cpu_history_extension_20260826.json"


def _runner():
    source = str(PROJECT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("tep_cpu_history_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": [255, 256, 260, 300],
            "causal_history_floor": [0, 0, 4, 40],
            "latest_available_target_index": [250, 251, 255, 295],
            "sample_support_contract": ["NATIVE_K_COMMON_ASSEMBLY_R1"] * 4,
            "base_origin_id": [f"b{i}" for i in range(4)],
            "view_sample_id": [f"v{i}" for i in range(4)],
        }
    )


def test_frozen_scope_is_tep_only_and_single_worker() -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["protocol_id"] == "TEP_CPU_HISTORY_EXTENSION_L256_V1"
    assert value["active_datasets"] == ["tep"]
    assert value["active_tasks"] == ["TEP_G_REP_H1"]
    assert value["history_aware_steps"] == [2, 4, 8, 128, 256]
    assert value["common_support_history_steps"] == 256
    assert value["resources"]["workers"] == 1
    assert value["resources"]["hard_memory_limit_gib"] == 90
    assert value["disabled"]["sru"] == "NOT_RUN_BY_USER_SCOPE"
    assert value["disabled"]["cz_czochralski"] == "NOT_RUN_BY_USER_SCOPE"
    assert value["disabled"]["neural3"] == "NOT_RUN_BY_USER_SCOPE"


def test_l256_input_support_uses_origin_exclusive_history() -> None:
    module = _runner()
    assert module._l256_mask(_frame(), "input_only").tolist() == [False, True, True, True]


def test_l256_dynamic_support_also_requires_available_target_history() -> None:
    module = _runner()
    # origin history is legal for rows 1--3, but analyzer availability leaves
    # only the last row with a full 256-point target history.
    assert module._l256_mask(_frame(), "dynamic").tolist() == [False, False, False, True]


def test_launcher_has_no_forbidden_execution_stage() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    choices = text[text.index('choices=("scope"') : text.index('parser.add_argument("--shared"')]
    for forbidden in ("cz", "sru", "neural3", "stage2", "test", "ood"):
        assert f'"{forbidden}"' not in choices.lower()


def test_incompatible_registry_is_explicit() -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["registered_protocol_incompatible"] == [
        "SEASONAL_PERSISTENCE",
        "N4SID",
    ]
    assert value["history_method_overrides"]["dpls_maximum_lags_per_channel"] == 16
    assert value["history_method_overrides"]["dpls_maximum_joint_configurations"] == 40
    assert value["history_method_overrides"]["hammerstein_profile_cap"] == 8
    assert value["history_method_overrides"]["state_delta_steps"] == [1, 2, 4]


def test_explicit_override_binds_every_history_and_preserves_dpls_lag_cap() -> None:
    source = str(PROJECT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from prism_benchmark.c3_models import _hammerstein_profiles
    from prism_benchmark.cpu_data import HeadSpec, ViewSpec, realized_state_profiles
    from prism_benchmark.v211_history_override import load_tep_history_override

    override = load_tep_history_override(CONFIG_PATH.resolve())
    assert override is not None
    assert override.dpls_maximum_lags_per_channel == 16
    assert override.dpls_maximum_joint_configurations == 40
    head = HeadSpec(
        head_id="TEP_G_REP_H1__H1__W2",
        task_id="TEP_G_REP_H1",
        dataset="tep",
        target="G",
        cadence_seconds=180.0,
        h_steps=1,
        w_steps=2,
        w0_steps=1,
        primary=True,
    )
    view = ViewSpec(head, "input_only", "record_time", "registered")
    profiles = realized_state_profiles(
        head,
        override.positive_h_history_multipliers,
        override.state_delta_steps,
    )
    assert {history for _, history in profiles} == {2, 4, 8, 128, 256}
    legacy = _hammerstein_profiles(view)
    capped = _hammerstein_profiles(
        view,
        override.positive_h_history_multipliers,
        override.state_delta_steps,
        override.hammerstein_profile_cap,
        True,
    )
    assert len(capped) == 8
    assert set(legacy).issubset(capped)
    assert {history for _, history in capped} == {2, 4, 8, 128, 256}


def test_pilot_gate_requires_actual_finite_four_fold_losses() -> None:
    module = _runner()
    expected = {2, 4, 8, 128, 256}
    k_losses = {
        str((1, history)): [1.0, 1.1, 1.2, 1.3] for history in expected
    }
    dpls_losses = {
        str((history, 1)): [1.0, 1.1, 1.2, 1.3] for history in expected
    }
    assert module._finite_fold_loss_histories(
        k_losses, history_position=1
    ) == expected
    assert module._finite_fold_loss_histories(
        dpls_losses, history_position=0
    ) == expected

    metadata_only = {
        "registered_history_support_audit": [
            {"history_steps": history, "available": True} for history in expected
        ]
    }
    assert module._finite_fold_loss_histories(
        metadata_only, history_position=1
    ) == set()
    assert module._finite_fold_loss_histories(
        {"(128, 1)": [1.0] * 4}, history_position=0
    ) == {128}
    assert module._finite_fold_loss_histories(
        {"(1, 256)": [1.0, float("inf"), 1.0, 1.0]}, history_position=1
    ) == set()


def test_launcher_uses_cgroup_75gib_and_kills_the_stage_process_group() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert 'Path("/sys/fs/cgroup/memory.current")' in text
    assert "if cgroup_gib >= recommended:" in text
    assert "os.killpg(process.pid, signal.SIGTERM)" in text
    assert "start_new_session=True" in text
    assert 'source_path = str(PROJECT / "src")' in text
    assert 'environment["PYTHONPATH"]' in text
    assert 'if stage == "scope"' in text
    assert 'mode.add_argument("--pilot-only"' in text


def test_k_smoothness_scoring_history_is_opt_in(monkeypatch) -> None:
    _runner()
    from prism_benchmark import v211_k

    observed: list[int] = []

    def fake_parallel_map(function, jobs, workers):
        del function, workers
        observed.extend(int(job[-1]) for job in jobs)
        return [[1.0, 1.0] for _ in jobs]

    monkeypatch.setattr(v211_k, "_ordered_parallel_map", fake_parallel_map)
    arguments = {
        "accessor": object(),
        "train": object(),
        "folds": [],
        "channel": "xmeas_1",
        "profile": (1, 8),
        "m_tau": 1,
        "family": "LINEAR_DISTRIBUTED_LAG",
        "m_x": 1,
        "v2": {
            "K_module": {
                "penalties": {
                    "pilot": {"lambda_tau": 1.0, "lambda_x": 1.0},
                    "lambda_tau": [1.0],
                    "lambda_x": [1.0],
                }
            }
        },
        "minimum_folds": 1,
    }

    v211_k._smoothness_selection(**arguments)
    assert observed == [8]

    observed.clear()
    v211_k._smoothness_selection(**arguments, scoring_history_steps=256)
    assert observed == [256]
