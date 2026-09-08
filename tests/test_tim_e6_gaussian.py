from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.tim_validation.e6_robustness.n1_gaussian import (
    _adjust_dynamic_sample_targets,
    _full_record,
    outer_train_sigma,
    perturb_gaussian_process_only,
    perturb_process_measurements,
)


def test_gaussian_levels_are_nested_and_target_missingness_is_preserved() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0],
            "target": [10.0, 11.0, 12.0, 13.0],
            "entity_id": ["a"] * 4,
            "row_in_entity": range(4),
        }
    )
    sigma = {"x": 2.0}
    low = perturb_gaussian_process_only(
        frame, task="T", seed=7, alpha=0.05, sigma=sigma
    )
    high = perturb_gaussian_process_only(
        frame, task="T", seed=7, alpha=0.10, sigma=sigma
    )
    finite = frame["x"].notna().to_numpy()
    low_delta = low.loc[finite, "x"].to_numpy() - frame.loc[finite, "x"].to_numpy()
    high_delta = high.loc[finite, "x"].to_numpy() - frame.loc[finite, "x"].to_numpy()
    assert np.allclose(high_delta, 2.0 * low_delta)
    assert low["x"].isna().tolist() == frame["x"].isna().tolist()
    assert low["target"].tolist() == frame["target"].tolist()


def test_alpha_zero_preserves_values_exactly() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "target": [3.0, 4.0]})
    observed = perturb_gaussian_process_only(
        frame, task="T", seed=1, alpha=0.0, sigma={"x": 1.0}
    )
    assert observed.equals(frame)


def test_bias_linear_drift_and_quantization_keep_target_clean() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.1, 0.4, 0.8, 1.2],
            "target": [4.0, 5.0, 6.0, 7.0],
            "entity_id": ["a"] * 4,
            "row_in_entity": range(4),
        }
    )
    for kind, direction in (("bias", 1), ("linear_drift", -1), ("quantization", 0)):
        observed, amplitudes = perturb_process_measurements(
            frame,
            perturbation=kind,
            task="T",
            seed=1,
            alpha=0.1,
            sigma={"x": 2.0},
            direction=direction,
        )
        assert observed["target"].equals(frame["target"])
        assert not observed["x"].equals(frame["x"])
        assert amplitudes["x"]["rms"] >= 0.0


def test_outer_train_sigma_can_include_dynamic_target(tmp_path) -> None:
    root = tmp_path / "shared" / "base_data" / "tep"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 4.0],
            "target": [10.0, 11.0, 14.0],
            "entity_id": ["a"] * 3,
            "row_in_entity": range(3),
        }
    ).to_parquet(root / "train.parquet", index=False)
    process, _ = outer_train_sigma(tmp_path / "shared", "tep", "target")
    realistic, _ = outer_train_sigma(
        tmp_path / "shared", "tep", "target", include_target=True
    )
    assert set(process) == {"x"}
    assert set(realistic) == {"x", "target"}


def test_realistic_dynamic_perturbs_sensor_target_but_not_other_contracts() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "target": [4.0, 5.0, 6.0],
            "entity_id": ["a"] * 3,
            "row_in_entity": range(3),
        }
    )
    observed = perturb_gaussian_process_only(
        frame,
        task="T",
        seed=7,
        alpha=0.1,
        sigma={"x": 1.0, "target": 2.0},
    )
    assert not observed["x"].equals(frame["x"])
    assert not observed["target"].equals(frame["target"])
    assert observed["entity_id"].equals(frame["entity_id"])
    assert observed["row_in_entity"].equals(frame["row_in_entity"])


def test_realistic_dynamic_keeps_future_level_clean_with_noisy_anchor(tmp_path) -> None:
    import json
    import shutil

    clean = tmp_path / "clean"
    base = clean / "base_data" / "d"
    samples_root = clean / "sample_ids" / "H" / "dynamic" / "record_time" / "primary"
    base.mkdir(parents=True)
    samples_root.mkdir(parents=True)
    (clean / "TASK_REGISTRY.json").write_text(
        json.dumps(
            {
                "heads": [
                    {
                        "head_id": "H", "task_id": "T", "dataset": "d",
                        "target": "target", "cadence_seconds": 1.0,
                        "h_steps": 0, "w_steps": 1, "w0_steps": 1,
                        "primary": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0],
            "entity_id": ["a"] * 3,
            "row_in_entity": [0, 1, 2],
        }
    ).to_parquet(base / "test.parquet", index=False)
    pd.DataFrame(
        {
            "entity_id": ["a", "a"],
            "origin": [1, 2],
            "y_true": [1.0, 1.0],
            "current_level": [10.0, 11.0],
        }
    ).to_parquet(samples_root / "test.parquet", index=False)
    perturbed = tmp_path / "perturbed"
    shutil.copytree(clean, perturbed)
    pd.DataFrame(
        {
            "target": [10.0, 11.5, 13.0],
            "entity_id": ["a"] * 3,
            "row_in_entity": [0, 1, 2],
        }
    ).to_parquet(perturbed / "base_data" / "d" / "test.parquet", index=False)
    audit = _adjust_dynamic_sample_targets(
        clean,
        perturbed,
        dataset="d",
        target="target",
        head_id="H",
        information_set="dynamic",
        availability_scenario="record_time",
        proxy_policy="primary",
        split="test",
    )
    adjusted = pd.read_parquet(
        perturbed / "sample_ids" / "H" / "dynamic" / "record_time" / "primary" / "test.parquet"
    )
    assert adjusted["current_level"].tolist() == [10.0, 11.5]
    assert adjusted["y_true"].tolist() == [1.0, 0.5]
    assert audit["clean_future_level_preserved"] is True
    assert audit["reconstruction_max_abs_error"] == 0.0


def test_n1_uses_input_only_terminal_stage_when_full_name_is_absent() -> None:
    record = _full_record(
        {
            "records": [
                {"status": "PASS", "model": "PRISM_V2_1_1_K"},
                {"status": "PASS", "model": "PRISM_V2_1_1_K_C_W"},
            ]
        }
    )
    assert record["model"] == "PRISM_V2_1_1_K_C_W"
