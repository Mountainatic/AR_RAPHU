from pathlib import Path

import pandas as pd

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.v21_baselines import (
    freeze_baseline_inventory,
    materialize_test_baselines,
)
from prism_benchmark.v21_config import V21Paths


def _view() -> ViewSpec:
    return ViewSpec(
        head=HeadSpec(
            head_id="SRU_H2S__H5__W1",
            task_id="SRU_H2S",
            dataset="sru",
            target="H2S",
            cadence_seconds=1.0,
            h_steps=5,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set="input_only",
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def _samples(split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "view_sample_id": [f"{split}-0", f"{split}-1"],
            "y_true": [1.0, 2.0],
        }
    )


def _prediction(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": [f"{split}-0", f"{split}-1"],
            "entity_id": ["run", "run"],
            "origin": [0, 1],
            "y_true": [1.0, 2.0],
            "y_pred": [1.1, 1.9],
            "model": ["MEAN", "MEAN"],
        }
    ).to_parquet(path, index=False)


def test_baseline_test_contents_wait_until_materialization(tmp_path, monkeypatch):
    view = _view()
    root = tmp_path / "baseline"
    for split in ("validation", "test"):
        _prediction(
            root
            / "PREDICTIONS"
            / view.head.head_id
            / split
            / view.information_set
            / view.availability_scenario
            / view.proxy_policy
            / "MEAN.parquet",
            split,
        )
    paths = V21Paths(
        project=tmp_path / "project",
        shared=tmp_path / "shared",
        output=tmp_path / "results_prism_v2_1_sru",
        baseline_root=root,
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines._views",
        lambda _paths: [view],
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.load_v21_config",
        lambda _project: {"baselines": {"input_only": ["MEAN"], "dynamic": []}},
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.load_samples",
        lambda _shared, _view, split: _samples(split),
    )
    frozen = freeze_baseline_inventory(paths)
    assert frozen["test_accessed"] is False
    assert frozen["entries"][0]["test_contents_accessed"] is False
    audits = materialize_test_baselines(paths)
    assert audits[0]["test_accessed"] is True
    assert (paths.output / audits[0]["prediction_path"]).is_file()
