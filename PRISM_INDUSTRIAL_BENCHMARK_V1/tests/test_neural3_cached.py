from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from prism_benchmark.cpu_data import HeadSpec, ViewSpec
from prism_benchmark.neural3 import Scaler, build_model, set_seed
from prism_benchmark.neural3_cached import (
    _pack_tree,
    _read_training_snapshot,
    _unpack_tree,
    _write_training_snapshot,
    load_cached_artifact,
    prepare_candidate,
    train_selection_candidate,
)
from prism_benchmark.neural_candidate_cache import (
    CandidateHashes,
    NeuralCandidateCache,
)


def _hashes() -> CandidateHashes:
    return CandidateHashes("config", "data", "support", "order")


def test_portable_tree_round_trips_optimizer_tensors() -> None:
    set_seed(7)
    model = build_model("LSTM", 3, "SMALL")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = model(torch.randn(4, 8, 3)).square().mean()
    loss.backward()
    optimizer.step()

    arrays, schema = _pack_tree(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    )
    restored = _unpack_tree(arrays, schema)

    clone = build_model("LSTM", 3, "SMALL")
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=3e-4)
    clone.load_state_dict(restored["model"])
    clone_optimizer.load_state_dict(restored["optimizer"])
    for left, right in zip(model.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert clone_optimizer.state_dict()["state"]


def test_dual_slot_training_snapshot_is_reloadable(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    with cache.lock("candidate"):
        writer = cache.begin_candidate("candidate", _hashes(), {"kind": "test"})
        _write_training_snapshot(
            writer,
            {
                "epoch": 1,
                "epoch_in_progress": False,
                "next_batch_number": 0,
                "value": torch.tensor([1.0]),
            },
        )
        _write_training_snapshot(
            writer,
            {
                "epoch": 2,
                "epoch_in_progress": True,
                "next_batch_number": 3,
                "value": torch.tensor([2.0]),
            },
        )
        restored = _read_training_snapshot(writer)
        assert restored is not None
        assert restored["epoch"] == 2
        assert restored["next_batch_number"] == 3
        torch.testing.assert_close(restored["value"], torch.tensor([2.0]))
        pointer = json.loads(
            (writer.staging_dir / "resume" / "LATEST.json").read_text()
        )
        assert pointer["slot"] == "B"
        writer.abort("test complete")


@pytest.mark.parametrize("model_name", ["LSTM", "iTransformer", "TimeMixer"])
def test_cached_model_reload_preserves_prediction_exactly(
    tmp_path: Path, model_name: str
) -> None:
    set_seed(11)
    model = build_model(model_name, 3, "SMALL")
    values = torch.randn(6, 16, 3)
    model.eval()
    with torch.no_grad():
        expected = model(values).numpy()

    cache = NeuralCandidateCache(tmp_path).initialize()
    logical_id = f"candidate-{model_name}"
    with cache.lock(logical_id):
        writer = cache.begin_candidate(logical_id, _hashes(), {"kind": "test"})
        writer.write_npz(
            "model_weights.npz",
            {
                name: tensor.detach().cpu().numpy()
                for name, tensor in model.state_dict().items()
            },
        )
        scaler = Scaler(np.zeros(3), np.ones(3), 0.0, 1.0)
        writer.write_npz(
            "scaler.npz",
            {
                "feature_mean": scaler.feature_mean,
                "feature_scale": scaler.feature_scale,
                "target_mean": np.asarray([scaler.target_mean]),
                "target_scale": np.asarray([scaler.target_scale]),
            },
        )
        writer.write_json(
            "model_config.json",
            {
                "model": model_name,
                "candidate_id": f"{model_name}__L16__SMALL__lr0.0003",
                "history_steps": 16,
                "capacity": "SMALL",
                "learning_rate": 3e-4,
                "input_dim": 3,
                "seed": 11,
            },
        )
        writer.write_json(
            "preprocessing.json",
            {"feature_order": ["a", "b", "c"]},
        )
        writer.finalize(validation_metrics={"mse": 0.0})

    loaded = load_cached_artifact(cache.candidate_dir(logical_id), torch.device("cpu"))
    with torch.no_grad():
        observed = loaded.model(values).numpy()
    np.testing.assert_array_equal(observed, expected)


def _write_toy_shared(root: Path) -> tuple[Path, ViewSpec]:
    shared = root / "shared"
    head = HeadSpec("TOY__H1__W1", "TOY", "toy", "y", 1.0, 1, 1, 1, True)
    view = ViewSpec(head, "input_only", "record_time", "primary")
    train_rows = np.arange(0, 48, dtype=np.int64)
    validation_rows = np.arange(48, 72, dtype=np.int64)
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        frame = pd.DataFrame(
            {
                "entity_id": "entity",
                "row_in_entity": rows,
                "x": np.sin(rows / 7.0),
                "y": np.cos(rows / 9.0),
            }
        )
        path = shared / "base_data" / "toy" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    values = np.cos(np.arange(0, 72) / 9.0)
    for split, origins in (
        ("train", np.arange(8, 46, dtype=np.int64)),
        ("validation", np.arange(52, 70, dtype=np.int64)),
    ):
        samples = pd.DataFrame(
            {
                "base_origin_id": [f"base-{split}-{v}" for v in origins],
                "view_sample_id": [f"view-{split}-{v}" for v in origins],
                "dataset": "toy",
                "entity_id": "entity",
                "task_id": "TOY",
                "target_head": head.head_id,
                "split": split,
                "origin": origins,
                "dependency_start": origins - 4,
                "dependency_stop_exclusive": origins + 1,
                "latest_available_target_index": origins - 1,
                "y_true": values[origins] - values[origins - 1],
                "causal_history_floor": 0,
                "anchor_history_steps": 1,
                "sample_support_contract": "NATIVE_K_COMMON_ASSEMBLY_R1",
            }
        )
        path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(path, index=False)
    (shared / "dataset_views").mkdir(parents=True, exist_ok=True)
    (shared / "TASK_REGISTRY.json").write_text(
        json.dumps({"heads": [{
            "head_id": head.head_id,
            "task_id": head.task_id,
            "dataset": head.dataset,
            "target": head.target,
            "cadence_seconds": head.cadence_seconds,
            "h_steps": head.h_steps,
            "w_steps": head.w_steps,
            "w0_steps": head.w0_steps,
            "primary": True,
        }]}), encoding="utf-8"
    )
    (shared / "PROTOCOL.json").write_text(
        json.dumps({"tasks": [{"task_id": "TOY", "proxy_policies": ["primary"]}]}),
        encoding="utf-8",
    )
    (shared / "dataset_views" / "VIEW_REGISTRY.json").write_text(
        json.dumps([{"task_id": "TOY", "proxy_policy": "primary", "input_columns": ["x"]}]),
        encoding="utf-8",
    )
    (shared / "DATASET_HASHES.json").write_text(
        json.dumps({"toy": {"raw": "fixture"}}), encoding="utf-8"
    )
    return shared, view


def test_end_to_end_candidate_cache_reuses_without_mtime_change(tmp_path: Path) -> None:
    shared, view = _write_toy_shared(tmp_path)
    prepared = prepare_candidate(
        shared=shared,
        view=view,
        model_name="LSTM",
        histories=[4],
        history_labels={4: "4pts"},
        candidate_id="LSTM__L4pts__SMALL__lr0.0003",
        fit_row_cap=64,
        validation_row_cap=64,
    )
    cache = NeuralCandidateCache(tmp_path / "run").initialize()
    first = train_selection_candidate(
        cache=cache,
        prepared=prepared,
        code_commit="fixture-commit",
        device=torch.device("cpu"),
        max_epochs=2,
        patience=1,
    )
    assert first["status"] == "TRAINED_AND_CACHED"
    candidate_dir = Path(first["candidate_dir"])
    before = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in candidate_dir.iterdir()
        if path.is_file()
    }
    second = train_selection_candidate(
        cache=cache,
        prepared=prepared,
        code_commit="fixture-commit",
        device=torch.device("cpu"),
        max_epochs=2,
        patience=1,
    )
    assert second["status"] == "REUSED"
    after = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in candidate_dir.iterdir()
        if path.is_file()
    }
    assert after == before
    record = cache.validate_candidate(first["candidate_id"])
    assert record["validation_prediction_hash"]
