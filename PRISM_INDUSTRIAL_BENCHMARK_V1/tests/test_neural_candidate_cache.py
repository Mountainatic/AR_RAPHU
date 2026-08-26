from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from prism_benchmark.neural_candidate_cache import (
    CANDIDATE_MANIFEST_NAME,
    MANIFEST_NAME,
    CandidateConflictError,
    CandidateHashes,
    CandidateLockError,
    NeuralCandidateCache,
    SafetensorsUnavailableError,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_safetensors,
    sha256_file,
)


def _hashes(suffix: str = "a") -> CandidateHashes:
    return CandidateHashes(
        config_hash=f"config-{suffix}",
        data_hash=f"data-{suffix}",
        support_hash=f"support-{suffix}",
        sample_order_hash=f"order-{suffix}",
    )


def _complete(cache: NeuralCandidateCache, candidate_id: str, suffix: str = "a"):
    hashes = _hashes(suffix)
    with cache.lock(candidate_id, {"worker": "pytest"}):
        writer = cache.begin_candidate(
            candidate_id,
            hashes,
            {"model": "LSTM", "seed": 20260826},
        )
        writer.write_json("config.json", {"hidden": 16, "seed": 20260826})
        writer.write_npz(
            "preprocessing.npz",
            {"mean": np.array([1.0, 2.0]), "scale": np.array([0.5, 0.25])},
        )
        writer.write_bytes("weights.bin", b"portable-test-weights")
        record = writer.finalize(
            validation_metrics={"r2_level_reconstructed": 0.75},
            validation_prediction_hash="validation-prediction-sha256",
        )
    return hashes, record


def test_atomic_json_and_npz_replace_without_temporary_residue(tmp_path: Path) -> None:
    json_path = tmp_path / "state.json"
    npz_path = tmp_path / "arrays.npz"
    atomic_write_json(json_path, {"generation": 1})
    atomic_write_json(json_path, {"generation": 2})
    atomic_write_npz(npz_path, {"x": np.arange(4, dtype=np.float64)})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"generation": 2}
    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["x"], np.arange(4, dtype=np.float64))
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_npz_refuses_pickle_dependent_object_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="object arrays are forbidden"):
        atomic_write_npz(tmp_path / "bad.npz", {"bad": np.array([object()])})


def test_safetensors_is_atomic_or_fails_explicitly(tmp_path: Path) -> None:
    destination = tmp_path / "weights.safetensors"
    try:
        atomic_write_safetensors(destination, {"weight": np.arange(3, dtype=np.float32)})
    except SafetensorsUnavailableError as exc:
        assert "DEPENDENCY_UNAVAILABLE" in str(exc)
        assert not destination.exists()
    else:
        assert destination.is_file()
        assert destination.stat().st_size > 0


def test_candidate_lock_is_exclusive_and_owner_is_never_auto_broken(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    first = cache.lock("candidate", {"worker": "first"})
    first.acquire()
    owner = json.loads(first.path.read_text(encoding="utf-8"))
    assert owner["pid"] == os.getpid()
    assert owner["owner"] == {"worker": "first"}
    assert owner["automatic_break_forbidden"] is True
    second = cache.lock("candidate", {"worker": "second"})
    with pytest.raises(CandidateLockError, match="LOCK_HELD"):
        second.acquire()
    assert first.path.exists()
    first.release()
    with second:
        assert second.path.exists()
    assert not second.path.exists()


def test_finalize_records_files_and_only_exact_four_hashes_reuse(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    hashes, record = _complete(cache, "TEP/input_only/LSTM/candidate-1")
    final_dir = cache.candidate_dir("TEP/input_only/LSTM/candidate-1")
    assert final_dir.is_dir()
    assert not list(cache.staging_dir.iterdir())
    assert record["validation_prediction_hash"] == "validation-prediction-sha256"
    assert {item["name"] for item in record["files"]} == {
        "CANDIDATE_STATE.json",
        "config.json",
        "preprocessing.npz",
        "weights.bin",
    }
    for item in record["files"]:
        assert item["bytes"] >= 0
        assert len(item["sha256"]) == 64
        assert item["mtime_ns"] > 0
        assert item["mtime_utc"].endswith("+00:00")
    assert cache.reuse_status("TEP/input_only/LSTM/candidate-1", hashes).reusable
    mismatch = CandidateHashes(
        hashes.config_hash,
        hashes.data_hash,
        hashes.support_hash,
        "different-order",
    )
    decision = cache.reuse_status("TEP/input_only/LSTM/candidate-1", mismatch)
    assert decision.status == "HASH_MISMATCH"
    assert final_dir.is_dir()
    with cache.lock("TEP/input_only/LSTM/candidate-1"):
        with pytest.raises(CandidateConflictError, match="EXISTING_CANDIDATE"):
            cache.begin_candidate("TEP/input_only/LSTM/candidate-1", mismatch)


def test_manifest_updates_incrementally_without_dropping_candidates(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    _complete(cache, "one", "1")
    first = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    _complete(cache, "two", "2")
    second = cache.validate_manifest()
    assert set(second["candidates"]) == {"one", "two"}
    assert second["generation"] > first["generation"]
    for summary in second["candidates"].values():
        manifest_file = summary["candidate_manifest_file"]
        assert manifest_file["bytes"] > 0
        assert len(manifest_file["sha256"]) == 64
        assert manifest_file["mtime_ns"] > 0


def test_incomplete_candidate_is_preserved_and_resumed(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    hashes = _hashes()
    with cache.lock("resume-me"):
        with cache.begin_candidate("resume-me", hashes, {"epoch_budget": 5}) as writer:
            writer.write_npz("epoch_2.npz", {"weight": np.array([2.0])})
            writer.checkpoint_state(epoch=2, early_stopping_bad_epochs=1)
    assert cache.reuse_status("resume-me", hashes).status == "INCOMPLETE"
    with cache.lock("resume-me"):
        resumed = cache.begin_candidate("resume-me", hashes, {"epoch_budget": 5})
        assert resumed.resumed is True
        assert (resumed.staging_dir / "epoch_2.npz").is_file()
        resumed.write_npz("epoch_3.npz", {"weight": np.array([3.0])})
        resumed.finalize(validation_metrics={"loss": 0.1})
    assert cache.reuse_status("resume-me", hashes).reusable


def test_corrupt_candidate_is_moved_intact_and_retrain_requires_approval(
    tmp_path: Path,
) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    hashes, _ = _complete(cache, "corrupt-me")
    final_dir = cache.candidate_dir("corrupt-me")
    weights = final_dir / "weights.bin"
    original_hash = sha256_file(weights)
    weights.write_bytes(b"damaged-but-must-be-preserved")
    damaged_hash = sha256_file(weights)
    assert damaged_hash != original_hash
    decision = cache.reuse_status("corrupt-me", hashes)
    assert decision.status == "CORRUPT_QUARANTINED"
    assert not final_dir.exists()
    quarantine_path = Path(str(decision.record["quarantine_path"]))
    assert quarantine_path.is_dir()
    assert sha256_file(quarantine_path / "weights.bin") == damaged_hash
    marker = quarantine_path.parent / f"{quarantine_path.name}.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "CORRUPT_QUARANTINED"
    with cache.lock("corrupt-me"):
        with pytest.raises(CandidateConflictError, match="EXPLICIT_QUARANTINE_APPROVAL"):
            cache.begin_candidate("corrupt-me", hashes)
        writer = cache.begin_candidate(
            "corrupt-me", hashes, allow_retrain_after_quarantine=True
        )
        writer.abort("explicitly approved but test does not retrain")


def test_symlink_candidate_is_rejected_and_quarantined(tmp_path: Path) -> None:
    """A cache candidate may not resolve payloads outside its namespace."""

    cache = NeuralCandidateCache(tmp_path / "cache").initialize()
    hashes, _ = _complete(cache, "symlink-candidate")
    final_dir = cache.candidate_dir("symlink-candidate")
    outside = tmp_path / "outside-candidate"
    shutil.copytree(final_dir, outside)
    shutil.rmtree(final_dir)
    try:
        final_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable in this environment")

    decision = cache.reuse_status("symlink-candidate", hashes)
    assert decision.status == "CORRUPT_QUARANTINED"
    assert not final_dir.exists()
    assert outside.is_dir()


def test_seal_sets_deletion_forbidden_and_read_only_then_revalidates(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    hashes, _ = _complete(cache, "seal-me")
    with cache.lock("seal-me"):
        sealed = cache.seal_candidate("seal-me", hashes)
    assert sealed["deletion_forbidden"] is True
    final_dir = cache.candidate_dir("seal-me")
    assert not (final_dir / "weights.bin").stat().st_mode & stat.S_IWUSR
    assert cache.validate_candidate("seal-me", hashes)["deletion_forbidden"] is True
    assert cache.reuse_status("seal-me", hashes).reusable
    manifest = cache.seal_manifest()
    assert manifest["status"] == "CANDIDATE_CHECKPOINTS_SEALED"
    assert manifest["sealed"] is True
    assert manifest["deletion_forbidden"] is True
    assert not cache.manifest_path.stat().st_mode & stat.S_IWUSR
    assert cache.validate_manifest()["deletion_forbidden"] is True
    # A final sealed run remains reusable without mutating the global manifest.
    before = cache.manifest_path.read_bytes()
    assert cache.reuse_status("seal-me", hashes).reusable
    assert cache.manifest_path.read_bytes() == before


def test_manifest_and_candidate_manifests_are_not_self_hashed(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    _complete(cache, "self-exclusion")
    final_dir = cache.candidate_dir("self-exclusion")
    local = json.loads((final_dir / CANDIDATE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert CANDIDATE_MANIFEST_NAME not in {item["name"] for item in local["files"]}
    global_manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "manifest_hash" in global_manifest


def test_explicit_resume_reclaims_only_provably_abandoned_lock(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-now", encoding="utf-8")
    path = cache._lock_path("lost")
    path.write_text(
        json.dumps(
            {
                "candidate_id": "lost",
                "pid": 999999,
                "hostname": __import__("socket").gethostname(),
                "boot_id": "boot-now",
                "proc_start_ticks": 123,
            }
        ),
        encoding="utf-8",
    )
    evidence = cache.reclaim_abandoned_lock(
        "lost", requested_by="pytest --resume", proc_root=proc
    )
    assert evidence["status"] == "ABANDONED_LOCK_RECLAIMED_FOR_EXPLICIT_RESUME"
    assert evidence["reasons"] == ["OWNER_PID_NOT_RUNNING"]
    assert not path.exists()
    assert Path(evidence["preserved_lock_path"]).is_file()


def test_explicit_resume_refuses_live_lock(tmp_path: Path) -> None:
    cache = NeuralCandidateCache(tmp_path).initialize()
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-now", encoding="utf-8")
    pid = 42
    (proc / str(pid)).mkdir()
    # The parser takes token index 19 after the closing parenthesis (field 22).
    fields = ["S", *(["0"] * 18), "123", *(["0"] * 4)]
    (proc / str(pid) / "stat").write_text(
        f"{pid} (worker) {' '.join(fields)}", encoding="utf-8"
    )
    cache._lock_path("live").write_text(
        json.dumps(
            {
                "candidate_id": "live",
                "pid": pid,
                "hostname": __import__("socket").gethostname(),
                "boot_id": "boot-now",
                "proc_start_ticks": 123,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CandidateLockError, match="MAY_STILL_BE_ACTIVE"):
        cache.reclaim_abandoned_lock(
            "live", requested_by="pytest --resume", proc_root=proc
        )
