"""Crash-safe, permanent cache primitives for Neural-3 candidate training.

The cache deliberately separates an incomplete staging directory from the
immutable completed candidate directory.  Candidate locks are created with
``O_EXCL`` and are never considered stale automatically: an operator must
inspect and remove an abandoned lock explicitly.

All persisted formats in this module are portable and pickle-free.  A complete
candidate is reusable only after all four provenance hashes and every recorded
file attribute have been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = "PRISM_NEURAL_CANDIDATE_CACHE_V1"
LOCK_FORMAT = "PRISM_NEURAL_CANDIDATE_LOCK_V1"
MANIFEST_NAME = "CANDIDATE_CHECKPOINT_MANIFEST.json"
CANDIDATE_MANIFEST_NAME = "CANDIDATE_MANIFEST.json"
CANDIDATE_STATE_NAME = "CANDIDATE_STATE.json"


class CandidateCacheError(RuntimeError):
    """Base class for fail-closed cache errors."""


class CandidateLockError(CandidateCacheError):
    """Raised when a candidate lock cannot safely be acquired or released."""


class CandidateIntegrityError(CandidateCacheError):
    """Raised when persisted candidate content fails validation."""


class CandidateConflictError(CandidateCacheError):
    """Raised before an existing candidate or incompatible staging is reused."""


class SafetensorsUnavailableError(CandidateCacheError):
    """Raised when safetensors output was requested without the dependency."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _linux_process_identity(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Return the boot and process-start identity used for explicit lock recovery."""

    result: dict[str, Any] = {}
    try:
        result["boot_id"] = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        result["boot_id"] = None
    try:
        # /proc/<pid>/stat field 22 is process start time.  The comm field may
        # contain spaces, so split only after the final closing parenthesis.
        stat_line = (proc_root / str(int(pid)) / "stat").read_text(
            encoding="utf-8"
        )
        remainder = stat_line.rsplit(")", 1)[1].strip().split()
        result["proc_start_ticks"] = int(remainder[19])
        result["process_exists"] = True
    except (OSError, ValueError, IndexError):
        result["proc_start_ticks"] = None
        result["process_exists"] = False
    return result


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    """Return a canonical JSON SHA-256 for configuration/provenance values."""

    return hashlib.sha256(_canonical_json(value).rstrip(b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync (not available on every Windows runtime)."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _temporary_path(path: Path) -> Path:
    # Do not repeat the destination basename: deeply nested experiment names can
    # otherwise cross the legacy Windows MAX_PATH boundary during local QA.
    return path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex[:16]}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace one UTF-8 canonical JSON file."""

    _atomic_write_bytes(Path(path), _canonical_json(value))


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed, pickle-free NumPy archive."""

    normalized: dict[str, np.ndarray] = {}
    for name, value in sorted(arrays.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("NPZ array names must be non-empty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden in portable NPZ: {name}")
        normalized[name] = array
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w+b") as handle:
            descriptor = None
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def atomic_write_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    """Atomically write safetensors, failing explicitly if it is unavailable."""

    try:
        from safetensors.numpy import save as save_safetensors
    except ImportError as exc:  # pragma: no cover - availability is environment-specific
        raise SafetensorsUnavailableError(
            "STOP_SAFETENSORS_REQUESTED_BUT_DEPENDENCY_UNAVAILABLE"
        ) from exc
    normalized: dict[str, np.ndarray] = {}
    for name, value in sorted(tensors.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("safetensors names must be non-empty strings")
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError(f"object tensors are forbidden: {name}")
        normalized[name] = array
    _atomic_write_bytes(Path(path), save_safetensors(normalized))


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Capture the immutable attributes required for later reuse validation."""

    path = Path(path)
    info = path.stat()
    name = path.name if relative_to is None else path.relative_to(relative_to).as_posix()
    return {
        "name": name,
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
        "mtime": float(info.st_mtime),
        "mtime_ns": int(info.st_mtime_ns),
        "mtime_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
    }


@dataclass(frozen=True)
class CandidateHashes:
    config_hash: str
    data_hash: str
    support_hash: str
    sample_order_hash: str

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "support_hash": self.support_hash,
            "sample_order_hash": self.sample_order_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateHashes":
        required = ("config_hash", "data_hash", "support_hash", "sample_order_hash")
        if set(value) != set(required):
            raise ValueError(f"candidate hash keys must be exactly {required}")
        return cls(**{name: value[name] for name in required})  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReuseDecision:
    status: str
    candidate_id: str
    candidate_dir: Path
    reason: str | None = None
    record: Mapping[str, Any] | None = None

    @property
    def reusable(self) -> bool:
        return self.status == "REUSABLE"


class CandidateLock:
    """An ``O_EXCL`` lock with inspectable ownership and no stale-lock policy."""

    def __init__(
        self,
        path: Path,
        candidate_id: str,
        *,
        owner: Mapping[str, Any] | None = None,
        timeout_seconds: float = 0.0,
        poll_seconds: float = 0.05,
        on_acquire: Any = None,
        on_release: Any = None,
    ) -> None:
        self.path = Path(path)
        self.candidate_id = candidate_id
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._on_acquire = on_acquire
        self._on_release = on_release
        self._token = uuid.uuid4().hex
        self.metadata = {
            "format": LOCK_FORMAT,
            "candidate_id": candidate_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "thread_id": threading.get_ident(),
            "acquired_utc": _utc_now(),
            "token": self._token,
            "owner": dict(owner or {}),
            "automatic_break_forbidden": True,
            **_linux_process_identity(os.getpid()),
        }
        self.held = False

    def read_owner(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"unreadable": True, "error": f"{type(exc).__name__}: {exc}"}

    def acquire(self) -> "CandidateLock":
        if self.held:
            raise CandidateLockError(f"lock is already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(self.timeout_seconds, 0.0)
        while True:
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                payload = _canonical_json(self.metadata)
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = None
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(self.path.parent)
                self.held = True
                if self._on_acquire is not None:
                    self._on_acquire(self)
                return self
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    owner = self.read_owner()
                    raise CandidateLockError(
                        f"STOP_CANDIDATE_LOCK_HELD:{self.path}:{owner}"
                    ) from exc
                time.sleep(self.poll_seconds)
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                # Only this code path can own a partially initialized new file.
                if self.path.exists():
                    try:
                        observed = json.loads(self.path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        observed = {}
                    if observed.get("token") == self._token:
                        self.path.unlink()
                raise

    def release(self) -> None:
        if not self.held:
            return
        observed = self.read_owner()
        if observed.get("token") != self._token:
            raise CandidateLockError(
                f"STOP_LOCK_OWNER_CHANGED_REFUSING_TO_REMOVE:{self.path}"
            )
        self.path.unlink()
        _fsync_directory(self.path.parent)
        self.held = False
        if self._on_release is not None:
            self._on_release(self)

    def __enter__(self) -> "CandidateLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.release()


def _safe_slug(candidate_id: str) -> str:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    readable = re.sub(r"[^A-Za-z0-9_.=-]+", "_", candidate_id).strip("._")
    readable = readable[:40] or "candidate"
    return f"{readable}__{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:16]}"


def candidate_key(*parts: Any) -> str:
    """Create an unambiguous logical candidate id from runner dimensions."""

    if not parts:
        raise ValueError("candidate_key requires at least one part")
    return "__".join(str(part).replace("\\", "_").replace("/", "_") for part in parts)


def _relative_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"invalid candidate file name: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"candidate file must be a safe relative path: {name!r}")
    if relative.name in (CANDIDATE_MANIFEST_NAME, CANDIDATE_STATE_NAME):
        raise ValueError(f"reserved candidate file name: {relative.name}")
    return relative


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_hash", None)
    return stable_hash(payload)


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


class CandidateWriter:
    """Writer for one locked staging directory."""

    def __init__(
        self,
        cache: "NeuralCandidateCache",
        candidate_id: str,
        hashes: CandidateHashes,
        staging_dir: Path,
        metadata: Mapping[str, Any],
        *,
        resumed: bool,
    ) -> None:
        self.cache = cache
        self.candidate_id = candidate_id
        self.hashes = hashes
        self.staging_dir = staging_dir
        self.metadata = dict(metadata)
        self.resumed = resumed
        self.finalized = False

    def _path(self, name: str) -> Path:
        return self.staging_dir.joinpath(*_relative_name(name).parts)

    def write_json(self, name: str, value: Any) -> Path:
        destination = self._path(name)
        atomic_write_json(destination, value)
        return destination

    def write_npz(self, name: str, arrays: Mapping[str, np.ndarray]) -> Path:
        destination = self._path(name)
        atomic_write_npz(destination, arrays)
        return destination

    def write_safetensors(self, name: str, tensors: Mapping[str, Any]) -> Path:
        destination = self._path(name)
        atomic_write_safetensors(destination, tensors)
        return destination

    def write_bytes(self, name: str, payload: bytes) -> Path:
        destination = self._path(name)
        _atomic_write_bytes(destination, bytes(payload))
        return destination

    def copy_file(self, name: str, source: Path) -> Path:
        destination = self._path(name)
        with Path(source).open("rb") as handle:
            _atomic_write_bytes(destination, handle.read())
        return destination

    def _write_state(self, status: str, **extra: Any) -> dict[str, Any]:
        previous_created = None
        state_path = self.staging_dir / CANDIDATE_STATE_NAME
        if state_path.is_file():
            try:
                previous_created = json.loads(state_path.read_text(encoding="utf-8")).get(
                    "created_utc"
                )
            except (OSError, ValueError):
                previous_created = None
        state = {
            "format": FORMAT_VERSION,
            "candidate_id": self.candidate_id,
            "status": status,
            "hashes": self.hashes.as_dict(),
            "metadata": self.metadata,
            "created_utc": previous_created or _utc_now(),
            "updated_utc": _utc_now(),
            "deletion_forbidden": True,
            **extra,
        }
        atomic_write_json(state_path, state)
        return state

    def checkpoint_state(self, **progress: Any) -> Path:
        """Atomically persist resumable optimizer/epoch/early-stop progress metadata."""

        self._write_state("INCOMPLETE", progress=dict(progress))
        return self.staging_dir / CANDIDATE_STATE_NAME

    def finalize(
        self,
        *,
        validation_metrics: Mapping[str, Any] | None = None,
        validation_prediction_hash: str | None = None,
        seal: bool = False,
    ) -> dict[str, Any]:
        if self.finalized:
            raise CandidateConflictError("candidate writer was already finalized")
        self.cache._assert_lock_owned(self.candidate_id)
        final_dir = self.cache.candidate_dir(self.candidate_id)
        if final_dir.exists():
            raise CandidateConflictError(
                f"STOP_REFUSING_TO_OVERWRITE_EXISTING_CANDIDATE:{final_dir}"
            )
        state = self._write_state(
            "COMPLETE",
            validation_metrics=dict(validation_metrics or {}),
            validation_prediction_hash=validation_prediction_hash,
        )
        payload_files = [
            path
            for path in sorted(self.staging_dir.rglob("*"))
            if path.is_file() and path.name != CANDIDATE_MANIFEST_NAME
        ]
        record: dict[str, Any] = {
            "format": FORMAT_VERSION,
            "candidate_id": self.candidate_id,
            "status": "COMPLETE",
            "hashes": self.hashes.as_dict(),
            "metadata": self.metadata,
            "created_utc": state["created_utc"],
            "completed_utc": _utc_now(),
            # Permanence starts at first successful finalize.  ``sealed`` is a
            # separate physical read-only state applied after selection freeze.
            "deletion_forbidden": True,
            "sealed": False,
            "validation_prediction_hash": validation_prediction_hash,
            "validation_metrics": dict(validation_metrics or {}),
            "files": [file_record(path, relative_to=self.staging_dir) for path in payload_files],
        }
        record["record_hash"] = _record_hash(record)
        atomic_write_json(self.staging_dir / CANDIDATE_MANIFEST_NAME, record)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.staging_dir, final_dir)
        _fsync_directory(final_dir.parent)
        self.finalized = True
        self.cache._update_manifest_entry(final_dir, record)
        if seal:
            return self.cache.seal_candidate(self.candidate_id, self.hashes)
        return record

    def abort(self, reason: str | None = None) -> None:
        """Preserve an incomplete candidate for resume; never delete its files."""

        if not self.finalized and self.staging_dir.is_dir():
            self._write_state("INCOMPLETE", interruption_reason=reason)

    def __enter__(self) -> "CandidateWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del traceback
        if not self.finalized:
            reason = None if exc_type is None else f"{exc_type.__name__}: {exc}"
            self.abort(reason)


class NeuralCandidateCache:
    """Permanent candidate cache rooted in one independent experiment namespace."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)
        self.candidates_dir = self.run_root / "candidates"
        self.staging_dir = self.run_root / "candidate_staging"
        self.locks_dir = self.run_root / "candidate_locks"
        self.quarantine_dir = self.run_root / "candidate_quarantine"
        self.manifest_path = self.run_root / MANIFEST_NAME
        self._active_locks: dict[str, CandidateLock] = {}
        self._initialized = False

    def initialize(self) -> "NeuralCandidateCache":
        if self._initialized:
            return self
        for path in (
            self.run_root,
            self.candidates_dir,
            self.staging_dir,
            self.locks_dir,
            self.quarantine_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        # Serialize first creation so two fresh workers cannot both observe a
        # missing file and let the later empty manifest erase an early update.
        with self._manifest_lock():
            if not self.manifest_path.exists():
                manifest = {
                    "format": FORMAT_VERSION,
                    "status": "ACTIVE_CANDIDATE_CACHE",
                    "sealed": False,
                    "created_utc": _utc_now(),
                    "updated_utc": _utc_now(),
                    "generation": 0,
                    "deletion_forbidden": False,
                    "candidates": {},
                }
                manifest["manifest_hash"] = _manifest_hash(manifest)
                atomic_write_json(self.manifest_path, manifest)
            else:
                self.validate_manifest(validate_candidates=False)
        self._initialized = True
        return self

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / _safe_slug(candidate_id)

    def _lock_path(self, candidate_id: str) -> Path:
        return self.locks_dir / f"{_safe_slug(candidate_id)}.lock"

    def reclaim_abandoned_lock(
        self,
        candidate_id: str,
        *,
        requested_by: str,
        proc_root: Path = Path("/proc"),
    ) -> dict[str, Any]:
        """Explicitly preserve and release a provably abandoned local lock.

        This is never called by ``lock()``.  A resume command must request it,
        and an owner that could still be alive always wins.
        """

        if not requested_by:
            raise CandidateLockError("STOP_ABANDONED_LOCK_RECLAIM_REQUIRES_REQUESTOR")
        path = self._lock_path(candidate_id)
        if not path.is_file():
            return {"status": "NO_LOCK", "candidate_id": candidate_id}
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CandidateLockError(
                f"STOP_UNREADABLE_LOCK_REQUIRES_OPERATOR_REVIEW:{path}"
            ) from exc
        if owner.get("hostname") != socket.gethostname():
            raise CandidateLockError(
                f"STOP_REMOTE_HOST_LOCK_CANNOT_BE_RECLAIMED:{path}:{owner.get('hostname')}"
            )
        pid = int(owner.get("pid", -1))
        current = _linux_process_identity(pid, proc_root)
        owner_boot = owner.get("boot_id")
        owner_start = owner.get("proc_start_ticks")
        current_boot = current.get("boot_id")
        reasons: list[str] = []
        if owner_boot and current_boot and owner_boot != current_boot:
            reasons.append("BOOT_ID_CHANGED")
        elif current.get("process_exists") is False:
            reasons.append("OWNER_PID_NOT_RUNNING")
        elif owner_start is not None and current.get("proc_start_ticks") != owner_start:
            reasons.append("PID_REUSED_WITH_DIFFERENT_START_TICKS")
        else:
            raise CandidateLockError(
                f"STOP_LOCK_OWNER_MAY_STILL_BE_ACTIVE:{path}:{owner}"
            )
        abandoned_root = self.locks_dir / "abandoned"
        abandoned_root.mkdir(parents=True, exist_ok=True)
        destination = abandoned_root / (
            f"{path.name}.{uuid.uuid4().hex}.ABANDONED"
        )
        os.replace(path, destination)
        evidence = {
            "status": "ABANDONED_LOCK_RECLAIMED_FOR_EXPLICIT_RESUME",
            "candidate_id": candidate_id,
            "requested_by": requested_by,
            "reasons": reasons,
            "owner": owner,
            "observed_identity": current,
            "preserved_lock_path": str(destination),
            "reclaimed_utc": _utc_now(),
            "active_lock_was_never_broken": True,
        }
        atomic_write_json(destination.with_suffix(destination.suffix + ".json"), evidence)
        _fsync_directory(abandoned_root)
        return evidence

    def _register_lock(self, lock: CandidateLock) -> None:
        if lock.candidate_id in self._active_locks:
            raise CandidateLockError(f"duplicate local lock: {lock.candidate_id}")
        self._active_locks[lock.candidate_id] = lock

    def _unregister_lock(self, lock: CandidateLock) -> None:
        if self._active_locks.get(lock.candidate_id) is lock:
            del self._active_locks[lock.candidate_id]

    def lock(
        self,
        candidate_id: str,
        owner: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float = 0.0,
    ) -> CandidateLock:
        self.initialize()
        return CandidateLock(
            self._lock_path(candidate_id),
            candidate_id,
            owner=owner,
            timeout_seconds=timeout_seconds,
            on_acquire=self._register_lock,
            on_release=self._unregister_lock,
        )

    def _assert_lock_owned(self, candidate_id: str) -> None:
        lock = self._active_locks.get(candidate_id)
        if lock is None or not lock.held:
            raise CandidateLockError(
                f"STOP_CANDIDATE_LOCK_REQUIRED:{candidate_id}"
            )
        observed = lock.read_owner()
        if observed.get("token") != lock.metadata["token"]:
            raise CandidateLockError(f"STOP_CANDIDATE_LOCK_OWNERSHIP_LOST:{candidate_id}")

    def _staging_matches(self, candidate_id: str) -> list[Path]:
        prefix = f"{_safe_slug(candidate_id)}.partial."
        if not self.staging_dir.is_dir():
            return []
        matches = []
        for path in self.staging_dir.iterdir():
            if not path.name.startswith(prefix):
                continue
            if path.is_symlink():
                raise CandidateIntegrityError(
                    f"STOP_INCOMPLETE_CANDIDATE_SYMLINK:{path}"
                )
            if path.is_dir():
                matches.append(path)
        return sorted(matches)

    def begin_candidate(
        self,
        candidate_id: str,
        hashes: CandidateHashes,
        metadata: Mapping[str, Any] | None = None,
        *,
        resume: bool = True,
        allow_retrain_after_quarantine: bool = False,
    ) -> CandidateWriter:
        self.initialize()
        self._assert_lock_owned(candidate_id)
        if self.candidate_dir(candidate_id).exists():
            raise CandidateConflictError(
                f"STOP_EXISTING_CANDIDATE_MUST_BE_INSPECTED_FIRST:{candidate_id}"
            )
        global_record = self._manifest_candidate(candidate_id)
        if (
            global_record
            and global_record.get("status") == "CORRUPT_QUARANTINED"
            and not allow_retrain_after_quarantine
        ):
            raise CandidateConflictError(
                f"STOP_RETRAIN_REQUIRES_EXPLICIT_QUARANTINE_APPROVAL:{candidate_id}"
            )
        stages = self._staging_matches(candidate_id)
        if stages and not resume:
            raise CandidateConflictError(f"STOP_INCOMPLETE_CANDIDATE_EXISTS:{stages}")
        if len(stages) > 1:
            raise CandidateConflictError(f"STOP_MULTIPLE_INCOMPLETE_CANDIDATES:{stages}")
        if stages:
            state_path = stages[0] / CANDIDATE_STATE_NAME
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CandidateIntegrityError(
                    f"STOP_INCOMPLETE_STATE_UNREADABLE:{state_path}"
                ) from exc
            if (
                state.get("format") != FORMAT_VERSION
                or state.get("status") != "INCOMPLETE"
                or state.get("candidate_id") != candidate_id
                or state.get("hashes") != hashes.as_dict()
                or state.get("deletion_forbidden") is not True
                or not isinstance(state.get("metadata"), Mapping)
                or not isinstance(state.get("progress", {}), Mapping)
            ):
                raise CandidateIntegrityError(
                    f"STOP_INCOMPLETE_STATE_SCHEMA_INVALID:{state_path}"
                )
            for path in stages[0].rglob("*"):
                if path.is_symlink():
                    raise CandidateIntegrityError(
                        f"STOP_INCOMPLETE_CANDIDATE_SYMLINK:{path}"
                    )
            stored_metadata = state.get("metadata", {})
            if metadata is not None and dict(metadata) != stored_metadata:
                raise CandidateConflictError(
                    f"STOP_INCOMPLETE_CANDIDATE_METADATA_MISMATCH:{stages[0]}"
                )
            return CandidateWriter(
                self,
                candidate_id,
                hashes,
                stages[0],
                stored_metadata,
                resumed=True,
            )
        stage = self.staging_dir / f"{_safe_slug(candidate_id)}.partial.{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        writer = CandidateWriter(
            self, candidate_id, hashes, stage, dict(metadata or {}), resumed=False
        )
        writer.checkpoint_state(created_by_pid=os.getpid())
        return writer

    def _read_candidate_record(self, candidate_dir: Path) -> dict[str, Any]:
        if candidate_dir.is_symlink():
            raise CandidateIntegrityError(
                f"STOP_CANDIDATE_DIRECTORY_SYMLINK:{candidate_dir}"
            )
        manifest_path = candidate_dir / CANDIDATE_MANIFEST_NAME
        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CandidateIntegrityError(
                f"STOP_CANDIDATE_MANIFEST_UNREADABLE:{manifest_path}"
            ) from exc
        if record.get("format") != FORMAT_VERSION or record.get("status") != "COMPLETE":
            raise CandidateIntegrityError(f"STOP_INVALID_CANDIDATE_MANIFEST:{manifest_path}")
        if _record_hash(record) != record.get("record_hash"):
            raise CandidateIntegrityError(f"STOP_CANDIDATE_RECORD_HASH_MISMATCH:{manifest_path}")
        records = record.get("files")
        if not isinstance(records, list):
            raise CandidateIntegrityError(f"STOP_CANDIDATE_FILE_LIST_MISSING:{manifest_path}")
        expected_names = {item.get("name") for item in records if isinstance(item, dict)}
        if len(expected_names) != len(records) or None in expected_names:
            raise CandidateIntegrityError(f"STOP_CANDIDATE_FILE_LIST_INVALID:{manifest_path}")
        actual_names = set()
        for path in candidate_dir.rglob("*"):
            # Candidate artifacts are copied into the cache and must never
            # resolve outside it.  Data overlays may use symlinks elsewhere,
            # but a cached weight or manifest symlink is always invalid.
            if path.is_symlink():
                raise CandidateIntegrityError(
                    f"STOP_CANDIDATE_SYMLINK_NOT_ALLOWED:{path}"
                )
            if path.is_file() and path.name != CANDIDATE_MANIFEST_NAME:
                actual_names.add(path.relative_to(candidate_dir).as_posix())
        if actual_names != expected_names:
            raise CandidateIntegrityError(
                f"STOP_CANDIDATE_FILE_SET_MISMATCH:{candidate_dir}:"
                f"expected={sorted(expected_names)}:actual={sorted(actual_names)}"
            )
        for expected in records:
            path = candidate_dir.joinpath(*PurePosixPath(expected["name"]).parts)
            observed = file_record(path, relative_to=candidate_dir)
            for field in ("bytes", "sha256", "mtime_ns"):
                if observed[field] != expected.get(field):
                    raise CandidateIntegrityError(
                        f"STOP_CANDIDATE_FILE_{field.upper()}_MISMATCH:{path}"
                    )
        return record

    def validate_candidate(
        self, candidate_id: str, hashes: CandidateHashes | None = None
    ) -> dict[str, Any]:
        candidate_dir = self.candidate_dir(candidate_id)
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise CandidateIntegrityError(f"STOP_CANDIDATE_DIRECTORY_MISSING:{candidate_dir}")
        record = self._read_candidate_record(candidate_dir)
        if record.get("candidate_id") != candidate_id:
            raise CandidateIntegrityError(f"STOP_CANDIDATE_ID_MISMATCH:{candidate_dir}")
        if hashes is not None and record.get("hashes") != hashes.as_dict():
            raise CandidateConflictError(f"STOP_CANDIDATE_PROVENANCE_HASH_MISMATCH:{candidate_id}")
        return record

    def reuse_status(
        self,
        candidate_id: str,
        hashes: CandidateHashes,
        *,
        quarantine_corrupt: bool = True,
    ) -> ReuseDecision:
        self.initialize()
        candidate_dir = self.candidate_dir(candidate_id)
        if not candidate_dir.exists():
            prior = self._manifest_candidate(candidate_id)
            if prior and prior.get("status") == "CORRUPT_QUARANTINED":
                return ReuseDecision(
                    "CORRUPT_QUARANTINED", candidate_id, candidate_dir,
                    str(prior.get("reason")), prior,
                )
            if self._staging_matches(candidate_id):
                return ReuseDecision("INCOMPLETE", candidate_id, candidate_dir)
            return ReuseDecision("MISSING", candidate_id, candidate_dir)
        try:
            record = self.validate_candidate(candidate_id)
        except CandidateIntegrityError as exc:
            if not quarantine_corrupt:
                return ReuseDecision("CORRUPT", candidate_id, candidate_dir, str(exc))
            quarantined = self.quarantine_candidate(candidate_id, str(exc))
            return ReuseDecision(
                "CORRUPT_QUARANTINED", candidate_id, candidate_dir, str(exc), quarantined
            )
        if record.get("hashes") != hashes.as_dict():
            return ReuseDecision(
                "HASH_MISMATCH",
                candidate_id,
                candidate_dir,
                "all config/data/support/sample-order hashes must match exactly",
                record,
            )
        self._update_manifest_entry(candidate_dir, record)
        return ReuseDecision("REUSABLE", candidate_id, candidate_dir, record=record)

    def quarantine_candidate(self, candidate_id: str, reason: str) -> dict[str, Any]:
        self.initialize()
        owned = candidate_id in self._active_locks
        if not owned:
            with self.lock(candidate_id, {"operation": "quarantine"}):
                return self._quarantine_candidate_locked(candidate_id, reason)
        return self._quarantine_candidate_locked(candidate_id, reason)

    def quarantine_staging(
        self, candidate_id: str, staging_dir: Path, reason: str
    ) -> dict[str, Any]:
        """Move a corrupt incomplete candidate aside without deleting evidence."""

        self._assert_lock_owned(candidate_id)
        source = Path(staging_dir)
        if not source.is_dir() or source.is_symlink():
            raise CandidateIntegrityError(
                f"STOP_STAGING_QUARANTINE_SOURCE_INVALID:{source}"
            )
        try:
            source.resolve().relative_to(self.staging_dir.resolve())
        except ValueError as error:
            raise CandidateIntegrityError(
                f"STOP_STAGING_QUARANTINE_SOURCE_OUTSIDE_CACHE:{source}"
            ) from error
        destination = self.quarantine_dir / (
            f"{_safe_slug(candidate_id)}.STAGING_CORRUPT.{uuid.uuid4().hex}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        _fsync_directory(destination.parent)
        marker = {
            "format": FORMAT_VERSION,
            "candidate_id": candidate_id,
            "status": "CORRUPT_QUARANTINED",
            "source_kind": "INCOMPLETE_STAGING",
            "reason": reason,
            "quarantined_utc": _utc_now(),
            "original_path": str(source),
            "quarantine_path": str(destination),
            "preserved_without_overwrite": True,
            "retrain_requires_explicit_approval": True,
        }
        atomic_write_json(
            destination.parent / f"{destination.name}.json", marker
        )
        self._update_global_candidate(candidate_id, marker)
        return marker

    def _quarantine_candidate_locked(self, candidate_id: str, reason: str) -> dict[str, Any]:
        self._assert_lock_owned(candidate_id)
        source = self.candidate_dir(candidate_id)
        if not source.exists():
            raise CandidateIntegrityError(f"STOP_QUARANTINE_SOURCE_MISSING:{source}")
        destination = self.quarantine_dir / (
            f"{_safe_slug(candidate_id)}.CORRUPT_QUARANTINED.{uuid.uuid4().hex}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        _fsync_directory(source.parent)
        marker = {
            "format": FORMAT_VERSION,
            "candidate_id": candidate_id,
            "status": "CORRUPT_QUARANTINED",
            "reason": reason,
            "quarantined_utc": _utc_now(),
            "original_path": str(source),
            "quarantine_path": str(destination),
            "preserved_without_overwrite": True,
            "retrain_requires_explicit_approval": True,
        }
        atomic_write_json(destination.parent / f"{destination.name}.json", marker)
        self._update_global_candidate(candidate_id, marker)
        return marker

    def _manifest_lock(self) -> CandidateLock:
        return CandidateLock(
            self.locks_dir / ".candidate_manifest.lock",
            "__GLOBAL_MANIFEST__",
            owner={"operation": "incremental_manifest_update"},
            timeout_seconds=60.0,
        )

    def _read_global_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CandidateIntegrityError(
                f"STOP_GLOBAL_CANDIDATE_MANIFEST_UNREADABLE:{self.manifest_path}"
            ) from exc
        if manifest.get("format") != FORMAT_VERSION:
            raise CandidateIntegrityError("STOP_GLOBAL_CANDIDATE_MANIFEST_FORMAT_MISMATCH")
        if _manifest_hash(manifest) != manifest.get("manifest_hash"):
            raise CandidateIntegrityError("STOP_GLOBAL_CANDIDATE_MANIFEST_HASH_MISMATCH")
        if not isinstance(manifest.get("candidates"), dict):
            raise CandidateIntegrityError("STOP_GLOBAL_CANDIDATE_MANIFEST_SCHEMA_INVALID")
        return manifest

    def _manifest_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        return self._read_global_manifest()["candidates"].get(candidate_id)

    def _update_global_candidate(
        self, candidate_id: str, summary: Mapping[str, Any]
    ) -> None:
        with self._manifest_lock():
            manifest = self._read_global_manifest()
            if manifest.get("deletion_forbidden") is True:
                existing = manifest["candidates"].get(candidate_id)
                immutable_fields = (
                    "candidate_id",
                    "status",
                    "hashes",
                    "record_hash",
                    "candidate_dir",
                    "candidate_manifest_file",
                    "deletion_forbidden",
                )
                if existing is not None and all(
                    existing.get(field) == summary.get(field) for field in immutable_fields
                ):
                    # Read-only reuse after the final seal is allowed, but must
                    # not rewrite the manifest merely to refresh a timestamp.
                    return
                raise CandidateConflictError("STOP_SEALED_GLOBAL_MANIFEST_CANNOT_BE_UPDATED")
            manifest["candidates"][candidate_id] = dict(summary)
            manifest["generation"] = int(manifest.get("generation", 0)) + 1
            manifest["updated_utc"] = _utc_now()
            manifest["manifest_hash"] = _manifest_hash(manifest)
            atomic_write_json(self.manifest_path, manifest)

    def _update_manifest_entry(
        self, candidate_dir: Path, record: Mapping[str, Any]
    ) -> None:
        summary = {
            "format": FORMAT_VERSION,
            "candidate_id": record["candidate_id"],
            "status": record["status"],
            "hashes": dict(record["hashes"]),
            "record_hash": record["record_hash"],
            "candidate_dir": str(candidate_dir),
            "candidate_manifest_file": file_record(
                candidate_dir / CANDIDATE_MANIFEST_NAME, relative_to=candidate_dir
            ),
            "deletion_forbidden": bool(record.get("deletion_forbidden")),
            "updated_utc": _utc_now(),
        }
        self._update_global_candidate(str(record["candidate_id"]), summary)

    def refresh_manifest_record(self, candidate_id: str) -> dict[str, Any]:
        record = self.validate_candidate(candidate_id)
        self._update_manifest_entry(self.candidate_dir(candidate_id), record)
        return record

    def seal_candidate(
        self, candidate_id: str, hashes: CandidateHashes | None = None
    ) -> dict[str, Any]:
        self._assert_lock_owned(candidate_id)
        candidate_dir = self.candidate_dir(candidate_id)
        record = self.validate_candidate(candidate_id, hashes)
        if not record.get("sealed"):
            state_path = candidate_dir / CANDIDATE_STATE_NAME
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["deletion_forbidden"] = True
            state["sealed_utc"] = _utc_now()
            atomic_write_json(state_path, state)
            files = [
                path
                for path in sorted(candidate_dir.rglob("*"))
                if path.is_file() and path.name != CANDIDATE_MANIFEST_NAME
            ]
            record["files"] = [file_record(path, relative_to=candidate_dir) for path in files]
            record["deletion_forbidden"] = True
            record["sealed"] = True
            record["sealed_utc"] = state["sealed_utc"]
            record["record_hash"] = _record_hash(record)
            atomic_write_json(candidate_dir / CANDIDATE_MANIFEST_NAME, record)
        self.validate_candidate(candidate_id, hashes)
        self._update_manifest_entry(candidate_dir, record)
        self._seal_tree(candidate_dir)
        return record

    @staticmethod
    def _seal_tree(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            elif path.is_dir():
                path.chmod(
                    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
                    | stat.S_IROTH | stat.S_IXOTH
                )
        root.chmod(
            stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH
        )

    def validate_manifest(self, *, validate_candidates: bool = True) -> dict[str, Any]:
        manifest = self._read_global_manifest()
        if validate_candidates:
            for candidate_id, summary in manifest["candidates"].items():
                if summary.get("status") != "COMPLETE":
                    continue
                record = self.validate_candidate(candidate_id)
                if record.get("record_hash") != summary.get("record_hash"):
                    raise CandidateIntegrityError(
                        f"STOP_GLOBAL_RECORD_HASH_MISMATCH:{candidate_id}"
                    )
                observed = file_record(
                    self.candidate_dir(candidate_id) / CANDIDATE_MANIFEST_NAME,
                    relative_to=self.candidate_dir(candidate_id),
                )
                expected = summary.get("candidate_manifest_file", {})
                for field in ("bytes", "sha256", "mtime_ns"):
                    if observed[field] != expected.get(field):
                        raise CandidateIntegrityError(
                            f"STOP_GLOBAL_CANDIDATE_MANIFEST_{field.upper()}_MISMATCH:"
                            f"{candidate_id}"
                        )
        return manifest

    def seal_manifest(self) -> dict[str, Any]:
        """Seal the global manifest after all candidates and selections are frozen."""

        manifest = self.validate_manifest(validate_candidates=True)
        if not manifest.get("deletion_forbidden"):
            manifest["status"] = "CANDIDATE_CHECKPOINTS_SEALED"
            manifest["sealed"] = True
            manifest["deletion_forbidden"] = True
            manifest["sealed_utc"] = _utc_now()
            manifest["updated_utc"] = _utc_now()
            manifest["manifest_hash"] = _manifest_hash(manifest)
            atomic_write_json(self.manifest_path, manifest)
        self.manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return manifest


__all__ = [
    "CANDIDATE_MANIFEST_NAME",
    "CANDIDATE_STATE_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "CandidateCacheError",
    "CandidateConflictError",
    "CandidateHashes",
    "CandidateIntegrityError",
    "CandidateLock",
    "CandidateLockError",
    "CandidateWriter",
    "NeuralCandidateCache",
    "ReuseDecision",
    "SafetensorsUnavailableError",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_safetensors",
    "candidate_key",
    "file_record",
    "sha256_file",
    "stable_hash",
]
