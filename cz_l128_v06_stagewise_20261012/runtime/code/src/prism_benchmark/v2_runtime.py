from __future__ import annotations

import ctypes
import gc
import json
import math
import multiprocessing as mp
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable


_GIB = 1024**3


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def cgroup_memory() -> tuple[int | None, int | None]:
    """Return the effective cgroup-v2 memory limit and current use."""
    root = Path("/sys/fs/cgroup")
    return _read_integer(root / "memory.max"), _read_integer(root / "memory.current")


def release_process_memory() -> None:
    """Release Python cycles and return free glibc arenas to the OS when available."""
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def resolve_worker_count(
    requested: int,
    task_count: int,
    *,
    per_worker_gib: float,
    reserve_gib: float | None = None,
) -> int:
    if requested < 1 or task_count < 1 or per_worker_gib <= 0:
        return 0 if task_count < 1 else 1
    reserve = float(os.environ.get("PRISM_MEMORY_RESERVE_GIB", reserve_gib or 4.0))
    limit, current = cgroup_memory()
    memory_cap = requested
    if limit is not None and current is not None:
        available = max(0, limit - current - int(reserve * _GIB))
        memory_cap = max(1, math.floor(available / (per_worker_gib * _GIB)))
    return max(1, min(requested, task_count, memory_cap))


def _call_and_trim(function: Callable[..., Any], arguments: tuple[Any, ...]) -> Any:
    try:
        return function(*arguments)
    finally:
        release_process_memory()


def _install_parent_death_signal(expected_parent_pid: int) -> None:
    """Terminate a Linux worker if its pool parent disappears.

    ProcessPoolExecutor does not otherwise guarantee that already-forked
    workers die with a crashed/killed parent.  PR_SET_PDEATHSIG closes that
    hole, while the post-prctl parent check handles the setup race.
    """
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG = 1
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def run_parallel(
    function: Callable[..., Any],
    jobs: Iterable[tuple[Any, ...]],
    requested_workers: int,
    *,
    per_worker_gib: float,
    label: str,
    fork: bool = True,
) -> list[Any]:
    """Run independent jobs with a cgroup-aware process budget and memory trimming."""
    materialized = list(jobs)
    if not materialized:
        return []
    workers = resolve_worker_count(
        requested_workers,
        len(materialized),
        per_worker_gib=per_worker_gib,
    )
    limit, current = cgroup_memory()
    telemetry = {
        "event": "PRISM_PROCESS_POOL_START",
        "label": label,
        "jobs": len(materialized),
        "requested_workers": requested_workers,
        "resolved_workers": workers,
        "per_worker_gib": per_worker_gib,
        "memory_limit_bytes": limit,
        "memory_current_bytes": current,
        "parent_pid": os.getpid(),
        "parent_death_signal": "SIGTERM" if sys.platform.startswith("linux") else None,
    }
    print(json.dumps(telemetry, sort_keys=True), file=sys.stderr, flush=True)
    context = mp.get_context("fork") if fork and sys.platform.startswith("linux") else None
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_install_parent_death_signal,
        initargs=(os.getpid(),),
    ) as executor:
        futures = [executor.submit(_call_and_trim, function, arguments) for arguments in materialized]
        for future in as_completed(futures):
            results.append(future.result())
    release_process_memory()
    return results


def ordered_fork_map(
    function: Callable[..., Any],
    jobs: Iterable[tuple[Any, ...]],
    workers: int,
    *,
    label: str,
) -> list[Any]:
    """Evaluate indexed, fork-safe jobs concurrently and collect in input order."""
    materialized = list(jobs)
    if workers <= 1 or len(materialized) <= 1:
        return [function(*arguments) for arguments in materialized]
    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"{label} inner process parallelism requires Linux fork")
    resolved = min(int(workers), len(materialized))
    limit, current = cgroup_memory()
    print(
        json.dumps(
            {
                "event": "PRISM_ORDERED_FORK_POOL_START",
                "label": label,
                "jobs": len(materialized),
                "requested_workers": int(workers),
                "resolved_workers": resolved,
                "memory_limit_bytes": limit,
                "memory_current_bytes": current,
                "parent_pid": os.getpid(),
                "parent_death_signal": "SIGTERM",
                "collection_order": "REGISTRATION_ORDER",
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=resolved,
        mp_context=context,
        initializer=_install_parent_death_signal,
        initargs=(os.getpid(),),
    ) as executor:
        futures = [
            executor.submit(_call_and_trim, function, arguments)
            for arguments in materialized
        ]
        results = [future.result() for future in futures]
    release_process_memory()
    return results
