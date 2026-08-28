"""Sequential low-memory launcher for the TEP CPU L256 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts" / "run_tep_cpu_nowcast_history_extension_20260828.py"
CONFIG = PROJECT / "configs" / "tep_cpu_nowcast_history_extension_20260828.json"
EXPECTED_PROTOCOL = "TEP_CPU_NOWCAST_HISTORY_L256_V1"
EXPECTED_HISTORIES = (128, 256)
EXPECTED_COMMON_HISTORY = 256
EXPECTED_PILOT_STAGE = "TEP_CPU_HISTORY_EXTENSION_PILOT"
EXPECTED_PILOT_JOBS = 4
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_READ:{path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_NOT_OBJECT:{path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_FILE_UNREADABLE:{path}") from error
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT.parent), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_GIT_UNAVAILABLE") from error
    return result.stdout.strip()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value.lower()) is None:
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_INVALID_SHA:{label}")
    return value.lower()


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_ACCESS:{label}")


def _relative_artifact_path(run_root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PATH:{label}")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PATH:{label}")
    candidate = (run_root / relative).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError as error:
        raise RuntimeError(
            f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PATH:{label}"
        ) from error
    return candidate


def _reject_test_ood_relative_path(run_root: Path, path: Path, label: str) -> None:
    """Keep pilot evidence from naming lockbox/OOD artifacts by accident."""

    relative_parts = {
        part.casefold()
        for part in path.relative_to(run_root).parts
    }
    forbidden = {
        "test",
        "testing",
        "ood",
        "discarded",
        "unseen",
        "lockbox",
    }
    if relative_parts.intersection(forbidden):
        raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_FORBIDDEN_PATH:{label}")


def _verify_source_registry_inventory(
    scope: dict[str, Any], evidence: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    """Re-hash the six read-only C1 registry files used by the pilot.

    The TEP extension has no test or OOD inputs in scope.  The registry
    inventory therefore serves as the data manifest: a changed source file,
    size, or inventory ordering invalidates the pilot seal.
    """

    raw_inventory = scope.get("source_registry_inventory")
    if not isinstance(raw_inventory, list) or len(raw_inventory) != 6:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_MANIFEST")
    if evidence.get("source_registry_inventory") != raw_inventory:
        raise RuntimeError(
            "STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_MANIFEST_MISMATCH"
        )
    expected_names = {
        "DATASET_HASHES.json",
        "TASK_REGISTRY.json",
        "SPLIT_REGISTRY.json",
        "SAMPLE_ID_REGISTRY.json",
        "PROTOCOL.json",
        "LOCKBOX.json",
    }
    names = {
        item.get("name")
        for item in raw_inventory
        if isinstance(item, dict)
    }
    if names != expected_names:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_REGISTRY")
    source_value = scope.get("source_shared")
    if not isinstance(source_value, str) or not source_value:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_ROOT")
    source_shared = Path(source_value).resolve()
    if not source_shared.is_dir():
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_ROOT")
    for item in raw_inventory:
        if not isinstance(item, dict):
            raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_REGISTRY")
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_REGISTRY")
        path = source_shared / name
        if not path.is_file():
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_FILE_MISSING:{name}"
            )
        try:
            expected_bytes = int(item["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_REGISTRY"
            ) from error
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_SIZE:{name}"
            )
        if _sha256_file(path) != _require_sha(item.get("sha256"), f"data:{name}"):
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_SHA:{name}"
            )
    data_manifest_sha256 = _stable_hash(raw_inventory)
    if scope.get("data_manifest_sha256") != data_manifest_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_DATA_MANIFEST_DRIFT")
    if evidence.get("data_manifest_sha256") != data_manifest_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DATA_MANIFEST_SHA")
    protocol_path = source_shared / "PROTOCOL.json"
    protocol_sha256 = _sha256_file(protocol_path)
    if scope.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_PROTOCOL_MANIFEST_DRIFT")
    if evidence.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PROTOCOL_SHA")
    return str(source_shared), protocol_sha256, raw_inventory


def _verify_support_artifacts(run_root: Path, support: dict[str, Any]) -> None:
    """Verify every derived sample and base-data file named by support."""

    records = support.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_RECORDS")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_RECORD:{index}"
            )
        path = _relative_artifact_path(
            run_root, record.get("path"), f"support:{index}"
        )
        _reject_test_ood_relative_path(run_root, path, f"support:{index}")
        expected_sha256 = _require_sha(
            record.get("sha256"), f"support:{index}"
        )
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_FILE:{index}"
            )
    base_inventory = support.get("base_data_inventory", [])
    if not isinstance(base_inventory, list):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_BASE_DATA_RECORDS")
    for index, record in enumerate(base_inventory):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_BASE_DATA_RECORD:{index}"
            )
        raw_path = record.get("source_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_BASE_DATA_PATH:{index}"
            )
        path = Path(raw_path).resolve()
        expected_sha256 = _require_sha(
            record.get("sha256"), f"base_data:{index}"
        )
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_BASE_DATA_FILE:{index}"
            )


def verify_pilot_evidence(
    evidence_path: Path,
    run_root: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a sealed local PILOT.json before skipping pilot execution.

    Validation is intentionally independent of the pilot process.  Every
    provenance hash and prediction artifact is recomputed from the current
    checkout/run root, and no formal stage is allowed to start on a mismatch.
    """

    run_root = run_root.resolve()
    evidence_path = evidence_path.resolve()
    if not evidence_path.is_file():
        raise RuntimeError(
            f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_MISSING:{evidence_path}"
        )
    evidence = _read_json(evidence_path)
    if config is None:
        config = _read_config()
    if evidence.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_STATUS")
    if evidence.get("stage") != EXPECTED_PILOT_STAGE:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_STAGE")
    if evidence.get("pilot_evidence_version") != 1:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_VERSION")
    if evidence.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PROTOCOL")
    _require_false(evidence.get("test_accessed"), "pilot.test_accessed")
    _require_false(evidence.get("ood_accessed"), "pilot.ood_accessed")

    scope_path = run_root / "logs" / "SCOPE.json"
    support_path = run_root / "logs" / "L256_DEVELOPMENT_SUPPORT.json"
    if not scope_path.is_file() or not support_path.is_file():
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_MISSING")
    scope = _read_json(scope_path)
    support = _read_json(support_path)
    if scope.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SCOPE_STATUS")
    if scope.get("stage") != "TEP_CPU_HISTORY_EXTENSION_SCOPE":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SCOPE_STAGE")
    if support.get("status") != "PASS":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_STATUS")
    _require_false(scope.get("test_accessed"), "scope.test_accessed")
    _require_false(scope.get("ood_accessed"), "scope.ood_accessed")
    _require_false(support.get("test_accessed"), "support.test_accessed")
    _require_false(support.get("ood_accessed"), "support.ood_accessed")

    current_commit = _git("rev-parse", "HEAD")
    if not current_commit or evidence.get("source_commit") != current_commit:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_COMMIT")
    if scope.get("source_commit") != current_commit:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_COMMIT")
    if scope.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_PROTOCOL")

    config_sha256 = _sha256_file(CONFIG)
    if config.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("STOP_TEP_EXTENSION_CONFIG_PROTOCOL")
    if evidence.get("config_sha256") != config_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_CONFIG_SHA")
    if evidence.get("history_override_config_sha256") != config_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_OVERRIDE_SHA")
    if scope.get("config_sha256") != config_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_CONFIG_SHA")

    source_shared, protocol_sha256, inventory = _verify_source_registry_inventory(
        scope, evidence
    )
    del source_shared, protocol_sha256, inventory

    scope_manifest_sha256 = _sha256_file(scope_path)
    if evidence.get("scope_manifest_path") != "logs/SCOPE.json":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SCOPE_PATH")
    if evidence.get("scope_manifest_sha256") != scope_manifest_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SCOPE_SHA")
    _verify_support_artifacts(run_root, support)
    support_manifest_sha256 = _stable_hash(support)
    if scope.get("support_manifest_sha256") != support_manifest_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_SCOPE_SUPPORT_MANIFEST_DRIFT")
    if evidence.get("support_manifest_path") != "logs/L256_DEVELOPMENT_SUPPORT.json":
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_PATH")
    if evidence.get("support_manifest_sha256") != support_manifest_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_SHA")
    support_file_sha256 = _sha256_file(support_path)
    if evidence.get("support_manifest_file_sha256") != support_file_sha256:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_SUPPORT_FILE_SHA")

    expected_histories = list(EXPECTED_HISTORIES)
    if evidence.get("jobs") != EXPECTED_PILOT_JOBS:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_COUNT")
    if evidence.get("history_steps_observed") != expected_histories:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_HISTORY_COVERAGE")
    if evidence.get("common_support_history_steps") != EXPECTED_COMMON_HISTORY:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_COMMON_HISTORY")
    observed_by_job = evidence.get("history_steps_observed_by_job")
    if not isinstance(observed_by_job, dict):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_COVERAGE")
    if any(value != expected_histories for value in observed_by_job.values()):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_COVERAGE")
    if evidence.get("missing_registered_histories") != []:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_HISTORY_MISSING")
    if evidence.get("missing_registered_histories_by_job") != {}:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_HISTORY_MISSING")

    audits = evidence.get("job_audits")
    if not isinstance(audits, dict) or len(audits) != EXPECTED_PILOT_JOBS:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_AUDITS")
    if set(observed_by_job) != set(audits):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_COVERAGE")
    for job, audit in audits.items():
        if not isinstance(job, str) or not isinstance(audit, dict):
            raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_AUDITS")
        if audit.get("status") != "PASS" or audit.get("passed") is not True:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_STATUS:{job}")
        if audit.get("history_steps") != expected_histories:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_HISTORY:{job}")
        if audit.get("finite_exact_four_fold_grid") is not True:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_FOLDS:{job}")
        if audit.get("grid_size_valid") is not True:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_GRID:{job}")
        if audit.get("common_scoring_valid") is not True:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_SCORING:{job}"
            )
        if audit.get("override_config_sha256") != config_sha256:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_CONFIG:{job}"
            )
        if audit.get("override_protocol_id") != EXPECTED_PROTOCOL:
            raise RuntimeError(
                f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_PROTOCOL:{job}"
            )
        if audit.get("prediction_artifact_verified") is not True:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_JOB_PREDICTION:{job}")

    artifacts = evidence.get("prediction_artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != EXPECTED_PILOT_JOBS:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTIONS")
    if set(artifacts) != set(audits):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_JOBS")
    seen_paths: set[Path] = set()
    for job, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION:{job}")
        path = _relative_artifact_path(run_root, artifact.get("path"), f"prediction:{job}")
        _reject_test_ood_relative_path(run_root, path, f"prediction:{job}")
        if path in seen_paths:
            raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_DUPLICATE_PREDICTION")
        seen_paths.add(path)
        expected_sha256 = _require_sha(artifact.get("sha256"), f"prediction:{job}")
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_SHA:{job}")
        audit = audits[job]
        if audit.get("prediction_path") != path.relative_to(run_root).as_posix():
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_PATH:{job}")
        if audit.get("prediction_sha256") != expected_sha256:
            raise RuntimeError(f"STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_HASH:{job}")
    if evidence.get("prediction_artifact_count") != EXPECTED_PILOT_JOBS:
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_COUNT")
    if evidence.get("prediction_artifacts_sha256") != _stable_hash(artifacts):
        raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_MANIFEST")
    prediction_sha256 = evidence.get("prediction_sha256")
    if prediction_sha256 is not None:
        if not isinstance(prediction_sha256, dict) or prediction_sha256 != {
            job: artifact["sha256"] for job, artifact in artifacts.items()
        }:
            raise RuntimeError("STOP_TEP_EXTENSION_PILOT_EVIDENCE_PREDICTION_SHA_MAP")

    return {
        "status": "PASS",
        "stage": "TEP_CPU_HISTORY_EXTENSION_PILOT_EVIDENCE_GUARD",
        "source_commit": current_commit,
        "protocol_id": EXPECTED_PROTOCOL,
        "config_sha256": config_sha256,
        "prediction_artifact_count": EXPECTED_PILOT_JOBS,
        "evidence_path": str(evidence_path),
    }


# Keep a private spelling available to tests and downstream launch wrappers.
_verify_pilot_evidence = verify_pilot_evidence
validate_pilot_evidence = verify_pilot_evidence
_validate_pilot_evidence = verify_pilot_evidence


def _rss_bytes(pid: int) -> int:
    total = 0
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        status = Path(f"/proc/{current}/status")
        try:
            lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
            elif line.startswith("PPid:"):
                pass
        for child_status in Path("/proc").glob("[0-9]*/status"):
            try:
                child_lines = child_status.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                ppid = next(
                    int(line.split()[1])
                    for line in child_lines
                    if line.startswith("PPid:")
                )
                if ppid == current:
                    pending.append(int(child_status.parent.name))
            except (OSError, StopIteration, ValueError):
                continue
    return total


def _terminate_own_stage(process: subprocess.Popen, reason: str) -> None:
    marker = f"STOPPING_OWN_STAGE:{reason}\n"
    sys.stderr.write(marker)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    raise RuntimeError(reason)


def _cgroup_memory_bytes() -> int:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
    raise RuntimeError("STOP_CGROUP_MEMORY_ACCOUNTING_UNAVAILABLE")


def _run_stage(stage: str, shared: Path, run_root: Path, config: dict) -> None:
    logs = (
        run_root.parent / f".{run_root.name}_launcher_logs"
        if stage == "scope"
        else run_root / "launcher_logs"
    )
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{stage}.log"
    command = [
        sys.executable,
        str(RUNNER),
        stage,
        "--shared",
        str(shared),
        "--run-root",
        str(run_root),
    ]
    environment = os.environ.copy()
    source_path = str(PROJECT / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not inherited_pythonpath
        else os.pathsep.join((source_path, inherited_pythonpath))
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PRISM_V211_K_INNER_WORKERS",
        "PRISM_V211_C_INNER_WORKERS",
        "PRISM_V211_W_INNER_WORKERS",
        "PRISM_V211_A_INNER_WORKERS",
        "PRISM_V211_JOINT_STABILITY_J_INNER_WORKERS",
    ):
        environment[name] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        recommended = float(config["resources"]["recommended_memory_limit_gib"])
        stopline = float(config["resources"]["minimum_runtime_free_gib"])
        while process.poll() is None:
            rss_gib = _rss_bytes(process.pid) / (1024**3)
            try:
                cgroup_gib = _cgroup_memory_bytes() / (1024**3)
            except RuntimeError as error:
                _terminate_own_stage(process, str(error))
            free_gib = shutil.disk_usage(run_root.parent).free / (1024**3)
            if cgroup_gib >= recommended:
                _terminate_own_stage(
                    process,
                    f"STOP_CGROUP_MEMORY_LIMIT:{cgroup_gib:.3f}>={recommended:.3f}GiB"
                )
            if free_gib < stopline:
                _terminate_own_stage(
                    process, f"STOP_LOW_STORAGE:{free_gib:.3f}<{stopline:.3f}GiB"
                )
            log.write(
                f"WATCHDOG stage={stage} cgroup_gib={cgroup_gib:.3f} "
                f"process_tree_rss_gib={rss_gib:.3f} free_gib={free_gib:.3f}\n"
            )
            log.flush()
            time.sleep(2)
        if process.returncode != 0:
            raise RuntimeError(
                f"TEP CPU stage failed: {stage}; inspect {log_path}"
            )
    if stage == "scope":
        final_logs = run_root / "launcher_logs"
        final_logs.mkdir(parents=True, exist_ok=True)
        log_path.replace(final_logs / log_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot-only", action="store_true")
    mode.add_argument("--skip-pilot", action="store_true")
    parser.add_argument(
        "--pilot-evidence",
        type=Path,
        help="sealed local PILOT.json required when --skip-pilot is used",
    )
    args = parser.parse_args()
    if args.skip_pilot and args.pilot_evidence is None:
        parser.error("--skip-pilot requires --pilot-evidence PATH")
    config = _read_config()
    shared = args.shared.resolve()
    run_root = args.run_root.resolve()
    stages = ["scope"]
    if args.pilot_only:
        stages.append("pilot")
    else:
        if not args.skip_pilot:
            stages.append("pilot")
        stages.extend(["k", "c", "w", "a", "joint", "baselines"])
    for stage in stages:
        _run_stage(stage, shared, run_root, config)
        if args.skip_pilot and stage == "scope":
            verify_pilot_evidence(args.pilot_evidence, run_root, config)
    print(
        json.dumps(
            {
                "status": "DEVELOPMENT_COMPLETE",
                "stages": stages,
                "run_root": str(run_root),
                "pilot_evidence": (
                    None
                    if args.pilot_evidence is None
                    else str(args.pilot_evidence.resolve())
                ),
                "test_accessed": False,
                "ood_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
