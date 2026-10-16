from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .cpu_data import ViewSpec, sha256_file
from .v211_public_all_config import PublicAllPaths
from .v211_public_all_views import public_all_k_jobs
from .v211_support import SUPPORT_CONTRACT


COMPLETED_STATUSES = frozenset(
    {
        "PASS",
        "FAILED_RETAINED",
        "NOT_RUN_IMPLEMENTATION_ABSENT",
        "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    }
)


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"result is not an object: {path}")
    return value


def k_result_path(output: Path, view: ViewSpec, channel: str) -> Path:
    return (
        output
        / "DEVELOPMENT"
        / "K"
        / view.head.head_id
        / view.proxy_policy
        / channel
        / "RESULT.json"
    )


def _check_list_lengths(result: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    lengths = {
        field: len(result[field])
        for field in fields
        if isinstance(result.get(field), list)
    }
    if not lengths or len(set(lengths.values())) != 1:
        raise RuntimeError(f"fold audit lengths disagree: {lengths}")


def audit_k_result(result: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    status = str(result.get("status"))
    if status not in COMPLETED_STATUSES:
        return [f"unexpected status: {status}"]
    if result.get("support_contract") != SUPPORT_CONTRACT:
        errors.append("support_contract mismatch")
    if result.get("test_accessed") is not False:
        errors.append("test_accessed is not false")
    if result.get("ood_accessed") not in (False, None):
        errors.append("ood_accessed is not false")
    if status != "PASS":
        return errors
    for field in (
        "selected_profile_history_steps",
        "selected_native_train_rows",
        "selected_native_validation_rows",
        "local_scoring_rows_by_fold",
        "local_scoring_support_hash_by_fold",
        "native_fit_rows_by_fold",
        "native_fit_support_hash_by_fold",
    ):
        if field not in result:
            errors.append(f"missing {field}")
    for field in (
        "row_cap_applied_after_native_mask",
        "cross_channel_loss_comparable",
        "historical_global_lmax_used",
    ):
        if field not in result:
            errors.append(f"missing {field}")
    if result.get("row_cap_applied_after_native_mask") is not True:
        errors.append("row cap was not recorded after native mask")
    if result.get("cross_channel_loss_comparable") is not False:
        errors.append("cross-channel loss comparability was not disabled")
    if result.get("historical_global_lmax_used") is not False:
        errors.append("historical global lmax was used")
    try:
        _check_list_lengths(
            result,
            (
                "local_scoring_rows_by_fold",
                "local_scoring_support_hash_by_fold",
                "native_fit_rows_by_fold",
                "native_fit_support_hash_by_fold",
                "exact_zero_scoring_support_hash",
                "nonzero_scoring_support_hash",
            ),
        )
        exact = result["exact_zero_scoring_support_hash"]
        nonzero = result["nonzero_scoring_support_hash"]
        if exact != nonzero:
            errors.append("candidate scoring support hashes disagree")
        if any(int(value) <= 0 for value in result["local_scoring_rows_by_fold"]):
            errors.append("empty local scoring fold")
        if int(result["selected_native_train_rows"]) <= 0:
            errors.append("empty selected native train support")
        if int(result["selected_native_validation_rows"]) <= 0:
            errors.append("empty selected native validation support")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        errors.append(f"invalid fold audit: {error}")
    support_audit = result.get("selected_native_support_audit")
    if not isinstance(support_audit, Mapping):
        errors.append("missing selected_native_support_audit")
    else:
        for split in ("train", "validation"):
            item = support_audit.get(split)
            if not isinstance(item, Mapping):
                errors.append(f"missing selected native {split} audit")
            elif item.get("support_contract") != SUPPORT_CONTRACT:
                errors.append(f"selected native {split} support mismatch")
    return errors


def audit_k_stage(shared: Path, output: Path) -> dict[str, Any]:
    jobs = public_all_k_jobs(shared)
    details: list[dict[str, Any]] = []
    for view, channel in jobs:
        path = k_result_path(output, view, channel)
        item = {
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "channel": channel,
            "path": str(path),
        }
        if not path.is_file():
            item.update({"status": "MISSING", "errors": ["RESULT.json absent"]})
        else:
            try:
                result = _read_result(path)
                item.update(
                    {
                        "status": result.get("status"),
                        "errors": audit_k_result(result),
                    }
                )
            except Exception as error:
                item.update(
                    {
                        "status": "INVALID",
                        "errors": [f"cannot read RESULT.json: {error}"],
                    }
                )
        details.append(item)
    errors = [
        {key: value for key, value in item.items() if key != "path"}
        for item in details
        if item["errors"]
    ]
    statuses = [str(item["status"]) for item in details]
    if errors:
        status = "FAILED"
    elif all(value == "PASS" for value in statuses):
        status = "PASS"
    else:
        status = "COMPLETED_WITH_RETAINED_FAILURES"
    return {
        "status": status,
        "stage": "G4_PUBLIC_ALL_K_NATIVE_SUPPORT_AUDIT",
        "jobs_expected": len(jobs),
        "jobs_observed": len(details),
        "pass": statuses.count("PASS"),
        "retained_failures": sum(value != "PASS" for value in statuses),
        "errors": errors,
        "details": details,
        "support_contract": SUPPORT_CONTRACT,
        "test_accessed": False,
        "ood_accessed": False,
    }


def shared_development_metadata_sha256(shared: Path) -> str:
    """Hash only C1 metadata and non-test split registries before freeze."""
    candidates = [
        path
        for path in shared.rglob("*")
        if path.is_file()
        and "test" not in {part.lower() for part in path.parts}
        and "ood" not in {part.lower() for part in path.parts}
    ]
    digest = hashlib.sha256()
    for path in sorted(candidates):
        relative = path.relative_to(shared).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def write_k_audit(paths: PublicAllPaths) -> dict[str, Any]:
    audit = audit_k_stage(paths.shared, paths.output)
    audit["shared_development_metadata_sha256"] = shared_development_metadata_sha256(
        paths.shared
    )
    destination = paths.freeze / "K_NATIVE_SUPPORT_AUDIT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit
