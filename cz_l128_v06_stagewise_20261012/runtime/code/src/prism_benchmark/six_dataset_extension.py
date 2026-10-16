from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .cpu_data import HeadSpec, ViewSpec, main_views, sha256_file
from .cz_extension import H_STEPS, MODEL_PERIOD_SEC, TASK_ID, W0_STEPS, W_STEPS
from .neural3 import MODEL_FAMILIES
from .stage0 import write_json
from .v211_public_all_baselines import (
    SupportRequirement,
    apply_common_requirements,
)
from .v211_public_all_closure import (
    LEGAL_JOINT_DEVELOPMENT_STATUSES,
    METADATA_COLUMNS,
    view_support_requirements,
)
from .v211_public_all_config import PublicAllPaths
from .v211_support import SUPPORT_COLUMNS, SUPPORT_CONTRACT, support_id_hash


EXTENSION_EVIDENCE_CLASS = (
    "POST_HOC_NEURAL_BASELINE_AND_CZ_DATASET_EXTENSION"
    "_WITH_FROZEN_DEVELOPMENT_ONLY_SELECTION"
)
EXTENSION_BRANCH = "prism-v2-1-1-cz-neural3-six-datasets-20260817"
DIRECTIONS = ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1")
PUBLIC5_DEFAULT_ROOTS = (
    Path(
        "/root/autodl-tmp/"
        "PRISM_V211_NATIVE_PUBLIC_ALL_20260815_MATERIALIZATION_REPAIR_R5"
    ),
    Path("/root/autodl-tmp/PRISM_V211_NATIVE_PUBLIC_ALL_20260815"),
)


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=project,
        text=True,
    ).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def _hash_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            continue
        digest.update(str(path).encode("utf-8"))
        digest.update(b"|")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_tree(root: Path, *, include_suffixes: set[str] | None = None) -> str:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            include_suffixes is None
            or path.suffix.lower() in include_suffixes
        )
    ]
    return _hash_paths(paths)


def public5_root() -> Path:
    for root in PUBLIC5_DEFAULT_ROOTS:
        if (root / "final" / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json").is_file():
            return root
    raise FileNotFoundError("completed public-five final root")


def cz_head() -> HeadSpec:
    return HeadSpec(
        head_id=TASK_ID,
        task_id=TASK_ID,
        dataset="cz_czochralski",
        target="crystal_diameter",
        cadence_seconds=MODEL_PERIOD_SEC,
        h_steps=H_STEPS,
        w_steps=W_STEPS,
        w0_steps=W0_STEPS,
        primary=True,
    )


def cz_view(
    run_root: Path,
    direction: str,
    information_set: str,
) -> ViewSpec:
    del run_root, direction
    return ViewSpec(
        cz_head(),
        information_set,
        "record_time",
        "primary",
    )


def cz_paths(
    project: Path,
    run_root: Path,
    direction: str,
) -> PublicAllPaths:
    return PublicAllPaths(
        project=project.resolve(),
        shared=(run_root / "shared" / direction).resolve(),
        run_root=(run_root / "directions" / direction).resolve(),
    )


def public5_views(root: Path) -> list[ViewSpec]:
    shared = root / "shared"
    return [
        *main_views(shared, "input_only"),
        *main_views(shared, "dynamic"),
    ]


def extension_views(
    run_root: Path,
    public_root: Path,
) -> list[tuple[str, Path, ViewSpec, str | None]]:
    jobs: list[tuple[str, Path, ViewSpec, str | None]] = []
    for view in public5_views(public_root):
        jobs.append(("public5", public_root / "shared", view, None))
    for direction in DIRECTIONS:
        shared = run_root / "shared" / direction
        for information_set in ("input_only", "dynamic"):
            jobs.append(
                (
                    "cz",
                    shared,
                    cz_view(run_root, direction, information_set),
                    direction,
                )
            )
    return jobs


def _sample_metadata(
    shared: Path,
    view: ViewSpec,
    split: str,
) -> pd.DataFrame:
    path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = list(
        dict.fromkeys(
            [
                "base_origin_id",
                "view_sample_id",
                *METADATA_COLUMNS,
                *SUPPORT_COLUMNS,
            ]
        )
    )
    return pd.read_parquet(path, columns=columns)


def _support_requirements_from_neural(
    run_root: Path,
    view: ViewSpec,
    scope: str,
    direction: str | None,
) -> tuple[SupportRequirement, ...]:
    if scope == "public5":
        destination = run_root / "results" / "NEURAL3" / "PUBLIC5"
    else:
        if direction is None:
            raise ValueError("CZ neural requirements need a direction")
        destination = run_root / "results" / "NEURAL3" / "CZ" / direction
    requirements: list[SupportRequirement] = []
    for model_name in MODEL_FAMILIES:
        selection_path = (
            destination / model_name / view.relative_root / "SELECTION.json"
        )
        selection = _read_json(selection_path)
        if selection.get("status") != "PASS":
            raise RuntimeError(f"neural selection is not PASS: {selection_path}")
        profile = selection["selected_profile"]
        history = int(profile["history_steps"])
        if view.information_set == "input_only":
            requirements.append(
                SupportRequirement(input_history_steps=history)
            )
        else:
            requirements.append(
                SupportRequirement(
                    input_history_steps=history,
                    target_delta_steps=1,
                    target_history_steps=history,
                )
            )
    return tuple(sorted(set(requirements)))


def _old_public_requirements(
    public_root: Path,
    view: ViewSpec,
) -> tuple[SupportRequirement, ...]:
    path = public_root / "freeze" / "TASK_LEADERBOARD_COMMON_SUPPORT.json"
    payload = _read_json(path)
    for item in payload.get("views", ()):
        if (
            item.get("target_head") == view.head.head_id
            and item.get("information_set") == view.information_set
            and item.get("availability_scenario")
            == view.availability_scenario
            and item.get("proxy_policy") == view.proxy_policy
        ):
            return tuple(
                SupportRequirement(**value)
                for value in item.get("requirements", ())
            ) or (SupportRequirement(),)
    raise KeyError(view.relative_root)


def _support_record(
    shared: Path,
    view: ViewSpec,
    requirements: Sequence[SupportRequirement],
    *,
    scope: str,
    direction: str | None,
) -> dict[str, Any]:
    split_details: dict[str, Any] = {}
    for split in ("train", "validation", "test", "ood"):
        path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
        if not path.is_file():
            continue
        frame = _sample_metadata(shared, view, split)
        # An explicitly materialized empty split has no row value from which to
        # observe the support contract.  Preserve it as an empty support set;
        # non-empty splits still pass the strict native-contract validation in
        # apply_common_requirements.
        common = (
            frame.copy()
            if frame.empty
            else apply_common_requirements(frame, requirements)
        )
        split_details[split] = {
            "rows": int(len(common)),
            "source_rows": int(len(frame)),
            "support_hash": support_id_hash(common),
            "support_contract": SUPPORT_CONTRACT,
        }
    return {
        "scope": scope,
        "direction": direction,
        "target_head": view.head.head_id,
        "dataset": view.head.dataset,
        "task_id": view.head.task_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "requirements": [item.to_json() for item in requirements],
        "splits": split_details,
    }


def build_extension_common_support(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root
    records: list[dict[str, Any]] = []
    for scope, shared, view, direction in extension_views(
        run_root, public_root
    ):
        if scope == "public5":
            base_requirements = _old_public_requirements(public_root, view)
        else:
            paths = cz_paths(project, run_root, str(direction))
            base_requirements = view_support_requirements(paths, view)
        neural_requirements = _support_requirements_from_neural(
            run_root,
            view,
            scope,
            direction,
        )
        requirements = tuple(
            sorted(set((*base_requirements, *neural_requirements)))
        )
        records.append(
            _support_record(
                shared,
                view,
                requirements,
                scope=scope,
                direction=direction,
            )
        )

    payload = {
        "status": "PASS",
        "stage": "F1_NEURAL3_EXTENSION_COMMON_SUPPORT",
        "support_contract": SUPPORT_CONTRACT,
        "records": records,
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    freeze = run_root / "freeze"
    freeze.mkdir(parents=True, exist_ok=True)
    _write_json(freeze / "NEURAL3_EXTENSION_COMMON_SUPPORT.json", payload)
    for direction in DIRECTIONS:
        direction_records = [
            item
            for item in records
            if item.get("direction") == direction
        ]
        direction_freeze = run_root / "directions" / direction / "freeze"
        direction_freeze.mkdir(parents=True, exist_ok=True)
        _write_json(
            direction_freeze / "TASK_LEADERBOARD_COMMON_SUPPORT.json",
            {
                "status": "PASS",
                "stage": "F1_CZ_DIRECTION_COMMON_SUPPORT",
                "support_contract": SUPPORT_CONTRACT,
                "views": direction_records,
                "test_y_read": False,
                "ood_y_read": False,
                "test_accessed": False,
                "ood_accessed": False,
            },
        )
    return payload


def _neural_selection_paths(
    run_root: Path,
    public_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    for scope, _shared, view, direction in extension_views(
        run_root, public_root
    ):
        if scope == "public5":
            destination = run_root / "results" / "NEURAL3" / "PUBLIC5"
        else:
            destination = (
                run_root / "results" / "NEURAL3" / "CZ" / str(direction)
            )
        paths.extend(
            destination / model / view.relative_root / "SELECTION.json"
            for model in MODEL_FAMILIES
        )
    return paths


def _cz_result_statuses(
    run_root: Path,
    direction: str,
) -> dict[str, Any]:
    log = _read_json(run_root / "logs" / f"PRISM_{direction}.json")
    for stage in ("C", "W", "A", "JOINT"):
        if stage not in log:
            raise RuntimeError(f"CZ PRISM stage missing: {direction}/{stage}")
    if log["C"].get("status") != "PASS":
        raise RuntimeError(f"CZ C did not PASS: {direction}")
    if log["W"].get("status") != "PASS":
        raise RuntimeError(f"CZ W did not PASS: {direction}")
    if log["A"].get("status") != "PASS":
        raise RuntimeError(f"CZ A did not PASS: {direction}")
    if log["JOINT"].get("status") not in LEGAL_JOINT_DEVELOPMENT_STATUSES:
        raise RuntimeError(f"illegal CZ Joint status: {direction}")
    return log


def _cz_registry_hash() -> str:
    registry = (
        Path(__file__).resolve().parents[2]
        / "PRISM_INDUSTRIAL_BENCHMARK_V1"
        / "dataset_registry"
        / "cz_czochralski"
    )
    if not registry.is_dir():
        registry = (
            Path(__file__).resolve().parents[1]
            / "dataset_registry"
            / "cz_czochralski"
        )
    return _hash_tree(registry)


def _cz_raw_hash() -> tuple[str, int, str]:
    candidates = sorted(
        Path(
            "/root/autodl-tmp/PRISM_DATASETS_V1/raw_sources/"
            "cz_czochralski"
        ).glob("*.xlsx")
    )
    if not candidates:
        raise FileNotFoundError("CZ raw workbook")
    path = candidates[0]
    return sha256_file(path), int(path.stat().st_size), str(path)


def freeze_extension(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
    generating_commit: str | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root
    public_evidence = (
        public_root / "final" / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json"
    )
    public_summary = _read_json(public_evidence)
    if public_summary.get("datasets") != 5:
        raise RuntimeError("STOP_PUBLIC5_NOT_COMPLETED")
    if public_summary.get("support_contract") != SUPPORT_CONTRACT:
        raise RuntimeError("public-five support contract mismatch")
    if public_summary.get("post_test_reselection") is not False:
        raise RuntimeError("public-five post-test reselection flag is not false")
    for direction in DIRECTIONS:
        _cz_result_statuses(run_root, direction)
        baseline_summary = _read_json(
            run_root
            / "directions"
            / direction
            / "results"
            / "BASELINE_DEVELOPMENT"
            / "SUMMARY.json"
        )
        if baseline_summary.get("status") == "FAILED":
            raise RuntimeError(f"CZ baseline development failed: {direction}")
    public_root = public_root.resolve()
    selection_paths = _neural_selection_paths(run_root, public_root)
    selections = []
    for path in selection_paths:
        value = _read_json(path)
        if value.get("status") != "PASS":
            raise RuntimeError(f"neural selection did not PASS: {path}")
        if value.get("test_accessed") is not False:
            raise RuntimeError(f"neural selection accessed test: {path}")
        selections.append(
            {
                "path": str(path),
                "model": value.get("model"),
                "view": value.get("view"),
                "selected_profile": value.get("selected_profile"),
                "support": value.get("native_support"),
            }
        )
    build_extension_common_support(
        run_root,
        project,
        public_root=public_root,
    )
    raw_sha, raw_bytes, raw_path = _cz_raw_hash()
    commit = generating_commit or _git(project, "rev-parse", "HEAD")
    shared_metadata_paths = list(
        (run_root / "shared").rglob("*.json")
    ) + list((run_root / "shared").rglob("*.csv"))
    freeze = {
        "status": "FROZEN",
        "evidence_class": EXTENSION_EVIDENCE_CLASS,
        "extension_branch": _git(project, "branch", "--show-current"),
        "extension_generating_commit": commit,
        "public5_source_branch": public_summary.get("source_branch"),
        "public5_source_commit": public_summary.get("source_commit"),
        "public5_original_freeze_sha": public_summary.get(
            "development_freeze_sha"
        ),
        "public5_final_evidence_sha256": sha256_file(public_evidence),
        "support_contract": SUPPORT_CONTRACT,
        "cz_raw_path": raw_path,
        "cz_raw_sha256": raw_sha,
        "cz_raw_bytes": raw_bytes,
        "cz_registry_sha256": _cz_registry_hash(),
        "cz_shared_metadata_sha256": _hash_paths(shared_metadata_paths),
        "neural_models": list(MODEL_FAMILIES),
        "public5_views": len(public5_views(public_root)),
        "cz_transfer_directions": list(DIRECTIONS),
        "neural_selection_count": len(selections),
        "neural_selections": selections,
        "common_support_sha256": sha256_file(
            run_root / "freeze" / "NEURAL3_EXTENSION_COMMON_SUPPORT.json"
        ),
        "test_access_started": False,
        "test_accessed": False,
        "ood_accessed": False,
        "post_test_reselection": False,
    }
    _write_json(
        run_root / "freeze" / "SIX_DATASET_CZ_NEURAL3_DEVELOPMENT_FREEZE.json",
        freeze,
    )
    return freeze
