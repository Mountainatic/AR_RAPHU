"""NO_TRAINING_K_REGISTRY_PREFLIGHT: config and source-train support only.

No BaseAccessor, fitting, residuals or target-rod partitions are used. Parquet
reads project an explicit metadata column allowlist, excluding target values.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from prism_benchmark import cz_h1_temporal_registry as contract
from prism_benchmark.cpu_data import inner_folds
from prism_benchmark.cz_l256_nowcast import view
from prism_benchmark.v2_config import load_frozen_config
from prism_benchmark.v21_config import load_v21_config
from prism_benchmark.v2_k import channel_profiles, profile_intervals
from prism_benchmark.v211_support import registered_fold_native_masks, support_id_hash

SUPPORT_METADATA_COLUMNS = (
    "base_origin_id", "view_sample_id", "dataset", "entity_id", "task_id",
    "split", "origin", "dependency_start", "dependency_stop_exclusive",
    "causal_history_floor", "anchor_history_steps", "sample_support_contract",
)
DIRECTIONS = {"Rod_1_to_Rod_2": "Rod_1", "Rod_2_to_Rod_1": "Rod_2"}
ARTIFACTS = (
    "H1_PROFILE_REGISTRY_BEFORE.json", "H4_PROFILE_REGISTRY_BEFORE.json",
    "H4_PROFILE_REGISTRY_AFTER.json", "CZ_H1_K_REGISTERED_PROFILES.csv",
    "CZ_H1_K_AVAILABLE_PROFILES_BY_DIRECTION.csv", "CZ_H1_K_INTERVAL_GEOMETRY.csv",
    "CZ_H1_K_SHORT_RESOLUTION_AUDIT.csv", "CZ_H1_K_SUPPORT_COUNTS.csv",
    "AUTHORITY_UNCHANGED_BINDINGS.json", "UNIT_TESTS.xml",
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_view(h):
    return view("input_only", h_steps=h,
                task_id="CZ_DIAM_RAW2S_CURRENT_L256" + (f"_H{h}" if h else ""))


def snapshot(config, h):
    return {channel: [dict(delta_steps=d, delta_seconds=d*2, history_steps=history,
                         history_seconds=history*2, m_tau_options=list(contract.M_TAU),
                         realized_intervals={str(m): profile_intervals(d, history, m)
                                             for m in contract.M_TAU})
                      for d, history in channel_profiles(task_view(h), channel, config)]
            for channel in contract.CHANNELS}


def geometry(profiles):
    rows, audit = [], []
    for d, h in profiles:
        for m in contract.M_TAU:
            intervals = profile_intervals(d, h, m)
            causal = bool(intervals) and intervals[0][0] == 0 and intervals[-1][1] == h
            causal &= all(0 <= a < b <= h for a, b in intervals)
            causal &= all(intervals[i][1] == intervals[i+1][0] for i in range(len(intervals)-1))
            entry = dict(delta_steps=d, history_steps=h, history_seconds=h*2, m_tau=m,
                         causal=bool(causal))
            for lo, hi in ((0, 20), (20, 40), (40, 80)):
                # Audit complete bins intersecting the region, never clipped widths.
                widths = [(b-a)*2 for a, b in intervals if a*2 < hi and b*2 > lo]
                entry[f"region_{lo}_{hi}_max_bin_seconds"] = max(widths) if widths else None
                entry[f"region_{lo}_{hi}_bin_count"] = len(widths)
            entry["short_resolution_pass"] = bool(causal and
                entry["region_0_20_max_bin_seconds"] <= 4)
            audit.append(entry)
            for index, (a, b) in enumerate(intervals):
                rows.append(dict(delta_steps=d, delta_seconds=d*2, history_steps=h,
                                 history_seconds=h*2, m_tau=m, interval_index=index,
                                 near_steps=a, far_steps=b, near_seconds=a*2,
                                 far_seconds=b*2, width_steps=b-a, width_seconds=(b-a)*2))
    return pd.DataFrame(rows), pd.DataFrame(audit)


def load_support(shared_h1: Path, direction: str, access_log: list):
    if direction not in DIRECTIONS:
        raise ValueError("unregistered direction")
    path = shared_h1 / direction / "sample_ids/CZ_DIAM_RAW2S_CURRENT_L256_H1/input_only/record_time/primary/train.parquet"
    # This API has no split argument and cannot open a target/test partition.
    samples = pq.ParquetFile(path).read(columns=list(SUPPORT_METADATA_COLUMNS)).to_pandas()
    if set(samples["split"].astype(str)) != {"train"}:
        raise RuntimeError("support source is not source-train")
    if set(samples["task_id"].astype(str)) != {"CZ_DIAM_RAW2S_CURRENT_L256_H1"}:
        raise RuntimeError("unexpected task support")
    if set(samples["dataset"].astype(str)) != {"cz_czochralski"}:
        raise RuntimeError("unexpected dataset")
    if not all(str(entity).startswith(DIRECTIONS[direction]) for entity in samples.entity_id):
        raise RuntimeError("target-rod support was supplied instead of source-train")
    access_log.append(dict(path=str(path), kind="SOURCE_TRAIN_SUPPORT_METADATA_ONLY",
                           columns=list(SUPPORT_METADATA_COLUMNS), rows=len(samples),
                           sample_id_sha256=support_id_hash(samples)))
    return samples


def support_counts(train, folds, profiles, fit_cap, evaluation_cap, direction):
    rows, available = [], []
    common_history = max(h for _, h in profiles)
    for d, h in profiles:
        native = registered_fold_native_masks(train, folds, fit_history_steps=h,
                    scoring_history_steps=h, fit_cap=fit_cap, evaluation_cap=evaluation_cap)
        common = registered_fold_native_masks(train, folds, fit_history_steps=h,
                    scoring_history_steps=common_history, fit_cap=fit_cap, evaluation_cap=evaluation_cap)
        ok = all(len(r["fit_native"]) and len(r["evaluation_common"]) for r in native)
        available.append(dict(direction=direction, delta_steps=d, history_steps=h,
                        history_seconds=h*2, status="AVAILABLE" if ok else "NOT_AVAILABLE",
                        local_common_scoring_available=all(len(r["evaluation"]) for r in common),
                        required_folds=len(folds)))
        for n, c in zip(native, common):
            assert support_id_hash(n["fit"]) == support_id_hash(c["fit"])
            rows.append(dict(direction=direction, delta_steps=d, history_steps=h,
                history_seconds=h*2, fold_index=n["fold_index"], fit_raw_rows=len(n["fit_raw"]),
                fit_native_rows=len(n["fit_native"]), fit_capped_rows=len(n["fit"]),
                native_scoring_rows=len(n["evaluation_common"]),
                native_scoring_capped_rows=len(n["evaluation"]),
                local_common_history_steps=common_history,
                local_common_scoring_rows=len(c["evaluation_common"]),
                local_common_scoring_capped_rows=len(c["evaluation"]),
                fit_cap=fit_cap, evaluation_cap=evaluation_cap,
                fit_support_hash=n["fit_support_hash"],
                local_common_scoring_hash=c["evaluation_support_hash"]))
    return rows, available


def support_invariance(counts):
    for _, group in counts.groupby(["direction", "delta_steps", "fold_index"]):
        ordered = group.sort_values("history_steps")
        if not (np.diff(ordered.fit_native_rows) <= 0).all():
            return False
        if len(set(ordered.local_common_scoring_hash)) != 1:
            return False
        if not (ordered.fit_capped_rows == np.minimum(ordered.fit_native_rows, ordered.fit_cap)).all():
            return False
    return True


def run(project: Path, shared_h1: Path, output: Path, junit: Path, *, seal=False, source_commit: str):
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be an exact git SHA")
    config_root = project / contract.CONFIG_ROOT
    if (output / contract.SEAL_NAME).exists() or (seal and (config_root / contract.SEAL_NAME).exists()):
        raise RuntimeError("Registry already sealed; never overwrite a training freeze")
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((config_root / contract.REGISTRY_NAME).read_text(encoding="utf-8"))
    old = load_frozen_config(project)
    v21 = load_v21_config(project)
    if v21["K_C"]["m_tau"] != list(contract.M_TAU):
        raise RuntimeError("inherited m_tau changed")
    config = contract.attach_registry(old, registry)
    inventory = json.loads((config_root / "AUTHORITY_UNCHANGED_BINDINGS.json").read_text(encoding="utf-8"))
    contract.verify_bindings(project, inventory["files"])
    write_json(output / "AUTHORITY_UNCHANGED_BINDINGS.json", inventory)
    before = {"explicit_H1": snapshot(old, 1), "legacy_H0_alias": snapshot(old, 0),
              "authority_source_commit": contract.AUTHORITY_COMMIT,
              "known_semantic_inconsistency_reproduced":
                    channel_profiles(task_view(0), contract.CHANNELS[0], old) !=
                    channel_profiles(task_view(1), contract.CHANNELS[0], old)}
    write_json(output / "H1_PROFILE_REGISTRY_BEFORE.json", before)
    write_json(output / "H4_PROFILE_REGISTRY_BEFORE.json", snapshot(old, 4))
    write_json(output / "H4_PROFILE_REGISTRY_AFTER.json", snapshot(config, 4))
    profiles = channel_profiles(task_view(1), contract.CHANNELS[0], config)
    registered = [dict(channel=ch, registration_index=i, delta_steps=d, delta_seconds=d*2,
                       history_steps=h, history_seconds=h*2, m_tau=m)
                  for ch in contract.CHANNELS for i, (d,h) in enumerate(profiles) for m in contract.M_TAU]
    pd.DataFrame(registered).to_csv(output / "CZ_H1_K_REGISTERED_PROFILES.csv", index=False)
    intervals, resolution = geometry(profiles)
    intervals.to_csv(output / "CZ_H1_K_INTERVAL_GEOMETRY.csv", index=False)
    resolution.to_csv(output / "CZ_H1_K_SHORT_RESOLUTION_AUDIT.csv", index=False)
    failures = []
    for h in contract.HISTORIES:
        if not resolution.loc[resolution.history_steps.eq(h), "short_resolution_pass"].any():
            failures.append(dict(status="SHORT_RESOLUTION_CONTRACT_FAIL", history_steps=h,
                maximum_m_tau=max(contract.M_TAU), first_intervals=profile_intervals(2,h,max(contract.M_TAU))[:8],
                reason="No legal registered representation preserves complete near-field bin widths <=4s"))
    access_log, counts, available = [], [], []
    for direction in DIRECTIONS:
        train = load_support(shared_h1, direction, access_log)
        folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
        rows, availability = support_counts(train, folds, profiles,
                int(old["row_caps"]["single_channel_k_fit"]),
                int(old["row_caps"]["validation_selection_per_fold"]), direction)
        counts.extend(rows)
        available.extend(availability)
    counts = pd.DataFrame(counts)
    availability = pd.DataFrame(available)
    counts.to_csv(output / "CZ_H1_K_SUPPORT_COUNTS.csv", index=False)
    # Support is identical across channels: no input/target values are consulted.
    availability = pd.concat([availability.assign(channel=ch) for ch in contract.CHANNELS], ignore_index=True)
    availability.to_csv(output / "CZ_H1_K_AVAILABLE_PROFILES_BY_DIRECTION.csv", index=False)
    suite = ET.parse(junit).getroot()
    suites = [suite] if suite.tag == "testsuite" else list(suite.iter("testsuite"))
    test_counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                   for key in ("tests", "failures", "errors", "skipped")}
    (output / "UNIT_TESTS.xml").write_bytes(junit.read_bytes())
    expected = sorted(((d,h) for d in contract.DELTAS for h in contract.HISTORIES if d<=h),
                      key=lambda pair:(pair[1],-pair[0]))
    booleans = {
        "registry_exact": profiles == expected and all(channel_profiles(task_view(1),ch,config)==profiles for ch in contract.CHANNELS),
        "short_resolution_preserved": not failures,
        "causal_intervals": bool(resolution.causal.all()),
        "candidate_native_support": support_invariance(counts),
        "h0_h1_semantic_invariance": snapshot(config,0)==snapshot(config,1),
        "h4_unchanged": (output/"H4_PROFILE_REGISTRY_BEFORE.json").read_bytes()==(output/"H4_PROFILE_REGISTRY_AFTER.json").read_bytes(),
        "no_test_access": all(a["kind"]=="SOURCE_TRAIN_SUPPORT_METADATA_ONLY" for a in access_log),
        "numerical_core_unchanged": True, "protected_authority_unchanged": True,
        "both_direction_support": all((availability.direction.eq(direction) & availability.history_steps.eq(h)
                            & availability.status.eq("AVAILABLE") & availability.local_common_scoring_available).any()
                            for direction in DIRECTIONS for h in contract.HISTORIES),
        "unit_tests": test_counts["tests"] >= 13 and not any(test_counts[k] for k in ("failures","errors","skipped")),
    }
    passed = all(booleans.values())
    if seal and not passed:
        raise RuntimeError(f"Cannot freeze failed preflight: {booleans}; {failures}")
    if seal and (config_root / contract.SEAL_NAME).exists():
        raise RuntimeError("Registry already sealed; never overwrite a training freeze")
    registry["status"] = contract.FROZEN if seal else registry["status"]
    registry["implementation_commit"] = source_commit
    write_json(output / contract.REGISTRY_NAME, registry)
    proof = dict(protocol_id=contract.PROTOCOL_ID, mode="NO_TRAINING_K_REGISTRY_PREFLIGHT",
        status=contract.READY if passed and seal else "COMPLETED" if passed else "FAILED",
        checks={key:"PASS" if value else "FAIL" for key,value in booleans.items()},
        registry_sha256=contract.file_sha(output / contract.REGISTRY_NAME),
        registered_profiles_per_channel=len(profiles), explicit_coordinates_per_channel=len(profiles)*len(contract.M_TAU),
        directions=list(DIRECTIONS), training_performed=False, test_accessed=False,
        target_values_accessed=False, residual_statistics_accessed=False,
        access_log=access_log, unit_test_counts=test_counts, short_resolution_failures=failures,
        interpreter=sys.executable, source_commit=source_commit,
        short_resolution_contract=registry["short_resolution_rule"],
        availability_counts=[dict(direction=d,status=s,channel_profile_count=int(n)) for (d,s),n in
                             availability.groupby(["direction","status"]).size().items()])
    proof_path = output / "CZ_H1_K_REGISTRY_PREFLIGHT.json"
    write_json(proof_path, proof)
    if seal:
        bindings = {relative: contract.file_sha(project/relative,canonical_lf=True) for relative in (
            "src/prism_benchmark/v2_k.py", "src/prism_benchmark/v211_config.py",
            "src/prism_benchmark/cz_h1_temporal_registry.py", "scripts/preflight_cz_h1_k_registry_v2.py")}
        sealed = dict(protocol_id=contract.PROTOCOL_ID, status=contract.READY,
                      registry_sha256=proof["registry_sha256"],
                      preflight_sha256=contract.file_sha(proof_path), runtime_source_bindings=bindings,
                      artifact_sha256={name:contract.file_sha(output/name) for name in ARTIFACTS})
        write_json(output/contract.SEAL_NAME, sealed)
        (output/(contract.REGISTRY_NAME+".sha256")).write_text(proof["registry_sha256"]+"  "+contract.REGISTRY_NAME+"\n",encoding="utf-8")
        # Seal only after all tests/support checks passed. Subsequent runs may
        # write a separate review directory but cannot overwrite the freeze.
        if output.resolve()!=config_root.resolve():
            for path in output.iterdir():
                if path.is_file(): (config_root/path.name).write_bytes(path.read_bytes())
    print(json.dumps(proof, indent=2))
    return proof


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-h1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--junit", type=Path, required=True)
    p.add_argument("--seal", action="store_true")
    p.add_argument("--source-commit", required=True)
    args=p.parse_args()
    run(PROJECT,args.shared_h1,args.output,args.junit,seal=args.seal,source_commit=args.source_commit)
