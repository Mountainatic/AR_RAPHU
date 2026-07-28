#!/usr/bin/env python3
"""Summarize PB1 repair-v2 development artifacts without opening test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pwh", "whpn", "cascaded_tanks", "silverbox")
HORIZONS = (1, 5, 10, 20)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def recursive_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from recursive_values(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_values(item, key)


def audit_official_test_access(results_root: Path) -> tuple[int, list[str]]:
    maximum = 0
    violations: list[str] = []
    for path in sorted(results_root.rglob("*.json")):
        payload = load_json(path)
        counts = [
            int(value)
            for value in recursive_values(payload, "official_test_access_count")
            if value is not None
        ]
        rows = [
            int(value)
            for value in recursive_values(payload, "official_test_rows_loaded")
            if value is not None
        ]
        observed = max(counts + rows + [0])
        maximum = max(maximum, observed)
        if observed != 0:
            violations.append(str(path.relative_to(results_root)))
    return maximum, violations


def spectral_path(results_root: Path, dataset: str, horizon: int) -> Path:
    return (
        results_root
        / dataset
        / "development"
        / "H3_SHARED_HISTORY"
        / f"SPECTRAL_PILOT_H{horizon}"
        / "full_spectral.json"
    )


def bootstrap_path(results_root: Path, dataset: str, horizon: int) -> Path:
    return spectral_path(results_root, dataset, horizon).with_name("rank_bootstrap.json")


def direct_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "NOT_YET_RUN"}
    payload = load_json(path)
    selected = (payload.get("penalty") or {}).get("selected") or {}
    mse = selected.get("validation_mse_mean")
    free_run = payload.get("free_run_validation") or {}
    return {
        "status": payload.get("status", "FAILED"),
        "validation_mse": mse,
        "validation_rmse": math.sqrt(mse) if isinstance(mse, (int, float)) else None,
        "effective_df": selected.get("effective_df"),
        "relative_kkt_residual": selected.get("relative_kkt_residual"),
        "free_run_status": free_run.get("status", "NOT_APPLICABLE"),
        "source_commit": payload.get("source_commit"),
        "sha256": sha256(path),
    }


def bootstrap_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "NOT_YET_RUN"}
    payload = load_json(path)
    return {
        "status": payload.get("status", "FAILED"),
        "replicates": payload.get("replicates"),
        "automatic_block_length": payload.get("automatic_block_length"),
        "maximum_relative_kkt_residual": payload.get(
            "maximum_relative_kkt_residual"
        ),
        "rank_frequencies": payload.get("spectral_tail_budget_rank_frequencies"),
        "retunes_penalty": payload.get("retunes_penalty"),
        "retunes_resolution": payload.get("retunes_resolution"),
        "source_commit": payload.get("source_commit"),
        "sha256": sha256(path),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def rank_frequency_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "—"
    pieces = []
    for budget in ("0.1", "0.05", "0.02"):
        counts = value.get(budget)
        if not isinstance(counts, dict):
            pieces.append(f"{budget}:—")
            continue
        ordered = ",".join(
            f"r{rank}={count}" for rank, count in sorted(counts.items())
        )
        pieces.append(f"{budget}:{ordered}")
    return "; ".join(pieces)


def baseline_record(results_root: Path, dataset: str) -> dict[str, Any]:
    path = (
        results_root
        / dataset
        / "development"
        / "LITERATURE_BASELINES"
        / "baseline_selection.json"
    )
    if not path.exists():
        return {"status": "NOT_YET_RUN"}
    payload = load_json(path)
    arx = payload.get("arx") or {}
    pnarx = payload.get("pnarx") or {}
    selected_order = pnarx.get("selected_order")
    selected_candidate = next(
        (
            row
            for row in pnarx.get("candidates", [])
            if row.get("order") == selected_order
        ),
        {},
    )
    return {
        "status": payload.get("status", "FAILED"),
        "arx_history": arx.get("history"),
        "arx_validation_aic": arx.get("validation_aic_mean"),
        "arx_stable_simulation": arx.get("stable_simulation"),
        "pnarx_history": pnarx.get("history"),
        "pnarx_selected_order": selected_order,
        "pnarx_validation_aic": selected_candidate.get("validation_aic_mean"),
        "pnarx_stable_simulation": selected_candidate.get("stable_simulation"),
        "source_commit": payload.get("source_commit"),
        "sha256": sha256(path),
    }


def dependency_track_record(
    results_root: Path, dataset: str, horizon: int, track: str
) -> dict[str, Any]:
    path = spectral_path(results_root, dataset, horizon).with_name(
        f"spectral_{track}.json"
    )
    if not path.exists():
        return {"status": "NOT_YET_RUN"}
    payload = load_json(path)
    selected = (payload.get("penalty") or {}).get("selected") or {}
    return {
        "status": payload.get("status", "FAILED"),
        "validation_mse": selected.get("validation_mse_mean"),
        "effective_df": selected.get("effective_df"),
        "relative_kkt_residual": selected.get("relative_kkt_residual"),
        "source_commit": payload.get("source_commit"),
        "sha256": sha256(path),
    }


def build_status(results_root: Path) -> dict[str, Any]:
    test_count, violations = audit_official_test_access(results_root)
    if violations:
        raise RuntimeError(
            "Official test access guard failed in: " + ", ".join(violations)
        )

    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        direct = {
            str(horizon): direct_record(spectral_path(results_root, dataset, horizon))
            for horizon in HORIZONS
        }
        bootstrap = {
            str(horizon): bootstrap_record(
                bootstrap_path(results_root, dataset, horizon)
            )
            for horizon in HORIZONS
        }
        datasets[dataset] = {
            "h3_shared_history_direct": direct,
            "every_horizon_bootstrap": bootstrap,
            "literature_baseline": baseline_record(results_root, dataset),
            "dependency_ar_track": {
                str(horizon): dependency_track_record(
                    results_root, dataset, horizon, "AR"
                )
                for horizon in HORIZONS
            },
        }

    coverage_path = results_root / "amplitude_coverage_audit.json"
    coverage = load_json(coverage_path) if coverage_path.exists() else {}
    preflight_path = results_root / "PB1_REPAIR_PREFLIGHT_STATUS.json"
    preflight = load_json(preflight_path) if preflight_path.exists() else {}

    completed_direct = sum(
        record["status"] == "COMPLETED"
        for dataset in datasets.values()
        for record in dataset["h3_shared_history_direct"].values()
    )
    completed_bootstrap = sum(
        record["status"] == "COMPLETED"
        for dataset in datasets.values()
        for record in dataset["every_horizon_bootstrap"].values()
    )

    return {
        "schema_version": 2,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "artifact": "PB1_DEVELOPMENT_REPAIR_V2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_head(),
        "overall_status": "BLOCKED_BY_MISSING_METADATA",
        "protocol_freeze_allowed": False,
        "confirmation_allowed": False,
        "official_test_access_count": test_count,
        "official_test_rows_loaded": 0,
        "preflight_status": preflight.get("status", "NOT_YET_RUN"),
        "completed_h3_direct_count": completed_direct,
        "completed_every_horizon_bootstrap_count": completed_bootstrap,
        "expected_h3_direct_count_without_tanks": 12,
        "expected_bootstrap_count_without_tanks": 12,
        "datasets": datasets,
        "coverage_audit": {
            "status": coverage.get("status", "NOT_YET_RUN"),
            "sha256": sha256(coverage_path) if coverage_path.exists() else None,
        },
        "blocking_prerequisites": [
            {
                "id": "H2_REPRESENTATION_GATE_THRESHOLDS",
                "status": "BLOCKED_BY_MISSING_METADATA",
                "detail": (
                    "The repair plan enables a representation-coverage gate and "
                    "Lepski stability, but does not freeze their numerical thresholds."
                ),
            },
            {
                "id": "TANKS_TRAIN_ONLY_EXTRAPOLATION_POLICY",
                "status": "BLOCKED_BY_MISSING_METADATA",
                "detail": (
                    "13 of 324 validation X samples are outside the train-fitted "
                    "amplitude domain; clipping or validation-fitted expansion is forbidden."
                ),
            },
        ],
        "claim_boundary": {
            "development_predictive_evidence_allowed": True,
            "official_benchmark_claim_allowed": False,
            "structural_rank_claim_allowed": False,
            "causal_claim_allowed": False,
        },
    }


def build_report(status: dict[str, Any]) -> str:
    lines = [
        "# PB1 Development Repair V2 报告",
        "",
        "## 结论",
        "",
        (
            "本轮 repair-v2 的可执行 development 工作已汇总，但整体状态仍为 "
            "`BLOCKED_BY_MISSING_METADATA`，因此不能生成 `PB1_PROTOCOL_FREEZE_V2.json`，"
            "也不能进入 confirmation。所有已检查产物的 "
            f"`official_test_access_count={status['official_test_access_count']}`。"
        ),
        "",
        (
            f"当前完成 H3 direct {status['completed_h3_direct_count']} 项、"
            f"每-horizon bootstrap {status['completed_every_horizon_bootstrap_count']} 项。"
            "Tanks 的 spectral 轨道没有被静默裁剪或用 validation 扩张样条域。"
        ),
        "",
        "## H3 Shared-History Direct Forecast",
        "",
        "| 数据集 | h | 状态 | validation MSE | RMSE | EDF | KKT | free-run |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        records = status["datasets"][dataset]["h3_shared_history_direct"]
        for horizon in HORIZONS:
            record = records[str(horizon)]
            lines.append(
                "| {dataset} | {horizon} | {status} | {mse} | {rmse} | "
                "{edf} | {kkt} | {free} |".format(
                    dataset=dataset,
                    horizon=horizon,
                    status=record["status"],
                    mse=fmt(record.get("validation_mse")),
                    rmse=fmt(record.get("validation_rmse")),
                    edf=fmt(record.get("effective_df")),
                    kkt=fmt(record.get("relative_kkt_residual")),
                    free=record.get("free_run_status", "NOT_APPLICABLE"),
                )
            )

    lines.extend(
        [
            "",
            "## 固定模型的每-horizon Rank Bootstrap",
            "",
            "| 数据集 | h | 状态 | B | 自动块长 | 最大 KKT | rank 频数（尾能量预算） |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for dataset in DATASETS:
        records = status["datasets"][dataset]["every_horizon_bootstrap"]
        for horizon in HORIZONS:
            record = records[str(horizon)]
            lines.append(
                "| {dataset} | {horizon} | {status} | {replicates} | {block} | "
                "{kkt} | {ranks} |".format(
                    dataset=dataset,
                    horizon=horizon,
                    status=record["status"],
                    replicates=fmt(record.get("replicates")),
                    block=fmt(record.get("automatic_block_length")),
                    kkt=fmt(record.get("maximum_relative_kkt_residual")),
                    ranks=rank_frequency_text(record.get("rank_frequencies")),
                )
            )

    lines.extend(
        [
            "",
            "## 文献基线与覆盖审计",
            "",
            "| 数据集 | ARX history | ARX AIC | pNARX order | pNARX AIC | 状态 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for dataset in ("cascaded_tanks", "silverbox"):
        baseline = status["datasets"][dataset]["literature_baseline"]
        arx_history = baseline.get("arx_history") or {}
        lines.append(
            "| {dataset} | nx={nx}, ny={ny} | {arx_aic} | {order} | "
            "{pnarx_aic} | {status} |".format(
                dataset=dataset,
                nx=arx_history.get("nx", "—"),
                ny=arx_history.get("ny", "—"),
                arx_aic=fmt(baseline.get("arx_validation_aic")),
                order=fmt(baseline.get("pnarx_selected_order")),
                pnarx_aic=fmt(baseline.get("pnarx_validation_aic")),
                status=baseline["status"],
            )
        )
    lines.extend(
        [
            "",
            "| 数据集 | H3 spectral | 覆盖审计说明 |",
            "|---|---|---|",
            (
                "| pwh | 已运行 | train-fitted 幅值域覆盖 validation；"
                "H1 free-run 递推后越界，按协议失败。 |"
            ),
            (
                "| whpn | 已运行 | train-fitted 幅值域覆盖 validation；"
                "H1 free-run 递推后越界，按协议失败。 |"
            ),
            (
                "| cascaded_tanks | BLOCKED | X validation 中 "
                "13/324 点超出 train-fitted 域；未裁剪、未使用 validation 扩域。 |"
            ),
            (
                "| silverbox | 已运行 | 幅值域覆盖；H1 free-run COMPLETED。 |"
            ),
            "",
            "## WHPN AR-only 依赖轨道复核",
            "",
            "| h | 状态 | validation MSE | EDF | KKT |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    whpn_ar = status["datasets"]["whpn"]["dependency_ar_track"]
    for horizon in (10, 20):
        record = whpn_ar[str(horizon)]
        lines.append(
            "| {horizon} | {status} | {mse} | {edf} | {kkt} |".format(
                horizon=horizon,
                status=record["status"],
                mse=fmt(record.get("validation_mse")),
                edf=fmt(record.get("effective_df")),
                kkt=fmt(record.get("relative_kkt_residual")),
            )
        )
    lines.extend(
        [
            "",
            "修复后的 h20 AR-only 候选通过原坐标 KKT 门槛；该轨道仅用于依赖"
            "完整性复核，不改变已经冻结的 XAR penalty 或 rank bootstrap。",
            "",
            "## 停止线与尚缺的冻结项",
            "",
            "1. H2 的 representation-coverage gate 和 Lepski stability 只有开关，"
            "没有预注册数值阈值；这些阈值会实质改变 history/resolution 选择。",
            "2. Tanks 需要一条只由 train 决定的外推/域策略。当前协议明确禁止静默裁剪，"
            "也禁止根据 validation 扩张样条域。",
            "3. 在上述两项冻结前，H2、Tanks spectral、统一 protocol freeze 和 "
            "official confirmation 均保持未运行。",
            "",
            "## 结论边界",
            "",
            "- 当前数值只属于 development validation，不是官方 test 结果。",
            "- rank 是冻结表示与惩罚下的预测压缩审计，不是结构 rank 或因果发现。",
            "- 未读取 official test、未运行 confirmation、未涉及 PB2 或私有 CZ。",
            "",
            "## 复核信息",
            "",
            f"- 汇总源码提交：`{status['source_commit']}`",
            f"- 生成时间（UTC）：`{status['generated_at_utc']}`",
            "- 机器可读状态：`PB1_DEVELOPMENT_REPAIR_V2_STATUS.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results" / "public_benchmarks" / "pb1_repair_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "PB1_DEVELOPMENT_REPAIR_V2_REPORT.md",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=ROOT / "PB1_DEVELOPMENT_REPAIR_V2_STATUS.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = build_status(args.results_root.resolve())
    args.output.write_text(build_report(status), encoding="utf-8")
    args.status_output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "report": str(args.output),
                "status_file": str(args.status_output),
                "official_test_access_count": status["official_test_access_count"],
                "protocol_freeze_allowed": status["protocol_freeze_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
