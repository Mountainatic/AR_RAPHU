#!/usr/bin/env python3
"""Create the PB1 development-only technical report and reproducible result bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "public_benchmarks" / "pb1"
REPORT_DATE = "20260727"
REPORT_PATH = RESULT_ROOT / f"PB1_DEVELOPMENT_REPORT_{REPORT_DATE}.md"
STATUS_PATH = RESULT_ROOT / f"PB1_DEVELOPMENT_STATUS_{REPORT_DATE}.json"
RETURN_DIR = ROOT / "return"
ZIP_PATH = RETURN_DIR / f"OPS_UOI_PB1_DEVELOPMENT_RESULTS_{REPORT_DATE}.zip"
ZIP_SHA_PATH = ZIP_PATH.with_suffix(".zip.sha256")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.6e}"
    return f"{value:.6f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def horizon_rows(summary: dict) -> list[str]:
    rows = []
    for item in summary["horizons"]:
        ar = item["tracks"]["AR"]
        xar = item["tracks"]["XAR"]
        gain = item["incremental_external_value"]
        if gain["status"] == "COMPLETED":
            relative = pct(gain["relative_AR_mse_reduction"])
            delta = fmt(gain["delta_X_given_AR_mse"])
        else:
            relative = "未报告（依赖门槛失败）"
            delta = "未报告"
        ranks = item["external_predictive_effective_ranks"]
        rows.append(
            "| {h} | {ar_mse} | {ar_status} | {xar_mse} | {xar_status} | "
            "{delta} | {relative} | {r10}/{r05}/{r02} |".format(
                h=item["horizon"],
                ar_mse=fmt(ar["validation_mse_group_mean"]),
                ar_status=ar["status"],
                xar_mse=fmt(xar["validation_mse_group_mean"]),
                xar_status=xar["status"],
                delta=delta,
                relative=relative,
                r10=ranks["0.1"],
                r05=ranks["0.05"],
                r02=ranks["0.02"],
            )
        )
    return rows


def h1_row(dataset: str, comparison: dict) -> str:
    direct = comparison["direct_protocol_metrics"]
    spectral = comparison["spectral_h3_pilot_metrics"]["full"]
    return (
        f"| {dataset} | {fmt(direct['persistence']['rmse_pooled'])} | "
        f"{fmt(direct['arx']['rmse_pooled'])} | "
        f"{fmt(direct['pnarx']['rmse_pooled'])} | "
        f"{fmt(direct['mlpnarx']['rmse_pooled'])} | "
        f"{fmt(spectral['rmse_pooled'])} |"
    )


def bootstrap_text(dataset: str, bootstrap: dict) -> str:
    frequencies = bootstrap["spectral_tail_budget_rank_frequencies"]
    parts = []
    for budget in ("0.1", "0.05", "0.02"):
        counts = ", ".join(
            f"rank-{rank}: {count}/{bootstrap['replicates']}"
            for rank, count in sorted(frequencies[budget].items())
        )
        parts.append(f"{float(budget):.0%} 尾能量预算下 {counts}")
    return f"- **{dataset}**：{'; '.join(parts)}。"


def build_report(
    source_commit: str,
    pwh_comparison: dict,
    whpn_comparison: dict,
    pwh_horizons: dict,
    whpn_horizons: dict,
    pwh_bootstrap: dict,
    whpn_bootstrap: dict,
) -> str:
    lines = [
        "# OPS-UOI PB1 开发阶段技术报告",
        "",
        "## 技术摘要",
        "",
        "**结论：当前工作到达开发阶段停止线，但 PB1 整体尚未完成；正式状态为 "
        "`PB1_CONFIRMATION=NOT_YET_RUN`、`WHPN_DEVELOPMENT=FAILED`。** "
        "PWH 的已注册开发轨道全部完成；WHPN 的 XAR 轨道在四个"
        "直接预测 horizon 上均完成，但 AR-only 在 h=10 的正则区间认证和 h=20 "
        "的 KKT 门槛失败，因此这两个 horizon 的正式 "
        "`Delta_X_given_AR` 不报告。官方 test 从未读取，因而本文不是 confirmation "
        "结果，也不构成最终公开基准结论。",
        "",
        "在开发验证上，PWH 的过程输入相对 AR-only 在 h=1/5/10/20 分别降低 "
        "71.49%/88.80%/88.18%/69.22% 的 MSE，说明外生输入含有稳定的增量预测"
        "信息。WHPN 在 h=1/5 的对应降低为 26.27%/41.03%；h=10/20 虽然 XAR "
        "点估计更低，但因依赖门槛失败不能形成冻结结论。所有 rank 结果只解释为"
        "当前基函数、惩罚和开发分布下的预测有效秩，不允许解释为系统结构真秩。",
        "",
        "## 一步预测中，Spectral XAR 与 pNARX 接近",
        "",
        "下表使用相同的无未来 X 信息集、相同 train-only 目标 z-score 和相同开发"
        "记录，单位为 pooled RMSE。表格优先于趋势图，因为每个数据集只有一个"
        "一步预测比较点，绘图不会增加信息。",
        "",
        "| 数据集 | Persistence | ARX | pNARX | MLP-NARX | Spectral XAR |",
        "|---|---:|---:|---:|---:|---:|",
        h1_row("PWH", pwh_comparison),
        h1_row("WHPN", whpn_comparison),
        "",
        "PWH 上 Spectral XAR 的 RMSE 为 0.002476，较 pNARX 低约 0.33%，较 "
        "ARX 低约 11.0%；WHPN 上相对 pNARX 的开发优势约 0.11%，幅度很小。"
        "MLP-NARX 在两组数据上均明显较差，不能据此宣称神经基线具有优势。",
        "",
        "## PWH 的外生增量在全部开发 horizon 上成立",
        "",
        "| h | AR MSE | AR 状态 | XAR MSE | XAR 状态 | Delta(X|AR) | "
        "相对 AR 降低 | 预测秩(10%/5%/2%) |",
        "|---:|---:|---|---:|---|---:|---:|---|",
        *horizon_rows(pwh_horizons),
        "",
        "PWH 四个 horizon 的 AR 与 XAR 都通过惩罚区间和数值门槛。过程变量的"
        "边际价值在中等 horizon 最大；h=20 在严格 2% 尾能量预算下点估计升为 "
        "rank-2，但这仍是预测压缩需求，不是结构 rank 发现。",
        "",
        "## WHPN 在 h=10 和 h=20 只能保留点估计",
        "",
        "| h | AR MSE | AR 状态 | XAR MSE | XAR 状态 | Delta(X|AR) | "
        "相对 AR 降低 | 预测秩(10%/5%/2%) |",
        "|---:|---:|---|---:|---|---:|---:|---|",
        *horizon_rows(whpn_horizons),
        "",
        "WHPN h=10 的 AR penalty interval 在允许的两次边界扩展后仍未认证；"
        "h=20 的 AR relative KKT residual 为 1.718e-8，高于冻结的 1e-8。"
        "因此两个 horizon 的 XAR 数值可作为开发诊断保留，但不得计算或引用"
        "正式增量收益。",
        "",
        "## Bootstrap 支持低预测秩，但不支持结构秩声明",
        "",
        bootstrap_text("PWH", pwh_bootstrap),
        bootstrap_text("WHPN", whpn_bootstrap),
        "",
        "bootstrap 固定已选惩罚和分辨率，不在重采样中重新调参。PWH 在全部 "
        "250 次重采样和三个尾能量预算下均为 rank-1；WHPN 在 2% 严格预算下"
        "有 79/250 次需要 rank-2，说明细尾部存在不确定性。由于公开数据没有 "
        "K 层真值证书，`structural_rank_claim_allowed=false`。",
        "",
        "## 范围、数据和指标定义",
        "",
        "- 范围仅包括 PWH 与 WHPN 的 development split；Cascaded Tanks 仍为 "
        "`PENDING_SPLIT_ADEQUACY_AUDIT`，Silverbox 为 "
        "`BLOCKED_BY_MISSING_METADATA`。",
        "- 主预测协议是 direct forecast：X 与 y 只使用到时刻 t，预测 y[t+h]；"
        "不使用未来 X，也不使用中间真实 y。",
        "- `Delta_X_given_AR = MSE_AR - MSE_XAR`；只有 AR 与 XAR 两个依赖"
        "轨道同时通过冻结门槛时才报告。",
        "- 所有 scaler、历史和惩罚选择只使用 train/development validation；"
        "官方 test 访问计数为 0。",
        "",
        "## 模型与验证方法",
        "",
        "一步比较包含 persistence、线性 ARX、2024 文献配置的 pNARX、"
        "MLP-NARX 与 FP64 Spectral XAR。Spectral 轨道使用共享 H1 历史 "
        "(PWH Lx=16, Ly=20；WHPN Lx=18, Ly=15)、固定首个预注册分辨率、"
        "归一化三惩罚和 grouped validation one-SE。rank 在模型与惩罚冻结后"
        "才计算，并用 250 次按 phase/realization 聚类的 bootstrap 检查稳定性。",
        "",
        "## 限制、失败项与鲁棒性边界",
        "",
        "- PB1 confirmation、官方 test 和 OOD test 均为 `NOT_YET_RUN`。",
        "- H2 native-history 与完整 basis-resolution 选择尚未冻结；不能用当前 "
        "H3 pilot 冒充最终配置。",
        "- WHPN h=10/h=20 的 AR 依赖门槛失败，禁止补写正式增量结论。",
        "- Tanks 的 overflow 样本级定义缺失；Silverbox 许可证元数据未解决。",
        "- WHPN 的过程噪声专用 GRU/状态基线尚未实现。",
        "- 结果只支持预测层面的开发证据，不支持因果、结构真秩或官方 benchmark "
        "优胜声明。",
        "",
        "## 下一步必须先解决冻结前置条件",
        "",
        "1. 预注册 H2 history 与 basis-resolution 的嵌套选择顺序及候选空间。",
        "2. 决定 WHPN h=10 penalty 边界失败与 h=20 KKT 失败的预注册处理方式，"
        "不得事后按结果扩网格或放宽阈值。",
        "3. 补齐 Tanks overflow 元数据/替代门槛、Silverbox 许可证和 WHPN "
        "过程噪声对照。",
        "4. 上述条件冻结后生成 `PB1_PROTOCOL_FREEZE.json`，再一次性运行 "
        "confirmation；此前继续保持官方 test 锁箱。",
        "",
        "## 可复核性",
        "",
        f"- 打包前源码提交：`{source_commit}`。",
        "- 结果状态文件：`PB1_DEVELOPMENT_STATUS_20260727.json`。",
        "- 本报告及所有逐模型 JSON/NPZ、配置、相关源码、工具和测试均收入"
        "开发结果包；压缩包内 `PACKAGE_MANIFEST.json` 给出逐文件 SHA256。",
        "",
    ]
    return "\n".join(lines)


def package_files() -> list[Path]:
    roots = [
        RESULT_ROOT,
        ROOT / "configs" / "public_benchmarks",
        ROOT / "src" / "ar_raphu" / "baselines",
        ROOT / "src" / "ar_raphu" / "spectral",
        ROOT / "tests",
    ]
    files: set[Path] = set()
    for directory in roots:
        if directory.exists():
            files.update(path for path in directory.rglob("*") if path.is_file())
    for path in (ROOT / "tools").glob("*pb1*"):
        if path.is_file():
            files.add(path)
    for name in (
        "OPS_UOI_Public_Benchmark_to_CZ_Experiment_Design_and_Code_Reuse_v1_1.md",
        "OPS_UOI_PB1_Literature_Grounded_Development_Preflight_v1_0.md",
        "PROTOCOL_REVISION_PB1.md",
        "pyproject.toml",
    ):
        path = ROOT / name
        if path.exists():
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    pwh_comparison = load_json(
        RESULT_ROOT / "pwh" / "development" / "DEVELOPMENT_MODEL_COMPARISON_H1.json"
    )
    whpn_comparison = load_json(
        RESULT_ROOT / "whpn" / "development" / "DEVELOPMENT_MODEL_COMPARISON_H1.json"
    )
    pwh_horizons = load_json(
        RESULT_ROOT
        / "pwh"
        / "development"
        / "H3_SHARED_HISTORY"
        / "DIRECT_HORIZON_SUMMARY.json"
    )
    whpn_horizons = load_json(
        RESULT_ROOT
        / "whpn"
        / "development"
        / "H3_SHARED_HISTORY"
        / "DIRECT_HORIZON_SUMMARY.json"
    )
    pwh_bootstrap = load_json(
        RESULT_ROOT
        / "pwh"
        / "development"
        / "H3_SHARED_HISTORY"
        / "SPECTRAL_PILOT_H1"
        / "rank_bootstrap.json"
    )
    whpn_bootstrap = load_json(
        RESULT_ROOT
        / "whpn"
        / "development"
        / "H3_SHARED_HISTORY"
        / "SPECTRAL_PILOT_H1"
        / "rank_bootstrap.json"
    )
    source_commit = git("rev-parse", "HEAD")
    generated_at = datetime.now(timezone.utc).isoformat()

    status = {
        "schema_version": 1,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "artifact_scope": "DEVELOPMENT_ONLY",
        "generated_at_utc": generated_at,
        "source_commit_before_packaging": source_commit,
        "overall_status": "NOT_YET_RUN",
        "development_overall_status": "FAILED",
        "development_artifact_delivery_status": "COMPLETED",
        "running_experiment_processes_at_packaging": 0,
        "official_test_rows_loaded": 0,
        "official_test_access_count": 0,
        "protocol_frozen": False,
        "datasets": {
            "pwh": {
                "development": "COMPLETED",
                "direct_horizons": [1, 5, 10, 20],
                "failed_track_gates": [],
                "confirmation": "NOT_YET_RUN",
            },
            "whpn": {
                "development": "FAILED",
                "direct_horizons": [1, 5, 10, 20],
                "failed_track_gates": whpn_horizons["failed_track_gates"],
                "confirmation": "NOT_YET_RUN",
            },
            "cascaded_tanks": {
                "development": "NOT_YET_RUN",
                "status": "PENDING_SPLIT_ADEQUACY_AUDIT",
            },
            "silverbox": {
                "development": "NOT_YET_RUN",
                "status": "BLOCKED_BY_MISSING_METADATA",
                "reason": "LICENSE_METADATA_UNRESOLVED",
            },
        },
        "completed_components": [
            "H1 corrected no-future-X ARX history selection",
            "2024 literature-faithful pNARX development",
            "2024 literature-faithful MLP-NARX development",
            "H3 shared-history FP64 spectral X/AR/XAR direct horizons",
            "H1 frozen-model predictive-rank bootstrap with 250 replicates",
            "main repository and V20 regression suites",
        ],
        "unresolved_prerequisites": [
            "H2 native-history selection nesting not preregistered",
            "basis-resolution selection not frozen",
            "WHPN h10 AR penalty interval not certified",
            "WHPN h20 AR KKT gate failed",
            "Tanks overflow split adequacy unresolved",
            "Silverbox license metadata unresolved",
            "WHPN process-noise comparator not implemented",
        ],
        "claim_boundary": {
            "predictive_development_evidence_allowed": True,
            "confirmation_claim_allowed": False,
            "official_test_claim_allowed": False,
            "structural_rank_claim_allowed": False,
            "causal_claim_allowed": False,
        },
    }
    STATUS_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(
        build_report(
            source_commit,
            pwh_comparison,
            whpn_comparison,
            pwh_horizons,
            whpn_horizons,
            pwh_bootstrap,
            whpn_bootstrap,
        ),
        encoding="utf-8",
    )

    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    files = package_files()
    manifest_files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "package": ZIP_PATH.name,
        "scope": "PB1_DEVELOPMENT_ONLY_NO_RAW_DATA_NO_OFFICIAL_TEST",
        "generated_at_utc": generated_at,
        "source_commit_before_packaging": source_commit,
        "privacy_guards": {
            "raw_data_included": False,
            "private_cz_included": False,
            "credentials_included": False,
            "official_test_rows_included": False,
        },
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr("SOURCE_COMMIT.txt", source_commit + "\n")
    zip_digest = sha256(ZIP_PATH)
    ZIP_SHA_PATH.write_text(f"{zip_digest}  {ZIP_PATH.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "report": str(REPORT_PATH.relative_to(ROOT)),
                "status_file": str(STATUS_PATH.relative_to(ROOT)),
                "zip": str(ZIP_PATH.relative_to(ROOT)),
                "zip_sha256": zip_digest,
                "packaged_files": len(files),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
