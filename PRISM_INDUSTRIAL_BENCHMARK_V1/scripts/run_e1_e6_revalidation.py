from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from prism_benchmark.e1e6_revalidation import (
    audit_before_gate,
    phase_a_gate,
    prepare_joint_diagnostic_inputs,
    run_e1_cached,
    run_e2,
    run_e5_pre,
    write_d1_so2_eta_report,
    write_d2_tep_joint_vs_a_report,
    write_freeze,
    write_json,
    write_provenance,
)
from prism_benchmark.v2_views import development_dynamic_views
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL
from prism_benchmark.v211_joint_stability import (
    ETA_PRED_GRID,
    JOINT_CANDIDATES,
    K_REPRESENTATIONS,
    run_joint_stability_view,
)


SO2_HEAD = "SRU_SO2_REP_H1__H1__W1"
TEP_HEAD = "TEP_G_NOWCAST_H0__H0__W1"


def _find_view(shared: Path, head: str, availability: str, proxy: str):
    matches = [
        view
        for view in development_dynamic_views(shared)
        if view.head.head_id == head
        and view.availability_scenario == availability
        and view.proxy_policy == proxy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one diagnostic view, found {len(matches)}")
    return matches[0]


def _counterfactual_spec(kind: str) -> str:
    if kind == "d1":
        return ",".join(
            f"J_KA|CHANNEL_COMPRESSED|{eta}" for eta in (0.01, 1.0)
        )
    return ",".join(
        f"{route}|{representation}|{eta}"
        for route in JOINT_CANDIDATES
        for representation in K_REPRESENTATIONS
        for eta in ETA_PRED_GRID
    )


def run_joint_diagnostic(args: argparse.Namespace) -> None:
    kind = args.kind
    head = SO2_HEAD if kind == "d1" else TEP_HEAD
    availability = "record_time"
    proxy = "primary" if kind == "d1" else "proxy_excluded"
    shared = (args.shared_root / ("sru_shared" if kind == "d1" else "tep_shared")).resolve()
    diagnostic_results = (args.output / "DIAGNOSTIC_WORK" / kind.upper() / "results").resolve()
    baseline_results = (args.baseline_run / "results").resolve()
    a_results = (
        baseline_results if kind == "d1" else (args.tep_repair_run / "results").resolve()
    )
    prepare_joint_diagnostic_inputs(
        diagnostic_results, baseline_results, a_results, head
    )
    os.environ["PRISM_REVALIDATION_JOINT_COUNTERFACTUALS"] = _counterfactual_spec(kind)
    view = _find_view(shared, head, availability, proxy)
    result = run_joint_stability_view(
        shared,
        args.project.resolve(),
        diagnostic_results,
        None,
        view,
        PUBLIC_ALL_PROTOCOL,
    )
    if result.get("status") != "PASS":
        raise RuntimeError(f"{kind} Joint diagnostic failed: {result}")
    write_json(
        args.output / "DIAGNOSTIC_WORK" / kind.upper() / "STATUS.json",
        {
            "status": "COMPLETED",
            "head": head,
            "counterfactual_count": len(result.get("report_only_counterfactuals", {})),
            "test_accessed": result.get("test_accessed"),
            "ood_accessed": result.get("ood_accessed"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen strict nested-OOF E1--E6 revalidation."
    )
    parser.add_argument(
        "stage",
        choices=("provenance", "joint-diagnostic", "diagnostic-reports", "freeze", "phase-a"),
    )
    parser.add_argument("--kind", choices=("d1", "d2"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--tep-repair-run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.project = (
        args.project.resolve()
        if args.project is not None
        else (args.repo / "PRISM_INDUSTRIAL_BENCHMARK_V1").resolve()
    )
    args.output = args.output.resolve()
    args.shared_root = args.shared_root.resolve()
    args.baseline_run = args.baseline_run.resolve()
    args.tep_repair_run = args.tep_repair_run.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.stage == "provenance":
        write_provenance(
            args.output,
            args.repo,
            args.shared_root,
            args.baseline_run,
            args.tep_repair_run,
        )
    elif args.stage == "joint-diagnostic":
        if args.kind is None:
            parser.error("joint-diagnostic requires --kind d1 or d2")
        run_joint_diagnostic(args)
    elif args.stage == "diagnostic-reports":
        write_d1_so2_eta_report(
            args.output,
            args.output / "DIAGNOSTIC_WORK/D1/results",
        )
        write_d2_tep_joint_vs_a_report(
            args.output,
            args.output / "DIAGNOSTIC_WORK/D2/results",
            args.shared_root / "tep_shared",
        )
    elif args.stage == "freeze":
        provenance = write_provenance(
            args.output,
            args.repo,
            args.shared_root,
            args.baseline_run,
            args.tep_repair_run,
        )
        if provenance["status"]:
            raise RuntimeError("protocol freeze requires a clean git worktree")
        write_freeze(args.output, args.repo, provenance)
    elif args.stage == "phase-a":
        freeze_path = args.output / "PROVENANCE/FINAL_ABLATION_PROTOCOL_FREEZE.json"
        if not freeze_path.is_file():
            raise FileNotFoundError("Phase A requires the completed protocol freeze")
        e1 = run_e1_cached(
            args.output,
            args.baseline_run,
            args.tep_repair_run,
            args.shared_root,
        )
        e2, null = run_e2(args.output, workers=args.workers)
        run_e5_pre(args.output, e1, e2)
        audit = audit_before_gate(args.output, e2, null)
        verdict = phase_a_gate(args.output, e2, null, audit)
        print(json.dumps({"phase_a_verdict": verdict}, sort_keys=True))


if __name__ == "__main__":
    main()
