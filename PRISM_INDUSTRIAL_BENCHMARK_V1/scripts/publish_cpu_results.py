from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256-file", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    arguments = parser.parse_args()
    decision = arguments.results / "C6_FINAL/CPU_FINAL_DECISION.json"
    report = arguments.results / "C6_FINAL/CPU_FINAL_REPORT.md"
    if not decision.is_file() or not report.is_file():
        raise FileNotFoundError("C6 final report/decision missing")
    generated = arguments.project / "generated_results/PRISM_INDUSTRIAL_CPU_RESULTS_V1"
    generated.mkdir(parents=True, exist_ok=True)
    for path in (decision, report, arguments.results / "C6_FINAL/CPU_FINAL_METRICS.csv", arguments.results / "C6_FINAL/CROSS_TASK_RANKS.csv", arguments.results / "C6_FINAL/BOOTSTRAP_PAIRED.csv", arguments.results / "CHAIN_STATUS.json"):
        if path.is_file():
            shutil.copy2(path, generated / path.name)
    run(["git", "add", "PRISM_INDUSTRIAL_BENCHMARK_V1"], arguments.project.parent)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=arguments.project.parent)
    if status.returncode:
        run(["git", "commit", "-m", "Publish PRISM industrial CPU V1 results"], arguments.project.parent)
        run(["git", "push", "origin", "prism-industrial-benchmark-v1-strict-restart"], arguments.project.parent)
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("gh is not installed; source push completed but release upload cannot continue")
    existing = subprocess.run([gh, "release", "view", arguments.release_tag], cwd=arguments.project.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode:
        run([gh, "release", "create", arguments.release_tag, "--title", "PRISM Industrial CPU Results V1", "--notes-file", str(report), "--latest=false"], arguments.project.parent)
    run([gh, "release", "upload", arguments.release_tag, str(arguments.bundle), str(arguments.sha256_file), str(report), "--clobber"], arguments.project.parent)
    payload = {"status": "PASS", "release_tag": arguments.release_tag, "bundle": str(arguments.bundle)}
    (arguments.results / "PUBLISH_RESULT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RELEASE_TAG={arguments.release_tag}")


if __name__ == "__main__":
    main()
