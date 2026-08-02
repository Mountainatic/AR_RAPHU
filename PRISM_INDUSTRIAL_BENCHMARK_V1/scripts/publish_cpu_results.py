from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _upload_bundle(gh: str, tag: str, bundle: Path, project_root: Path) -> list[dict[str, object]]:
    """Upload a bundle directly or as ordered sub-2-GiB release assets."""
    maximum_asset_bytes = 1_800 * 1024 * 1024
    if bundle.stat().st_size <= maximum_asset_bytes:
        run([gh, "release", "upload", tag, str(bundle), "--clobber"], project_root)
        return [{"name": bundle.name, "bytes": bundle.stat().st_size}]
    records: list[dict[str, object]] = []
    with bundle.open("rb") as source:
        index = 0
        while True:
            part = bundle.with_name(f"{bundle.name}.part-{index:03d}")
            digest = hashlib.sha256()
            written = 0
            with part.open("wb") as target:
                while written < maximum_asset_bytes:
                    block = source.read(min(16 * 1024 * 1024, maximum_asset_bytes - written))
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                part.unlink(missing_ok=True)
                break
            run([gh, "release", "upload", tag, str(part), "--clobber"], project_root)
            records.append({"name": part.name, "bytes": written, "sha256": digest.hexdigest(), "order": index})
            part.unlink()
            index += 1
    return records


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
    run([gh, "release", "upload", arguments.release_tag, str(arguments.sha256_file), str(report), "--clobber"], arguments.project.parent)
    assets = _upload_bundle(gh, arguments.release_tag, arguments.bundle, arguments.project.parent)
    asset_manifest = arguments.bundle.with_name("RELEASE_ASSET_MANIFEST.json")
    asset_manifest.write_text(
        json.dumps(
            {
                "original_bundle": arguments.bundle.name,
                "original_bytes": arguments.bundle.stat().st_size,
                "original_sha256_file": arguments.sha256_file.name,
                "parts": assets,
                "reconstruction": f"cat {arguments.bundle.name}.part-* > {arguments.bundle.name}" if len(assets) > 1 else "direct_asset",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run([gh, "release", "upload", arguments.release_tag, str(asset_manifest), "--clobber"], arguments.project.parent)
    release_url = subprocess.check_output([gh, "release", "view", arguments.release_tag, "--json", "url", "--jq", ".url"], cwd=arguments.project.parent, text=True).strip()
    payload = {"status": "PASS", "release_tag": arguments.release_tag, "release_url": release_url, "bundle": str(arguments.bundle), "assets": assets}
    (arguments.results / "PUBLISH_RESULT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RELEASE_TAG={arguments.release_tag}")
    print(f"RELEASE_URL={release_url}")


if __name__ == "__main__":
    main()
