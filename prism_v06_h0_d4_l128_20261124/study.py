"""Prepare a private workspace and reproduce the matched PRISM development study."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RELEASE = Path(__file__).resolve().parent
HYBRID = "PRISM_V06_KCW_QC32_H0_D4_L128_SUITE_20261124"
LEGACY = "PRISM_V06_KCWA_H0_D4_L128_BASELINES_20261124"
DIAGNOSTICS = "PRISM_V06_KCWA_DIAGNOSTICS_20261124"
REPEATS = "PRISM_V06_CLASSIC_BASELINES_REPEAT_20261124"
NOVEMBER = "PRISM_CZ_D0_DEBUTANIZER_D4_20261122"
RODS = ("CZ_Rod1", "CZ_Rod2")
ENTITIES = ("SRU_H2S", *RODS)
SRU = "PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025"
CZ = "cz_sru_matched_20261024"
DEFAULT_NEURAL = (RELEASE.parent / "prism_paper_sru_cz_e128_20261026/source/baselines/CZ_Rod1")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as block:
        for chunk in iter(lambda: block.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    data = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise ValueError(f"Workspace metadata differs at {path}; use a new workspace")
    path.write_text(data, encoding="utf-8")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Workspace file differs at {target}; use a new workspace")
    else:
        shutil.copyfile(source, target)


def prepare(work, inputs, neural, verify_reference):
    if work.is_relative_to(RELEASE.parent):
        raise ValueError("Put the working directory outside the Git checkout")
    if not (neural / "baseline_models.py").is_file() or not (neural / "vendor/TimeXer").is_dir():
        raise ValueError(f"Missing previous publication's baseline source at {neural}")
    protocol = json.loads((RELEASE / "PROTOCOL.json").read_text(encoding="utf-8"))
    files = {}
    for entity in ENTITIES:
        spec = protocol["entities"][entity]
        for part in ("train", "development"):
            raw = inputs / entity / f"{part}.npz"
            got = digest(raw)
            if verify_reference and got != spec["reference_sha256"][part]:
                raise ValueError(f"Reference SHA-256 mismatch for {entity}/{part}.npz")
            files[entity, part] = (raw, got)

    sources = RELEASE / "workflow"
    for p in sorted(sources.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            copy(p, work / p.relative_to(sources))

    for entity in ENTITIES:
        folder = work / SRU if entity == "SRU_H2S" else work / CZ / entity
        cfg = protocol["entities"][entity]
        for part in ("train", "development"):
            copy(files[entity, part][0], folder / "data" / f"{part}.npz")
        copy(neural / "baseline_models.py", folder / "baseline_models.py")
        vendor = neural / "vendor/TimeXer"
        for p in sorted(vendor.rglob("*.py")):
            if "__pycache__" not in p.parts:
                copy(p, folder / "vendor/TimeXer" / p.relative_to(vendor))
        hashes = {part: files[entity, part][1] for part in ("train", "development")}
        if entity == "SRU_H2S":
            save(folder / "REGISTRATION.json", dict(train_sha256=hashes["train"],
                      development_file_sha256=hashes["development"]))
        else:
            save(folder / "REGISTRATION.json", {"data_files":
                {part: {"export_sha256": sha, "columns": cfg["columns"]} for part, sha in hashes.items()}})

    a_code = work / NOVEMBER / "CZ/runtime/delta_CZ"
    save(work / NOVEMBER / "CZ/CPU_REGISTRATION.json",
         {"code_sha256": {str(p): digest(p) for p in sorted(a_code.glob("*.py"))}})
    save(work / "WORKSPACE.json", {"prepared": True, "protocol": protocol["protocol"],
         "reference_input_policy": "verified" if verify_reference else "user supplied"})
    print(f"Prepared {work}")


def dispatch(work, args):
    if not (work / "WORKSPACE.json").exists():
        raise ValueError("Run the prepare command first")
    cpu = str(Path(args.cpu_python).resolve())
    gpu = str(Path(args.gpu_python or args.cpu_python).resolve())
    env = dict(os.environ, PRISM_CPU_PYTHON=cpu, PRISM_GPU_PYTHON=gpu)

    def run(group, script, *params, use_gpu=False):
        print(group, script, *params, flush=True)
        subprocess.run([gpu if use_gpu else cpu, "-B", "-u", script, *params],
                       cwd=work / group, env=env, check=True)

    def queue(jobs):
        with ThreadPoolExecutor(args.jobs) as pool:
            all_jobs = [pool.submit(run, *job) for job in jobs]
            for job in all_jobs:
                job.result()

    def hybrid():
        run(HYBRID, "run.py", "register")
        run(HYBRID, "sru.py", "register")
        queue([(HYBRID, "run.py", "train", e) for e in RODS] + [(HYBRID, "sru.py", "fit")])
        run(HYBRID, "run.py", "freeze")
        queue([(HYBRID, "run.py", "evaluate", e) for e in RODS] + [(HYBRID, "sru.py", "evaluate")])
        run(HYBRID, "run.py", "summarize")

    def baselines():
        run(HYBRID, "baselines/suite.py")
        run(HYBRID, "summarize.py")

    def legacy():
        run(LEGACY, "suite.py", "register")
        queue([(LEGACY, "suite.py", "fit", e) for e in ENTITIES])
        run(LEGACY, "suite.py", "freeze")
        queue([(LEGACY, "suite.py", "evaluate", e) for e in ENTITIES])
        queue([(LEGACY, "suite.py", "baseline", "linear", e) for e in ENTITIES])
        for e in ENTITIES:
            run(LEGACY, "suite.py", "baseline", "neural", e, use_gpu=True)
        run(LEGACY, "suite.py", "summarize")

    def diagnostics():
        run(DIAGNOSTICS, "ablation.py", "register")
        queue([(DIAGNOSTICS, "ablation.py", "fit", e, variant)
               for e in ENTITIES for variant in ("no_C", "no_W", "no_KCW")])
        run(DIAGNOSTICS, "ablation.py", "freeze")
        queue([(DIAGNOSTICS, "ablation.py", "evaluate", e) for e in ENTITIES] +
              [(DIAGNOSTICS, "explain.py", e) for e in ENTITIES])
        run(DIAGNOSTICS, "timing.py")
        if args.reference_timing:
            run(DIAGNOSTICS, "reference_timing.py")
        run(DIAGNOSTICS, "export.py")

    def repeats():
        with ThreadPoolExecutor(2) as pool:
            classical = pool.submit(run, REPEATS, "classical.py")
            neural = pool.submit(run, REPEATS, "neural.py", use_gpu=True)
            classical.result()
            neural.result()
        queue([(REPEATS, "core_runs.py", e) for e in ENTITIES])
        run(REPEATS, "summarize.py")

    stages = dict(hybrid=hybrid, baselines=baselines, legacy=legacy, diagnostics=diagnostics, repeats=repeats)
    for stage in stages if args.stage == "all" else (args.stage,):
        stages[stage]()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare", help="materialize a workspace from authorized NPZ files")
    setup.add_argument("--workspace", type=Path, required=True)
    setup.add_argument("--inputs", type=Path, required=True)
    setup.add_argument("--baseline-source", type=Path, default=DEFAULT_NEURAL)
    setup.add_argument("--verify-reference-files", action="store_true")
    execute = sub.add_parser("run", help="run one workflow stage or all of them")
    execute.add_argument("stage", choices=("hybrid", "baselines", "legacy", "diagnostics", "repeats", "all"))
    execute.add_argument("--workspace", type=Path, required=True)
    execute.add_argument("--cpu-python", default=sys.executable)
    execute.add_argument("--gpu-python")
    execute.add_argument("--jobs", type=int, default=3)
    execute.add_argument("--reference-timing", action="store_true")
    args = ap.parse_args()
    work = args.workspace.resolve()
    if args.command == "prepare":
        prepare(work, args.inputs.resolve(), args.baseline_source.resolve(), args.verify_reference_files)
    else:
        if args.jobs < 1:
            ap.error("--jobs must be positive")
        dispatch(work, args)


if __name__ == "__main__":
    main()
