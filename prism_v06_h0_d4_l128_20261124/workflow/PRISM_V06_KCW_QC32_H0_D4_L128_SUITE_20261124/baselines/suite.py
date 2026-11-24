"""Run the November 22 baseline code against the matched three-entity protocol."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import runpy
import subprocess
import sys

from protocol import (CPU, GPU, NAMES, REFERENCE, ROOT, RODS, ensure_bound, model_source,
                      read, register, sha, write)


def worker(action, entity, model=None):
    ensure_bound("neural.py" if model or action == "eval_neural" else "cpu.py")
    if action in ("linear", "eval_linear"):
        code = runpy.run_path(str(REFERENCE / "cpu.py"))
        code["train_linear" if action == "linear" else "evaluate_linear"](entity)
    else:
        sys.path.insert(0, str(model_source(entity)))
        import baseline_models
        assert Path(baseline_models.__file__).resolve() == (model_source(entity) / "baseline_models.py").resolve()
        code = runpy.run_path(str(REFERENCE / "neural.py"))
        if action == "neural":
            code["train"](entity, model)
        else:
            code["eval_models"](entity)


def dispatch(action, entity, model=None):
    label = f"{action}_{entity}" + (f"_{model}" if model else "")
    interpreter = GPU if action in ("neural", "eval_neural") else CPU
    log = ROOT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        command = [str(interpreter), "-B", "-u", __file__, action, entity]
        subprocess.run(command + ([model] if model else []), stdout=handle, stderr=subprocess.STDOUT, check=True)
    print("COMPLETE", label, flush=True)


def seal():
    receipts = {}
    for entity in RODS:
        base = ROOT / entity
        linear = base / "linear.json"
        assert sha(base / "linear.pkl") == read(linear)["sha256"]
        receipts[str(linear.relative_to(ROOT))] = sha(linear)
        for name in NAMES:
            receipt = base / "neural" / name / "FROZEN.json"
            assert all(sha(receipt.parent / p) == digest for p, digest in read(receipt)["files"].items())
            receipts[str(receipt.relative_to(ROOT))] = sha(receipt)
    write(ROOT / "ALL_FROZEN.json", dict(registration=sha(ROOT / "REGISTRATION.json"), receipts=receipts))


def main():
    register()
    # Start the longest searches first so the shorter jobs can fill the remaining GPU slots.
    names = ("Transformer_expanded", "Transformer_original", "LSTM", "TimeXer", "iTransformer")
    entities = ("CZ_Rod2", "CZ_Rod1", "SRU_H2S")
    with ThreadPoolExecutor(3) as cpu, ThreadPoolExecutor(int(os.getenv("PRISM_GPU_JOBS", "2"))) as gpu:
        tasks = [cpu.submit(dispatch, "linear", entity) for entity in RODS]
        tasks.extend(gpu.submit(dispatch, "neural", entity, name) for name in names for entity in entities)
        for task in tasks:
            task.result()
    seal()
    for entity in RODS:
        dispatch("eval_linear", entity)
        dispatch("eval_neural", entity)
    write(ROOT / "COMPLETE.json", dict(registration=sha(ROOT / "REGISTRATION.json"),
                                      freeze=sha(ROOT / "ALL_FROZEN.json")))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(*sys.argv[1:])
    else:
        main()
