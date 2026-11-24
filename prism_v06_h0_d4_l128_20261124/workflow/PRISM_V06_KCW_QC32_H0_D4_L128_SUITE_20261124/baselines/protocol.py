"""Matched H=0, D=4, L=128 data and train-only folds for all three entities."""

import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
CODE_ROOT = ROOT
REFERENCE = BASE / "PRISM_CZ_D0_DEBUTANIZER_D4_20261122"
CACHE = BASE / "cz_sru_matched_20261024"
SRU = BASE / "PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025"
DRIVER = BASE / "PRISM_PAPER_DRIVER_A_20261030"
CPU = Path(os.getenv("PRISM_CPU_PYTHON", sys.executable))
GPU = Path(os.getenv("PRISM_GPU_PYTHON", sys.executable))
CASE = "CZ"
H, D, L = 0, 4, 128
FEEDBACK = H + D
FIRST_ISSUE = 192
CHANNELS = 6
GEOMETRY = {"SRU_H2S": (3328, 6048, 8064), "CZ_Rod1": (6636, 12061, 16082),
            "CZ_Rod2": (6810, 12376, 16501)}
RODS = tuple(GEOMETRY)
NAMES = ("LSTM", "Transformer_original", "Transformer_expanded", "iTransformer", "TimeXer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def model_source(entity):
    return SRU if entity == "SRU_H2S" else CACHE / entity


def file_hashes(entity):
    if entity == "SRU_H2S":
        source = read(SRU / "REGISTRATION.json")
        return {"train": source["train_sha256"], "development": source["development_file_sha256"]}
    return read(ROOT.parent / "REGISTRATION.json")["data"][entity]["data_sha256"]


def register():
    sources = [ROOT / name for name in ("protocol.py", "suite.py")]
    sources += [REFERENCE / name for name in ("cpu.py", "neural.py")]
    for entity in RODS:
        base = model_source(entity)
        sources.append(base / "baseline_models.py")
        sources.extend(sorted((base / "vendor" / "TimeXer").rglob("*.py")))
    registration = dict(
        protocol=dict(H=H, D=D, L=L),
        rows={entity: dict(train_issues=[FIRST_ISSUE, t-D+1], development=[t, e],
                           data_sha256=file_hashes(entity)) for entity, (_, t, e) in GEOMETRY.items()},
        source_sha256={str(p): sha(p) for p in sources},
        baselines=["Persistence", "AR", "ARX", *NAMES],
        training="Fresh two-fold train-only level/delta selection, followed by full legal-training refits.",
        neural="November 22 candidate grids, epoch selection, strict FP32, three seeds (0, 1, 2).",
        score="Identical full development segment within each entity.",
    )
    path = ROOT / "REGISTRATION.json"
    if path.exists():
        assert read(path) == registration
    else:
        write(path, registration)
    write(ROOT / "CPU_REGISTRATION.json", {"code_sha256": registration["source_sha256"]})


def ensure_bound(name):
    registration = read(ROOT / "REGISTRATION.json")
    if name == "neural.py":
        assert str(REFERENCE / name) in registration["source_sha256"]
    assert all(sha(Path(p)) == digest for p, digest in registration["source_sha256"].items())
    return registration


def values(entity, evaluate=False):
    _, train_end, dev_end = GEOMETRY[entity]
    registration = read(ROOT / "REGISTRATION.json")
    if evaluate:
        seal = read(ROOT / "ALL_FROZEN.json")
        assert seal["registration"] == sha(ROOT / "REGISTRATION.json")
        assert all(sha(ROOT / p) == digest for p, digest in seal["receipts"].items())
    data = []
    for part, count in (("train", train_end), ("development", dev_end-train_end)):
        if not evaluate and part == "development":
            break
        path = model_source(entity) / "data" / f"{part}.npz"
        assert sha(path) == registration["rows"][entity]["data_sha256"][part]
        with np.load(path) as stored:
            a = stored["values"]
        assert a.shape == (count, CHANNELS) and np.isfinite(a).all()
        data.append(a)
    return np.vstack(data)


def issues(entity, evaluate=False):
    _, train_end, dev_end = GEOMETRY[entity]
    return np.arange(train_end, dev_end) if evaluate else np.arange(FIRST_ISSUE, train_end-D+1)


def folds(origins):
    for lo, hi in ((.70, .85), (.85, 1.)):
        held = origins[int(lo*len(origins)):int(hi*len(origins))]
        fit = origins[origins < held[0] - (L+D-1)]
        assert fit[-1] < held[0] - (L+D-1)
        yield fit, held


def sequence(data, origins):
    indices = origins[:, None] - np.arange(L-1, -1, -1)[None, :]
    assert indices.min() >= D
    return np.concatenate([data[indices, 1:], data[indices-D, 0, None]], axis=2)
