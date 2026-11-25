"""Bound training and development access for the PRISM follow-up experiments."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PRIOR = BASE / "PRISM_V06_KCW_QC32_H0_D4_L128_SUITE_20261124"
ORIGINAL = BASE / "PRISM_V06_KCWA_H0_D4_L128_BASELINES_20261124"
REFERENCE = BASE / "PRISM_V06_CLASSIC_BASELINES_REPEAT_20261124"
DIAGNOSTICS = BASE / "PRISM_V06_KCWA_DIAGNOSTICS_20261124"
sys.path[:0] = [str(ORIGINAL), str(PRIOR)]
import suite as legacy

h, np, matched = legacy.hybrid, legacy.np, legacy.baseline
read, write, sha = legacy.read, legacy.write, legacy.sha
CASES = matched.RODS


def inputs(entity, development=False):
    legacy.verify_freeze(entity)
    reg = legacy.registration()
    pipe, data, accessor = legacy.pipeline(entity, reg, development=development)
    model = read(ORIGINAL/entity/"FROZEN_MODEL.json")["original_kcwa"]
    return reg["geometry"][entity], pipe, data, accessor, chain(model)


def chain(model):
    result = {}
    while model is not None:
        result[model["stage"]] = model
        model = model["parent"]
    return result


def score(y, p):
    assert y.shape == p.shape and np.isfinite(p).all()
    error = y-p
    mse = float(np.mean(error*error))
    return dict(RMSE=float(np.sqrt(mse)), MSE=mse, MAE=float(np.mean(np.abs(error))),
                R2=float(1-mse/np.var(y)), bias=float(error.mean()))


def poisoned(data, cut):
    alt = data.copy()
    alt[cut-h.D+1:, 0] += 1234.
    alt[cut+1:, 1:] -= 1200.
    return alt


def registration(script, policy):
    previous = legacy.registration()
    result = dict(protocol=previous["protocol"], geometry=previous["geometry"], policy=policy,
                  source_sha256={p.name:sha(p) for p in (ROOT/"common.py",script)},
                  source_model_release=sha(ORIGINAL/"ALL_FROZEN.json"),
                  reference_scores=sha(REFERENCE/"RESULTS.json"), reserved_tail_accessed=False)
    target = ROOT/f"{script.stem.upper()}_REGISTRATION.json"
    serial = h.json.loads(h.json.dumps(result))
    if target.exists():
        assert read(target)==serial
    else:
        write(target,result)
    return serial


def bound(script):
    path = ROOT/f"{script.stem.upper()}_REGISTRATION.json"
    value = read(path)
    assert all(sha(ROOT/p)==digest for p,digest in value["source_sha256"].items())
    assert value["source_model_release"]==sha(ORIGINAL/"ALL_FROZEN.json")
    assert value["reference_scores"]==sha(REFERENCE/"RESULTS.json")
    return value
