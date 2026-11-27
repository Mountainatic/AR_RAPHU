"""Controlled near-lag and projected-ridge experiment on the fixed TEP/SRU protocol."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle

import numpy as np

from laguerre_ecm import admission, fit_joint
from laguerre_screen import (
    CONFIGS, SRU_SOURCE, Frozen, append, blocks, digest, endo_fit, innovation_coef,
    pulse, read_sru_prefix, read_tep, receipt, rmse, save_json, sru_series, train_rows,
    ExogenousMap, SRU_CLEAN_LIMIT, TEP_LIMIT,
)


RANGES = (4, 8, 12)
VARIANTS = ("fixed", "guided")
# A uses zero near-lag smoothness to isolate the extra exact taps. B includes zero
# plus a small near-lag curvature penalty, crossed with two curvatures for U.
B_GRID = ((0., .015), (0., .15), (.002, .015), (.002, .15))
TIM_SHIFT = .12133276609413964
SOURCES = ("laguerre_control.py", "laguerre_ecm.py", "laguerre_screen.py")


def endogenous_gate(train):
    losses = []
    for stop, test in blocks(train):
        data = train.through(test[-1]+1)
        _, endo, coef = endo_fit(data, train_rows(train.delay, stop))
        delta = (data.truth-data.anchor())[test]
        losses.append(delta**2-(delta-endo[test]@coef)**2)
    return admission(losses, train.delay)


def upstream(data, rows, cfg, allow_endo, variant, tau, lx):
    center, endo, coef = endo_fit(data, rows)
    if not allow_endo:
        coef[:] = 0
    view = data.views()
    mapping = ExogenousMap.fit(view, rows, cfg.dynamics, cfg.padding, cfg.width)
    design = mapping.design(view)
    resid = data.truth-data.anchor()-endo@coef
    if variant == "whole":
        joint = fit_joint(design, endo, resid, rows, cfg.sparsity, cfg.curvature, .008, cfg.refit)
    else:
        from laguerre_projection import fit_projected
        joint = fit_projected(design, endo, resid, rows, cfg, variant, tau, lx)
    model = Frozen(center, coef, mapping, joint, np.zeros(6))
    anchor = data.anchor()
    safe = anchor+endo@coef
    effect, trust = joint.predict(design, endo)
    return model, safe, safe+effect, trust


def fit_candidate(train, prefix, cfg, gate, variant, tau, lx):
    losses_exo, losses_inn, raw, folds = [], [], [], []
    for stop, test in blocks(train):
        rows = train_rows(train.delay, stop)
        data = train.through(test[-1]+1)
        base, safe, q, trust = upstream(data, rows, cfg, gate, variant, tau, lx)
        b = innovation_coef(data, rows, q, trust)
        full = q+trust*(data.innovation(q)@b)
        error = data.truth[test]-q[test]
        losses_exo.append((data.truth[test]-safe[test])**2-error**2)
        losses_inn.append(error**2-(data.truth[test]-full[test])**2)
        raw.append(error)
        folds.append(dict(n=len(error), rmse=float(np.sqrt(np.mean(error**2))),
                          variance=float(error.var()), joint=base.joint.audit))
    exo = admission(losses_exo, train.delay)
    inn = admission(losses_inn, train.delay)
    inn["deployed"] = bool(inn["admitted"] and exo["admitted"])
    rows = train_rows(train.delay, len(train))
    model, _, q, trust = upstream(train, rows, cfg, gate, variant, tau, lx)
    if not exo["admitted"]:
        model.joint = replace(model.joint, theta=np.zeros_like(model.joint.theta),
                              adjustment=np.zeros_like(model.joint.adjustment),
                              weights=np.zeros_like(model.joint.weights))
    if inn["deployed"]:
        model.innov_coef = innovation_coef(train, rows, q, trust)
    dev = slice(len(train), len(prefix))
    clean = model.predict(prefix)
    score = rmse(clean["prediction"][dev], prefix.truth[dev])
    if inn["deployed"]:
        coef = model.innov_coef
        model.innov_coef = np.zeros_like(coef)
        without = model.predict(prefix)
        off = rmse(without["prediction"][dev], prefix.truth[dev])
        if off <= score:
            score, clean = off, without
            inn["deployed"] = False
        else:
            model.innov_coef = coef
    residual = np.concatenate(raw)
    return model, dict(exo=exo, innovation=inn, joint=model.joint.audit,
                       folds=folds, exo_oof_variance=float(residual.var()),
                       exo_oof_rmse=float(np.sqrt(np.mean(residual**2))),
                       development_rmse=score)


def load_prefix(task, source, delay):
    read = (lambda part: read_tep(source, part, delay)) if task == "tep" else (
        lambda part: read_sru_prefix(source, part))
    train, dev = read("train"), read("development")
    expected = (6000, 2000) if task == "tep" else (6048, 2016)
    if (len(train), len(dev)) != expected:
        raise ValueError("Partition lengths changed")
    return train, dev


def projection_errors(dynamics):
    impulse = dynamics.batch(np.r_[1., np.zeros(4095)])
    u, _, _ = np.linalg.svd(impulse, full_matrices=False)
    return {str(lag):float(np.sqrt(max(1-np.sum(u[lag]**2), 0))) for lag in range(5, 10)}


def fit(task, delay, source, prior, out, near, variant, grid_index):
    evidence = receipt(source, task, ("train", "development"))
    previous = json.loads((prior/"FROZEN.json").read_text())
    if previous["data"] != evidence or previous["delay"] != delay or previous["task"] != task:
        raise ValueError("A/B must use the original training and development split")
    cfg = CONFIGS[task][previous["selected"]]
    stored = previous["configs"][previous["selected"]]
    if json.loads(json.dumps(asdict(cfg))) != stored:
        raise ValueError("Original support and penalties changed")
    cfg = replace(cfg, dynamics=replace(cfg.dynamics, near=near))
    tau, lx = (0., 0.) if variant == "whole" else B_GRID[grid_index]
    train, development = load_prefix(task, source, delay)
    gate = endogenous_gate(train)
    model, audit = fit_candidate(train, append(train, development), cfg, gate["admitted"], variant, tau, lx)
    out.mkdir(parents=True, exist_ok=False)
    artifact = out/"MODEL.pkl"
    artifact.write_bytes(pickle.dumps(model, protocol=5))
    files = SOURCES if variant == "whole" else (*SOURCES, "laguerre_projection.py")
    lock = dict(task=task, delay=delay, data=evidence, model_sha256=digest(artifact),
                config=asdict(cfg), variant=variant, prior_sha256=digest(prior/"FROZEN.json"),
                source={f:digest(Path(__file__).with_name(f)) for f in files}, tau=tau,
                lambda_x=lx, endo=gate, projection_error=projection_errors(cfg.dynamics), **audit)
    save_json(out/"FROZEN.json", lock)
    print(out.name, "dev RMSE", audit["development_rmse"], "OOF variance", audit["exo_oof_variance"],
          "innovation admitted", audit["innovation"]["admitted"], flush=True)


def freeze(case):
    results = {f.parent.name: json.loads(f.read_text()) for f in case.glob("*/FROZEN.json")}
    a = [f"A{near}" for near in RANGES]
    groups = {kind:[f"B_{kind}_{i}" for i in range(len(B_GRID))] for kind in VARIANTS}
    m = lambda arms: min(arms, key=lambda arm: results[arm]["development_rmse"])
    choices = {"A":m(a), **{kind:m(arms) for kind, arms in groups.items()}}
    choices["B"] = m(list(choices[k] for k in VARIANTS))
    selected = a + [choices[k] for k in VARIANTS]
    lock = {arm:digest(case/arm/"FROZEN.json") for arm in results}
    save_json(case/"SELECTION.json", dict(choices=choices, score=selected, manifests=lock))
    print(case.name, choices, flush=True)


def check_lock(task, delay, source, case, arm, lock):
    path = case/arm/"FROZEN.json"
    if digest(path) != lock["manifests"][arm]:
        raise ValueError("Fit has changed since selection")
    f = json.loads(path.read_text())
    if f["data"] != receipt(source, task, ("train", "development")) or (f["task"],f["delay"]) != (task,delay):
        raise ValueError("Score inputs changed")
    for file, sha in f["source"].items():
        if digest(Path(__file__).with_name(file)) != sha:
            raise ValueError(f"Source changed after fitting {arm}: {file}")
    model = path.with_name("MODEL.pkl")
    if digest(model) != f["model_sha256"]:
        raise ValueError("Frozen model changed")
    return model, f


def load_full(task, delay, source, workbook):
    train, dev = load_prefix(task, source, delay)
    if task == "tep":
        series = append(append(train, dev), read_tep(source, "tail", delay))
    else:
        import pandas as pd
        if digest(workbook) != SRU_SOURCE:
            raise ValueError("Unexpected SRU workbook")
        a = pd.read_excel(workbook, header=None, skiprows=1).to_numpy(dtype=float)
        series = sru_series(a[:, [5, 0, 1, 2, 3, 4]])
        np.testing.assert_allclose(series.x[:8064], np.vstack((train.x,dev.x)), rtol=0, atol=2e-16)
        np.testing.assert_allclose(series.lab[:8064], np.r_[train.lab,dev.lab], rtol=0, atol=2e-16)
    return train, dev, series


def srupressure(model, series, clean, start, scale):
    moved = model.predict(pulse(series, start, scale))
    window = slice(start,start+512)
    e = moved["prediction"][window]-series.truth[window]
    mae = lambda x: float(np.mean(abs(x)))
    shift = mae(moved["prediction"][window]-clean["prediction"][window])
    return dict(n=512, fallback_max_abs=float(np.max(abs(moved["prediction"][window]-moved["endo"][window]))),
                disturbed_rmse=float(np.sqrt(np.mean(e**2))), disturbed_mae=mae(e),
                output_shift_mae=shift, shift_reduction_vs_tim=1-shift/TIM_SHIFT,
                active_zero_fraction=float(np.mean(moved["trust"][window]==0)),
                oracle_clean_shift_mae=mae(moved["endo"][window]-series.truth[window]))


def score(task, delay, source, case, workbook):
    selection = json.loads((case/"SELECTION.json").read_text())
    arms = selection["score"]
    frozen = [check_lock(task, delay, source, case, arm, selection) for arm in arms]
    output = case/"RESULTS.json"
    if output.exists():
        raise FileExistsError(output)
    train, dev, series = load_full(task, delay, source, workbook)
    start = len(train)+len(dev)
    tail = slice(start,len(series))
    pressure_scale = train.x.std(axis=0)
    results = {}
    for arm,(artifact,f) in zip(arms,frozen,strict=True):
        model = pickle.loads(artifact.read_bytes())
        pred = model.predict(series)
        error = rmse(pred["prediction"][tail],series.truth[tail])
        info = dict(tail_rmse=error, clean_n=len(series)-start, development_rmse=f["development_rmse"],
                    projection_error=f["projection_error"], exo_oof_variance=f["exo_oof_variance"],
                    exo_oof_rmse=f["exo_oof_rmse"], innovation=f["innovation"], joint=f["joint"],
                    passed=error <= (SRU_CLEAN_LIMIT if task=="sru" else TEP_LIMIT))
        if task == "sru":
            info["pressure"] = srupressure(model, series, pred, start, pressure_scale)
        results[arm] = info
    tail_file = workbook if task == "sru" else source/"tail.npz"
    save_json(output, dict(seal=digest(case/"SELECTION.json"), tail_sha256=digest(tail_file),
                           choices=selection["choices"], results=results))
    print(case.name, [(arm,result["tail_rmse"]) for arm,result in results.items()], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fitting = sub.add_parser("fit")
    scoring = sub.add_parser("score")
    freezing = sub.add_parser("freeze")
    for p in (fitting, scoring):
        p.add_argument("task", choices=("tep","sru"))
        p.add_argument("delay", type=int)
        p.add_argument("source", type=Path)
    fitting.add_argument("prior", type=Path)
    fitting.add_argument("out", type=Path)
    fitting.add_argument("near", type=int, choices=RANGES)
    fitting.add_argument("--variant", choices=("whole",*VARIANTS), default="whole")
    fitting.add_argument("--grid", type=int, default=0, choices=range(len(B_GRID)))
    for p in (scoring, freezing):
        p.add_argument("case", type=Path)
    scoring.add_argument("--workbook", type=Path)
    a = parser.parse_args()
    if a.action == "fit":
        if a.variant != "whole" and a.near != 12:
            parser.error("The matched projected comparison fixes r=12")
        fit(a.task,a.delay,a.source,a.prior,a.out,a.near,a.variant,a.grid)
    elif a.action == "freeze":
        freeze(a.case)
    else:
        if a.task == "sru" and a.workbook is None:
            parser.error("SRU scoring requires its workbook")
        score(a.task,a.delay,a.source,a.case,a.workbook)


if __name__ == "__main__":
    main()

