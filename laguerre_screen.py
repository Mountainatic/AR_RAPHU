"""Fit and score the PRISM Laguerre feasibility screen on frozen TEP/SRU splits."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle

import numpy as np

from laguerre_ecm import Dynamics, ExogenousMap, admission, fit_joint, ridge, scale_columns


TEP_TRAIN = "d65900c94dc250fec38b411ee92b0abd4295078189f9976cbb790115b884afd0"
SRU_PREFIX = {
    "train.npz": "c4e9ac57c8a60b6e948941bca5cf9323ea1b95cd1c433d74444a46b9b970452f",
    "development.npz": "8118be3f5b728c8ad3f9ccea50ed957d8d41319f6ea207b9de251506f31b137a",
}
SRU_SOURCE = "952c7cbe2f945305a722a01e97d62e05e1f99301277c15c7f6bb528672ac81e4"
ENDO = Dynamics(4, ((.66, 3), (.92, 3)))
INNOVATION = Dynamics(2, ((.65, 2), (.90, 2)))
# Warmup keeps the historical scoring rows. It does not size either state bank.
WARMUP = 64
ALPHA = 0.0003
FIT_FRACTION = .50
TEP_LIMIT = .00270
SRU_CLEAN_LIMIT = .03050
SRU_SHIFT_LIMIT = .01850


@dataclass(frozen=True)
class Config:
    dynamics: Dynamics
    padding: tuple
    width: float
    sparsity: float
    curvature: float
    refit: float


COMPACT = Dynamics(4, ((.72, 4), (.94, 6)))
MEDIUM = Dynamics(4, ((.55, 4), (.87, 8)))
SRU_SHORT = Dynamics(4, ((.55, 4), (.86, 5)))
CONFIGS = {
    "tep": (
        Config(COMPACT, (.08, .08), .8, .005, .05, .001),
        Config(COMPACT, (2., 1.4), 1.1, .004, .35, .004),
        Config(MEDIUM, (.90, .75), 1.1, .004, .35, .003),
        Config(MEDIUM, (2., 1.4), 1.1, .002, .60, .002),
        Config(Dynamics(4, ((.50, 7), (.86, 9))), (2., 1.4), 1.1, .002, .60, .002),
    ),
    "sru": (
        Config(SRU_SHORT, (1.45, .78), .8, .004, .035, .001),
        Config(SRU_SHORT, (2., .98), .7, .007, .01, .003),
        Config(SRU_SHORT, (2., 1.13), .65, .007, .006, .004),
        Config(SRU_SHORT, (2., 1.13), .65, .007, .09, .007),
        Config(Dynamics(8, ((.52, 4), (.90, 5))), (1.45, .96), .8, .006, .04, .001),
        Config(Dynamics(4, ((.45, 6), (.86, 6))), (2., .98), .7, .006, .012, .002),
        Config(Dynamics(4, ((.5, 7), (.81, 4))), (2., 1.13), .65, .006, .008, .004),
        Config(MEDIUM, (2., 1.13), .65, .006, .012, .004),
    ),
}

@dataclass
class Series:
    x: np.ndarray
    lab: np.ndarray
    truth: np.ndarray
    delay: int

    def __len__(self):
        return len(self.x)

    def with_x(self, x):
        return Series(x, self.lab, self.truth, self.delay)

    def through(self, stop):
        return Series(self.x[:stop], self.lab[:stop], self.truth[:stop], self.delay)

    def anchor(self):
        return lag(self.lab, self.delay)

    @property
    def anchor_delay(self):
        return self.delay

    def views(self):
        return np.stack((self.x, self.x-lag(self.x, self.delay, fill="initial")), axis=2).reshape(len(self), -1)

    def endo(self, center):
        maturity = self.anchor()
        increments = np.diff(maturity, prepend=0.)
        increments[:self.delay+1] = 0.
        return np.column_stack((np.ones(len(self)), maturity-center, ENDO.batch(increments)))

    def innovation(self, endo_and_exo):
        valid = np.arange(len(self)) >= WARMUP+self.delay
        residual = np.where(valid, self.lab-endo_and_exo, 0.)
        return INNOVATION.batch(lag(residual, self.delay))


def lag(x, d, fill="zero"):
    if not d:
        return x.copy()
    out = np.zeros_like(x)
    out[d:] = x[:-d]
    if fill == "initial":
        out[:d] = x[:1]
    return out


def append(a, b):
    return Series(np.vstack((a.x,b.x)), np.r_[a.lab,b.lab], np.r_[a.truth,b.truth], a.delay)


def read_tep(folder, part, d):
    with np.load(folder/f"{part}.npz", allow_pickle=False) as a:
        if a["X"].shape[1] != 34 or not np.all(np.diff(a["tick"]) == 1) or np.unique(a["segment"]).size != 1:
            raise ValueError("Expected the regular TEP Stream 11 full-DCS clock")
        return Series(a["X"], a["lab"], a["y"], d)


def sru_series(values):
    return Series(values[:, 1:], values[:, 0], values[:, 0], 8)


def read_sru_prefix(folder, part):
    with np.load(folder/f"{part}.npz", allow_pickle=False) as data:
        return sru_series(data["values"])


def pulse(series, start, scale):
    x = series.x.copy()
    signs = np.random.default_rng(20261125).choice((-1, 1), size=256)
    x[start:start+512] += 4 * scale * np.column_stack((signs, -signs)).reshape(-1, 1)
    return series.with_x(x)


def endo_fit(series, rows):
    center = series.lab[:rows[-1]+1].mean()
    endo = series.endo(center)
    x = endo[rows]
    s = scale_columns(x)
    coefs = ridge(x/s, (series.truth-series.anchor())[rows], np.full(len(s), ALPHA))/s
    return center, endo, coefs


@dataclass
class Frozen:
    center: float
    endo_coef: np.ndarray
    exo_map: ExogenousMap
    joint: object
    innov_coef: np.ndarray

    def base(self, series):
        endo = series.endo(self.center)
        anchor = series.anchor()
        y0 = anchor+endo@self.endo_coef
        correction, trust = self.joint.predict(self.exo_map.design(series.views()), endo)
        return anchor, y0, correction, trust

    def predict(self, series):
        anchor, y0, correction, trust = self.base(series)
        q = y0+correction
        innov = trust*(series.innovation(q)@self.innov_coef)
        return dict(persistence=anchor, endo=y0, exo=q, prediction=q+innov, trust=trust)

    def stream(self, delay, channels):
        return RegularPredictor(self, delay, channels)


class RegularPredictor:
    """One tick accepts current X and the lab report released on this tick."""

    def __init__(self, model, delay, channels):
        if delay < 1:
            raise ValueError("The feasibility screen uses strictly positive report delays")
        self.model = model
        self.tick = 0
        self.delay = delay
        self.sensor_history = np.zeros((delay, channels))
        self.pending = np.zeros(delay)
        self.previous_anchor = 0.
        self.endo = ENDO.stream(())
        self.exo = model.exo_map.stream()
        self.innovation = INNOVATION.stream(())

    def step(self, x, mature_lab):
        i = self.tick % self.delay
        if self.tick == 0:
            self.sensor_history[:] = x
        change = mature_lab-self.previous_anchor if self.tick > self.delay else 0.
        self.previous_anchor = mature_lab
        endo = np.r_[1., mature_lab-self.model.center, self.endo.step(change)]
        raw, trust = self.exo.step(np.column_stack((x, x-self.sensor_history[i])).ravel())
        self.sensor_history[i] = x
        y0 = mature_lab+endo@self.model.endo_coef
        rho = trust@self.model.joint.weights
        q = y0+np.einsum("gtp,gtp,g->", raw, self.model.joint.theta, trust)+rho*(endo@self.model.joint.adjustment)
        residual = mature_lab-self.pending[i] if self.tick >= WARMUP+2*self.delay else 0.
        self.pending[i] = q
        self.tick += 1
        return q+rho*(self.innovation.step(residual)@self.model.innov_coef), y0, rho


def train_rows(delay, stop):
    # The last training target has its report available before the next fold.
    return np.arange(WARMUP+delay, stop-delay)


def blocks(train):
    start = WARMUP+train.delay
    edges = np.linspace(start+int(FIT_FRACTION*(len(train)-start)), len(train), 5, dtype=int)
    return tuple((lo, np.arange(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:]))


def innovation_coef(series, rows, prediction, trust):
    x = trust[rows, None]*series.innovation(prediction)[rows]
    s = scale_columns(x)
    return ridge(x/s, (series.truth-prediction)[rows], np.full(len(s), .004))/s


def fit_candidate(train, prefix, config, gate_endo):
    folds = []
    for stop, test in blocks(train):
        rows = train_rows(train.delay, stop)
        data = train.through(test[-1]+1)
        center, endo, coef = endo_fit(data, rows)
        if not gate_endo:
            coef[:] = 0
        views = data.views()
        mapping = ExogenousMap.fit(views, rows, config.dynamics, config.padding, config.width)
        joint = fit_joint(mapping.design(views), endo, data.truth-data.anchor()-endo@coef,
                          rows, config.sparsity, config.curvature, .008, config.refit)
        provisional = Frozen(center, coef, mapping, joint, np.zeros(6))
        _, y0, q, trust = provisional.base(data)
        q += y0
        inn = innovation_coef(data, rows, q, trust)
        forecast = q+trust*(data.innovation(q)@inn)
        folds.append((data.truth[test], y0[test], q[test], forecast[test], joint.audit))
    exo = admission([(y-y0)**2-(y-q)**2 for y,y0,q,_,_ in folds], train.delay)
    inn = admission([(y-q)**2-(y-f)**2 for y,_,q,f,_ in folds], train.delay)
    inn["admitted"] = bool(inn["admitted"] and exo["admitted"])
    rows = train_rows(train.delay, len(train))
    center, endo, coef = endo_fit(train, rows)
    if not gate_endo:
        coef[:] = 0
    mapping = ExogenousMap.fit(train.views(), rows, config.dynamics, config.padding, config.width)
    joint = fit_joint(mapping.design(train.views()), endo, train.truth-train.anchor()-endo@coef,
                      rows, config.sparsity, config.curvature, .008, config.refit)
    final_audit = joint.audit
    trial = Frozen(center, coef, mapping, joint, np.zeros(6)).predict(prefix)
    trial_dev = rmse(trial["prediction"][len(train):], prefix.truth[len(train):])
    if not exo["admitted"]:
        joint = type(joint)(np.zeros_like(joint.theta), np.zeros_like(joint.adjustment),
                            np.zeros_like(joint.weights), joint.audit)
    model = Frozen(center, coef, mapping, joint, np.zeros(6))
    if inn["admitted"]:
        _, y0, q, trust = model.base(train)
        model.innov_coef = innovation_coef(train, rows, q+y0, trust)
    dev = slice(len(train), len(prefix))
    pressure = pulse(prefix, len(train), train.x.std(axis=0)) if train.delay == 8 else None

    def dev_metrics():
        clean = model.predict(prefix)["prediction"]
        metrics = dict(development_rmse=rmse(clean[dev], prefix.truth[dev]))
        if pressure is not None:
            pulse_rows = slice(len(train), len(train)+512)
            shifted = model.predict(pressure)["prediction"]
            metrics["development_shift_mae"] = float(np.mean(abs(shifted[pulse_rows]-clean[pulse_rows])))
        return metrics

    audit = dict(exo=exo, innovation=inn, screening=final_audit,
                 forward_screening=[f[-1] for f in folds],
                 open_exo_development_rmse=trial_dev)
    selected_metrics = dev_metrics()
    inn["deployed"] = inn["admitted"]
    if inn["admitted"]:
        previous_coef = model.innov_coef
        model.innov_coef = np.zeros_like(previous_coef)
        base = dev_metrics()
        audit["development_innovation_comparison"] = dict(off=base, on=selected_metrics)
        rank = lambda m: (m.get("development_shift_mae", 0.) > SRU_SHIFT_LIMIT, m["development_rmse"])
        if rank(base) <= rank(selected_metrics):
            selected_metrics = base
            inn["deployed"] = False
        else:
            model.innov_coef = previous_coef
    audit.update(selected_metrics)
    return model, audit


def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def receipt(source, task, parts):
    files = [source/f"{p}.npz" for p in parts]
    if task == "tep":
        files.append(source/"DATA_MANIFEST.json")
        if digest(files[0]) != TEP_TRAIN or json.loads(files[-1].read_text())["source"]["event_log"] != (
                "4685b47e49d80742f50e5c3b1847a378b7258282fee9a08d76122db095393668"):
            raise ValueError("Unexpected TEP source")
    hashes = {p.name:digest(p) for p in files}
    if task == "sru" and hashes != SRU_PREFIX:
        raise ValueError("Unexpected SRU prefix")
    return hashes


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run_fit(task, delay, source, out):
    evidence = receipt(source, task, ("train", "development"))
    read = (lambda part: read_tep(source, part, delay)) if task == "tep" else (
        lambda part: read_sru_prefix(source, part))
    train, dev = read("train"), read("development")
    sizes = (6000, 2000) if task == "tep" else (6048, 2016)
    if (len(train), len(dev)) != sizes:
        raise ValueError("Unexpected partition sizes")
    folds = []
    for stop, rows in blocks(train):
        data = train.through(rows[-1]+1)
        _, endo, coef = endo_fit(data, train_rows(delay, stop))
        error = (data.truth-data.anchor())[rows]
        folds.append(error**2-(error-endo[rows]@coef)**2)
    gate_endo = admission(folds, delay)
    prefix = append(train, dev)
    models, audit = [], []
    for i, config in enumerate(CONFIGS[task]):
        model, results = fit_candidate(train, prefix, config, gate_endo["admitted"])
        models.append(model)
        audit.append(results)
        print(f"{task} D{delay}: candidate {i} development RMSE {results['development_rmse']:.7g}"
              f" shift {results.get('development_shift_mae', 0):.7g}", flush=True)
    candidates = [i for i, result in enumerate(audit) if task == "tep" or (
        result["development_shift_mae"] <= SRU_SHIFT_LIMIT)]
    if not candidates:
        best = min(range(len(audit)), key=lambda i: audit[i]["development_shift_mae"])
    else:
        best = min(candidates, key=lambda i: audit[i]["development_rmse"])
    out.mkdir(parents=True, exist_ok=False)
    path = out/"MODEL.pkl"
    path.write_bytes(pickle.dumps(models[best], protocol=5))

    save_json(out/"FROZEN.json", dict(task=task, delay=delay, data=evidence, model_sha256=digest(path),
              configs=list(map(asdict, CONFIGS[task])), selected=best, development_pressure_eligible=candidates,
              endo=gate_endo, candidates=audit,
              source={name:digest(Path(__file__).with_name(name)) for name in (
                  "laguerre_ecm.py", "laguerre_screen.py")}))


def run_score(task, delay, source, out, workbook):
    lock = json.loads((out/"FROZEN.json").read_text())
    path = out/"MODEL.pkl"
    if (lock["task"], lock["delay"], lock["data"], lock["model_sha256"]) != (
            task, delay, receipt(source, task, ("train", "development")), digest(path)):
        raise ValueError("Frozen inputs or model changed")
    if any(digest(Path(__file__).with_name(name)) != expected for name, expected in lock["source"].items()):
        raise ValueError("Prototype source changed since the fit")
    if (out/"RESULTS.json").exists():
        raise FileExistsError("This fit has already been scored")
    model = pickle.loads(path.read_bytes())
    if task == "tep":
        train, dev, tail = (read_tep(source, name, delay) for name in ("train", "development", "tail"))
        series = append(append(train, dev), tail)
        source_hash = digest(source/"tail.npz")
    else:
        import pandas as pd
        source_hash = digest(workbook)
        if source_hash != SRU_SOURCE:
            raise ValueError("Unexpected SRU workbook")
        a = pd.read_excel(workbook, header=None, skiprows=1).to_numpy(dtype=float)
        series = sru_series(a[:, [5, 0, 1, 2, 3, 4]])
        train, dev = (read_sru_prefix(source, name) for name in ("train", "development"))
        np.testing.assert_allclose(series.x[:8064], np.vstack((train.x,dev.x)), rtol=0, atol=2e-16)
        np.testing.assert_allclose(series.lab[:8064], np.r_[train.lab,dev.lab], rtol=0, atol=2e-16)
    start = len(train)+len(dev)
    tail = slice(start, len(series))
    clean = model.predict(series)
    stream = model.stream(delay, series.x.shape[1])
    replay = np.array([stream.step(x, y) for x, y in zip(series.x, lag(series.lab, delay))])
    expected = np.column_stack([clean[k] for k in ("prediction", "endo", "trust")])
    np.testing.assert_allclose(replay, expected, atol=2e-9, rtol=0)
    result = dict(task=task, delay=delay, n=len(series)-start, source_sha256=source_hash,
                  tail_rmse={name:rmse(y[tail], series.truth[tail]) for name,y in clean.items() if name != "trust"},
                  tail_mean_trust=float(clean["trust"][tail].mean()),
                  max_batch_stream_abs_error=float(abs(replay-expected).max()))
    if task == "sru":
        stressed = model.predict(pulse(series, start, train.x.std(axis=0)))
        region = slice(start, start+512)
        result["pressure"] = dict(n=512,
            shift_mae=float(np.mean(abs(stressed["prediction"][region]-clean["prediction"][region]))),
            pulse_rmse=rmse(stressed["prediction"][region], series.truth[region]),
            mean_trust=float(stressed["trust"][region].mean()),
            zero_trust_fraction=float(np.mean(stressed["trust"][region] == 0)))
        result["passed"] = (result["tail_rmse"]["prediction"] <= SRU_CLEAN_LIMIT and
                            result["pressure"]["shift_mae"] <= SRU_SHIFT_LIMIT)
    else:
        result["passed"] = result["tail_rmse"]["prediction"] <= TEP_LIMIT
    save_json(out/"RESULTS.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "score"))
    p.add_argument("task", choices=("tep", "sru"))
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("delay", type=int)
    p.add_argument("--workbook", type=Path)
    a = p.parse_args()
    if a.delay not in ((1, 2) if a.task == "tep" else (8,)):
        p.error("Supported cases are TEP D1/D2 and SRU D8")
    if a.action == "fit":
        run_fit(a.task, a.delay, a.source, a.output)
    else:
        if a.task == "sru" and a.workbook is None:
            p.error("SRU scoring requires --workbook")
        run_score(a.task, a.delay, a.source, a.output, a.workbook)


if __name__ == "__main__":
    main()

