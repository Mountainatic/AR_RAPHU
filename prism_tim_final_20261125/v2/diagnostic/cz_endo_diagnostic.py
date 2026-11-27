"""Development-only diagnostic of high-pass CZ process information conditional on A0."""

import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--v2", required=True, type=Path, help="PRISM v2 source directory")
parser.add_argument("--frozen", required=True, type=Path, help="Sealed E1–E3 run")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.v2.resolve()))

import experiments as base
import model
from scipy import signal
from scipy.linalg import solve


np = model.np
HORIZONS = (16, 48, 64)
DIFFERENCES = (1, 4, 8)
EDGES = (0, 1, 2, 4, 8, 16, 32)
HALF_LIFE = 16
RIDGE = (.1, 1., 10., 100.)
LAGS = np.arange(-32, 33)
N_SHIFTS = 999
SEED = 20261126
MIN_GAIN_PCT = 1.
ENTITY = "CZ_Rod1"


def verify_frozen():
    out = args.frozen
    registration = base.read(out/"REGISTRATION.json")
    results = base.read(out/"RESULTS.json")
    lock = base.read(out/"FROZEN.json")
    if results["frozen"] != base.sha(out/"FROZEN.json"):
        raise ValueError("Frozen manifest mismatch")
    if lock["registration"] != base.sha(out/"REGISTRATION.json"):
        raise ValueError("Registration mismatch")
    for name, digest in registration["source"].items():
        if base.sha(args.v2/name) != digest:
            raise ValueError("Training source mismatch: " + name)
    receipt = base.old.x.TAIL/"SOURCES.json"
    if registration["frozen_data_receipt"] != base.sha(receipt):
        raise ValueError("Data receipt mismatch")
    return lock, registration


def restricted_data(q):
    # Map the authorized array and copy only the training and development prefix.
    path = base.old.x.TAIL/f"{q.name}.npy"
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    if mapped.shape != (q.total, len(q.channels)+1):
        raise ValueError("Unexpected process data shape")
    data = np.array(mapped[:q.dev], order="C", dtype=np.float64)
    np.testing.assert_array_equal(data[:q.train], base.old.data(q))
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite train/development process data")
    return data


def impulses(raw):
    t = np.arange(len(raw))
    delta = [raw-raw[np.maximum(t-lag, 0)] for lag in DIFFERENCES]
    a = 1.-np.exp2(-1./HALF_LIFE)
    initial = np.broadcast_to((1.-a)*raw[0], (1, raw.shape[1])).copy()
    ema, _ = signal.lfilter([a], [1., a-1.], raw, axis=0, zi=initial)
    return np.column_stack([*delta, raw-ema]), delta[1]


def design(raw):
    hi, delta4 = impulses(raw)
    eligible = np.arange(EDGES[-1]-1, len(raw))
    features = np.zeros((len(raw), (len(EDGES)-1)*hi.shape[1]))
    features[eligible] = model.g.bucket(hi, eligible, EDGES).reshape(len(eligible), -1)
    return features, delta4


def load(q, lock):
    name = f"{q.name}_H{q.H}.json"
    path = args.frozen/"frozen"/name
    if lock["files"][name] != base.sha(path):
        raise ValueError("Frozen model mismatch: " + name)
    arm = base.read(path)["arms"]["full"]
    a0 = arm["model"]["A0"]
    if a0 is None or any(arm["model"][m] is not None for m in ("C", "W", "A")):
        raise ValueError("Expected an A0-only frozen route: " + name)
    return arm, a0


def training_residuals(q, data, k):
    end = q.calibration[1]
    engine = model.Engine(q, data[:q.train], k, end)
    starts = model.s.blocks(engine.lo, end, .52, 5)
    indices, errors, folds = [], [], []
    for first, stop in starts:
        prefix = first-q.mature+1
        selected = engine.feedback_select(prefix, True)
        if selected is None:
            raise ValueError("No endogenous candidate in a training fold")
        t = np.arange(first+q.mature, stop)
        # Match the issue-time restart and exclude its deterministic warmup.
        estimate = engine.a_eval(engine.y, first, t, selected)
        indices.append(t)
        errors.append(engine.y[t]-estimate)
        folds.append(dict(train_origin_exclusive=prefix, first_prediction_origin=first,
                          scoring_origins=[int(t[0]), int(t[-1])+1], A0_spec=selected["spec"]))
    return np.concatenate(indices), np.concatenate(errors), folds


def fit_ridge(features, target, lam):
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-10] = 1.
    x = (features-center)/scale
    n = len(x)
    gram = (x.T@x)/n + lam*np.eye(x.shape[1])
    weight = solve(gram, (x.T@target)/n, assume_a="pos")
    return center, scale, weight


def predict(x, fitted):
    center, scale, weight = fitted
    return ((x-center)/scale)@weight


def select_ridge(features, t, residual, q):
    scores = np.zeros(len(RIDGE))
    null = 0.
    n = 0
    first = t[0]
    end = t[-1]+1
    splits = model.s.blocks(first, end, .55, 3)
    for start, stop in splits:
        train = t+q.mature <= start
        test = (t >= start) & (t < stop)
        y = residual[test]
        n += len(y)
        null += y@y
        for i, lam in enumerate(RIDGE):
            fitted = fit_ridge(features[t[train]], residual[train], lam)
            e = y-predict(features[t[test]], fitted)
            scores[i] += e@e
    i = int(scores.argmin())
    risk = {str(lam): float(loss/n) for lam, loss in zip(RIDGE, scores, strict=True)}
    return RIDGE[i], risk, float(null/n), int(n)


def harmonized_rho(t, e, delta, blocks, seed):
    """Lag k is corr(e(t), Delta4 x(t-k)); positive k uses earlier x."""
    xx, yy = [], []
    for indices in blocks:
        inside = indices[32:-32]
        query = t[inside]
        y = e[inside]
        x = delta[query[:, None]-LAGS[None, :]].reshape(len(inside), -1)
        xx.append(x-x.mean(axis=0))
        yy.append(y-y.mean())
    x = np.vstack(xx)
    y = np.concatenate(yy)
    norm = np.sqrt((x*x).sum(axis=0)*(y@y))
    norm = np.maximum(norm, 1e-15)
    observed = (y@x/norm).reshape(len(LAGS), delta.shape[1])
    rng = np.random.default_rng(seed)
    pieces = []
    for y0 in yy:
        shifts = rng.integers(64, len(y0)-63, size=N_SHIFTS)
        pieces.append(y0[(np.arange(len(y0))[None, :]-shifts[:, None]) % len(y0)])
    draws = np.concatenate(pieces, axis=1)
    null = np.max(np.abs((draws@x)/norm), axis=1)
    return observed, null, len(y)


def peak(rho, null, channel, mask):
    lags, vals = LAGS[mask], rho[mask, channel]
    i = int(np.argmax(abs(vals)))
    r = float(vals[i])
    return dict(lag=int(lags[i]), r=r, p_maxT=float((1+np.count_nonzero(null >= abs(r)))/(len(null)+1)))


def diagnostics(h, lock):
    q = base.task(ENTITY, h)
    data = restricted_data(q)
    a, frozen_a0 = load(q, lock)
    features, delta4 = design(data[:, 1:])
    fit_t, train_e, folds = training_residuals(q, data, a["k"])
    lam, risks, cv_null, cv_n = select_ridge(features, fit_t, train_e, q)
    ridge = fit_ridge(features[fit_t], train_e, lam)
    # Restrict all evaluation targets to the end of development.
    t = np.arange(q.train, q.dev-q.H)
    e = model.Engine(q, data, a["k"], int(t[-1])+1)
    frozen_delta = e.a_eval(e.y, q.train, t, frozen_a0)
    z = data[t+q.H, 0]-data[t, 0]
    err = z-frozen_delta
    correction = predict(features[t], ridge)
    after = err-correction
    blocks = np.array_split(np.arange(len(t)), 4)

    def figures(x):
        old = float(np.mean(err[x]**2))
        new = float(np.mean(after[x]**2))
        return dict(n=len(x), A0_mse=old, ridge_mse=new, A0_rmse=float(np.sqrt(old)),
                    ridge_rmse=float(np.sqrt(new)), relative_gain_pct=100*(old-new)/old,
                    target_first=int(t[x[0]]+h), target_end=int(t[x[-1]]+h+1))
    full = figures(np.arange(len(t)))
    gains = [err[x]**2-after[x]**2 for x in blocks]
    gate = model.admission(gains, q.mature)
    rho, null, corr_n = harmonized_rho(t, err, delta4, blocks, SEED+h)
    leads = {}
    for c, name in enumerate(q.channels):
        leads[name] = dict(earlier=peak(rho, null, c, LAGS>0), later=peak(rho, null, c, LAGS<0),
                           at_origin=peak(rho, null, c, LAGS==0))
    # Verify the frozen parent against the already published development result.
    published = base.old.ROOT.parent/"PRISM_V2_E123_20261125/results_v2_final/E1/METRICS.csv"
    if h == 16 and published.is_file():
        import csv
        with published.open(encoding="utf-8-sig", newline="") as f:
            ref = next(r for r in csv.DictReader(f) if all(r[k]==v for k,v in
                       dict(data=q.name, split="development", method="PRISM_v2").items()))
        np.testing.assert_allclose(full["A0_mse"], float(ref["MSE"]), atol=1e-15, rtol=1e-12)
        assert full["n"] == int(ref["N"])
    return dict(h=h, frozen_A0=frozen_a0["spec"], train_oof_n=len(fit_t), train_folds=folds,
                ridge_lambda=lam, training_cv_mse=risks, training_cv_zero_mse=cv_null,
                training_cv_n=cv_n, development=full,
                after_warmup=figures(np.arange(q.mature, len(t))),
                blocks=[figures(x) for x in blocks], hac=gate,
                meets_rule=bool(full["relative_gain_pct"] >= MIN_GAIN_PCT and gate["admitted"]),
                correlation_n=corr_n, correlation_family=len(LAGS)*len(q.channels),
                correlation_peaks=leads)


def main():
    lock, registration = verify_frozen()
    cases = [diagnostics(h, lock) for h in HORIZONS]
    result = dict(dataset=ENTITY, primary_h=16, secondary_h=[48, 64],
                  base_commit=registration["base_commit"], frozen_manifest=base.sha(args.frozen/"FROZEN.json"),
                  source_sha256=base.sha(Path(__file__)), data_receipt=registration["frozen_data_receipt"],
                  read_range="raw rows [0,16082); targets remain before the reserved tail",
                  features=dict(differences=DIFFERENCES, EMA_half_life=HALF_LIFE,
                                bins=EDGES, absolute_levels=False, intercept=False),
                  ridge_grid=RIDGE, gain_rule=dict(min_gain_pct=MIN_GAIN_PCT, T=1.645, positive_blocks_min=3),
                  correlation=dict(transform="delta4", lags=[-32, 32], permutations=N_SHIFTS,
                                   positive_lag="process observation precedes prediction origin",
                                   null="Independent circular shifts of centered residuals within each block; max absolute correlation.",
                                   seed=SEED, validation="descriptive sensitivity; H16 primary; maxT within H"),
                  results=cases)
    base.write(args.out, result)
    for row in cases:
        print(row["h"], row["development"], row["ridge_lambda"], row["hac"]["statistic"], flush=True)


if __name__ == "__main__":
    main()

