"""Chronological K/C/W/A experiments under the frozen H=0, D=4, L=128 protocol."""

import itertools
import sys
from pathlib import Path

from sklearn.linear_model import ElasticNet, Ridge

from common import CASES, ROOT, bound, h, inputs, np, poisoned, read, registration, score, sha, write
from resolution import process
from prism_benchmark.v2_basis import natural_cubic_columns


SCRIPT = Path(__file__).resolve()
DEST = ROOT / "guarded"
D, L = h.D, h.L
EMBARGO = L + D - 1
RAW_WINDOWS = (0, 1, 2, 3, 4, 12, 32, 64, 128)
EMA = (1, 2, 4, 8, 16, 32, 64)
ARMS = {
    "raw_fp": ("legacy", False, False, "ar", "ridge"),
    "raw_robust": ("legacy", False, True, "ar", "ridge"),
    "guard_fp": ("legacy", True, False, "ar", "ridge"),
    "guard_robust": ("legacy", True, True, "ar", "ridge"),
    "natural_guard_robust": ("natural", True, True, "ar", "ridge"),
    "natural_guard_ema": ("natural", True, True, "ema", "ridge"),
    "hybrid_guard_ema_ridge": ("hybrid", True, True, "ema", "ridge"),
    "hybrid_guard_ema_enet": ("hybrid", True, True, "ema", "enet"),
}
POLICY = dict(
    arms=ARMS, K="legacy fixed reference and deployment switch; natural basis reuses original rank and penalties",
    hybrid=dict(windows=RAW_WINDOWS, differencing=D, pointwise_basis_before_bin_mean=True),
    trust="channel-wise cubic smoothstep decay outside early training 1/99 percentiles; zero at one training percentile span",
    W="layer always uses its finalized parent; guarded W clips to its fit 1/99 percentile interval and shrinks by trust",
    A=dict(legacy_profiles=h.LEGACY_PROFILES, ema_half_lives=EMA, ema_widths=[4, 7], fixed_fit=True),
    gate=dict(votes=4, min_wins=3, improvement_fraction=.01, standard_errors=1.,
              inner_label_gap=D-1, interstage_gap=EMBARGO, outer_gap=EMBARGO),
    support_source="K parent block", evaluation="frozen then scored on the original development origins",
    resolution_source=sha(ROOT/"resolution.py"),
)


def register():
    registration(SCRIPT, POLICY)


def indices(lo, hi):
    return np.arange(lo, hi, dtype=np.int64)


class NaturalBasis:
    def __init__(self, values, width):
        x = values.reshape(-1)
        self.mean, self.scale = float(x.mean()), max(float(x.std()), 1e-9)
        z = (x-self.mean)/self.scale
        self.knots = np.unique(np.quantile(z, np.arange(1, width+2)/(width+2)))
        self.center = self.columns(z).mean(axis=0)
        self.dimension = len(self.center)

    def columns(self, z):
        if len(self.knots) < 2:
            return z[:, None]
        lo, hi = self.knots[0], self.knots[-1]
        bounded = np.clip(z, lo, hi)
        x = natural_cubic_columns(bounded, self.knots)[:, 1:]
        # Outside the boundary knots evaluate the exact affine continuation.
        knots = self.knots
        if len(knots) > 2:
            slope = (3*(hi-knots[:-2])-3*(hi-knots[-2]))
            x[:, 1:] += np.maximum(z-hi, 0.)[:, None]*slope
        x[:, 0] = z
        return x

    def transform(self, values):
        z = (values.reshape(-1)-self.mean)/self.scale
        return (self.columns(z)-self.center).reshape(*values.shape, self.dimension)

    def metadata(self):
        return dict(kind="natural", mean=self.mean, scale=self.scale, knots=self.knots.tolist(),
                    center=self.center.tolist(), dimension=self.dimension)

    @classmethod
    def load(cls, state):
        obj = cls.__new__(cls)
        obj.mean, obj.scale, obj.dimension = (state[k] for k in ("mean", "scale", "dimension"))
        obj.knots, obj.center = (np.array(state[k]) for k in ("knots", "center"))
        return obj


def k_surface(pipe, values, y, phi, basis, rep):
    u = pipe.u
    prepared = u.PreparedContractFit(values, y, int(rep["m_x"]), basis, phi,
                                    u.DesignStatistics.from_design(phi.reshape(len(phi), -1), y),
                                    u.DesignStatistics.from_design(phi[:, :, 0], y))
    contract = u.fit_prepared_contract(prepared, rep["family"], int(rep["m_x"]), tuple(rep["lambdas"]))
    pred = u.predict_contract_from_design(phi, contract)
    assert np.isfinite(pred).all()
    return contract


def bucket(point, t, edges):
    x = np.asarray(point)
    if x.ndim == 1:
        x = x[:, None]
    prefix = np.concatenate((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    left, right = np.array(edges[:-1]), np.array(edges[1:])
    return (prefix[t[:, None]+1-left] - prefix[t[:, None]+1-right])/(right-left)[None, :, None]


def fit_k(entity, geo, pipe, data, k, kind):
    lo, hi = geo["parent"]
    f = h.frame(entity, lo, hi, "parent_fit")
    t = f.origin.to_numpy()
    y = data[t, 0] - data[t-D, 0]
    fitted = []
    for rep in k["representations"]:
        v, _ = pipe._values(f, rep)
        if kind == "legacy" or (kind == "natural" and
                                rep["family"] == "LINEAR_DISTRIBUTED_LAG"):
            fitted.append([dict(view="legacy", contract=rep["contract"], support=limits(v))])
            continue
        if kind == "natural":
            basis = NaturalBasis(v, int(rep["m_x"]))
            contract = k_surface(pipe, v, y, basis.transform(v), basis, rep)
            fitted.append([dict(view="natural", contract=contract, support=limits(v))])
            continue
        signal = process(data[:hi], rep, "reference")[:, 0]
        delta = np.r_[np.zeros(D), signal[D:]-signal[:-D]]
        views = []
        for view, x in (("level", signal), ("delta", delta)):
            edges = RAW_WINDOWS if view == "level" else (*RAW_WINDOWS[:-1], L-D)
            width = int(rep["m_x"])
            basis = NaturalBasis(x[lo-edges[-1]+1:hi], width)
            phi = bucket(basis.transform(x), t, edges)
            v = bucket(x, t, edges)[:, :, 0]
            contract = k_surface(pipe, v, y, phi, basis, rep)
            views.append(dict(view=view, edges=edges, contract=contract, support=limits(v)))
        fitted.append(views)
    return fitted


def limits(x):
    low, high = np.quantile(x, [.01, .99], axis=0)
    return np.stack((low, high)).tolist()


def trust(values, support):
    lo, hi = np.asarray(support)
    span = np.maximum(hi-lo, 1e-9*(1+np.maximum(np.abs(lo), np.abs(hi))))
    # Smoothstep from 1 at the support boundary to 0 far outside it.
    u = np.clip(np.max(np.maximum(lo-values, values-hi)/span, axis=1), 0., 1.)
    return (1.-u)**2*(1.+2.*u)


def context(entity, pipe, data, k, models, stop):
    # A source may precede the first fitted or scored row by up to 64 steps.
    start = L-1
    t = indices(start, stop)
    f = h.frame(entity, start, stop, "upstream")
    predictions, weights, names, ref = [], [], [], None
    for i, (rep, views) in enumerate(zip(k["representations"], models)):
        signal = None
        for view in views:
            family, ct = view["view"], view["contract"]
            if family in {"legacy", "natural"}:
                values, _ = pipe._values(f, rep)
                if family == "legacy":
                    prediction = pipe.u.predict_contract(values, ct)
                else:
                    phi = NaturalBasis.load(ct["basis"]).transform(values)
                    prediction = pipe.u.predict_contract_from_design(phi, ct)
            else:
                if signal is None:
                    signal = process(data[:stop], rep, "reference")[:, 0]
                x = signal if family == "level" else np.r_[np.zeros(D), signal[D:]-signal[:-D]]
                values = bucket(x, t, view["edges"])[:, :, 0]
                phi = bucket(NaturalBasis.load(ct["basis"]).transform(x), t, view["edges"])
                prediction = pipe.u.predict_contract_from_design(phi, ct)
            if i == k["reference_index"] and family != "delta":
                ref = len(predictions)
            predictions.append(prediction)
            weights.append(trust(values, view["support"]))
            names.append(f"{rep['channel']}/{family}")
    assert ref is not None
    y = data[:stop, 0]
    anchor = np.zeros(stop); anchor[D:] = y[:-D]
    x = np.zeros((stop, len(predictions))); x[t] = np.column_stack(predictions)
    g = np.zeros_like(x); g[t] = np.column_stack(weights)
    raw = x[:, ref] if k["routing_status"] in {"MODEL", "ADMITTED"} else np.zeros(stop)
    return dict(y=y, anchor=anchor, x=x, g=g, names=names, ref=ref, raw=raw)


def c_base(ctx, guarded):
    return ctx["raw"] * ctx["g"][:, ctx["ref"]] if guarded else ctx["raw"]


def c_features(ctx, rows, mean, scale, guarded):
    x = (ctx["x"][rows]-mean)/scale
    g = ctx["g"][rows]
    return x*g if guarded else x


def c_predict(ctx, rows, parent, model, guarded):
    if model is None:
        return parent[rows]
    x = c_features(ctx, rows, model["mean"], model["scale"], guarded)
    bias = model["intercept"]
    if guarded:
        bias = bias*ctx["g"][rows].max(axis=1)
    return parent[rows]+x@model["coef"]+bias


def w_input(ctx, t, kc, active_c, guarded):
    if active_c:
        coordinate = kc[t]
        weight = ctx["g"][t].min(axis=1)
    else:
        coordinate = ctx["x"][t, ctx["ref"]]
        weight = ctx["g"][t, ctx["ref"]]
    return coordinate, weight if guarded else np.ones(len(t))


def w_predict(pipe, ctx, t, kc, active_c, model, guarded):
    if model is None:
        return kc[t]
    z, weight = w_input(ctx, t, kc, active_c, guarded)
    if guarded:
        z = np.clip(z, *model["limits"])
    return kc[t] + weight*pipe.w.predict_w_correction(z, model["contract"])


def a_features(pipe, residual, rows, spec, start):
    mode, profile = spec
    if mode == "ar":
        offsets = h.a_state.offsets(profile, pipe.A_max)
        source = rows[:, None]-D-offsets
        assert source.min() >= start
        return residual[source]
    # Filter consumes each residual when it matures. Zero starts before any
    # residual from the currently frozen upstream has become available.
    alpha = 1.-np.exp2(-1./np.asarray(EMA[:profile]))
    from scipy.signal import lfilter
    source = residual[start:rows[-1]-D+1]
    if not len(source):
        return np.zeros((len(rows), profile))
    filtered = np.column_stack([lfilter([a], [1., -(1.-a)], source) for a in alpha])
    out = np.zeros((len(rows), profile))
    available = rows-D >= start
    out[available] = filtered[rows[available]-D-start]
    return out


def gate(stage, lo, hi, candidates, fit, predict, parent, target, robust):
    def select(end):
        p = lo + int(.72*(end-lo))
        validation = indices(p, end)
        inner = indices(lo, p-(D-1))
        assert len(inner) >= 26 and len(validation) >= 12
        risks = []
        for spec in candidates:
            child = fit(inner, spec)
            risk = float(np.mean((target[validation]-predict(validation, child))**2))
            risks.append(risk if child is not None else float("inf"))
        if np.isinf(min(risks)):
            return None, None
        spec = candidates[int(np.argmin(risks))]
        return spec, fit(indices(lo, end), spec)

    blocks = lo + np.floor(np.linspace(.55, 1., 5)*(hi-lo)).astype(int)
    blocks[-1] = hi
    audit = []
    for begin, end in zip(blocks[:-1], blocks[1:]):
        spec, model = select(begin-(D-1))
        t = indices(begin, end)
        loss = (target[t]-parent[t])**2
        child_loss = (target[t]-predict(t, model))**2
        audit.append(dict(rows=[int(begin), int(end)], fit_stop=int(begin-(D-1)),
                          parent_mse=float(loss.mean()), child_mse=float(child_loss.mean()),
                          gain=float((loss-child_loss).mean()), spec=spec))
    gains = np.array([r["gain"] for r in audit])
    p = float(np.mean([r["parent_mse"] for r in audit]))
    c = float(np.mean([r["child_mse"] for r in audit]))
    se = float(gains.std(ddof=1)/np.sqrt(len(gains)))
    wins = int(np.sum(gains > 0.))
    threshold = max(.01*p, se) if robust else 16*np.finfo(float).eps*(p+c)
    accepted = bool(gains.mean() > threshold and (wins >= 3 or not robust))
    spec, model = select(hi) if accepted else (None, None)
    return model, dict(stage=stage, rows=[lo, hi], wins=wins, mean_gain=float(gains.mean()),
                       standard_error=se, threshold=threshold, selected=spec, admitted=model is not None,
                       blocks=audit)


def combo(pipe, ctx, lo, hi, arm):
    _, guarded, robust, a_family, c_solver = ARMS[arm]
    k = c_base(ctx, guarded)
    d = ctx["y"]-ctx["anchor"]
    m1 = lo + (hi-lo)//3
    m2 = lo + 2*((hi-lo)//3)
    all_t = indices(0, len(d))

    def fit_c(t, spec):
        alpha, ratio = spec
        raw = ctx["x"][t]
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale < 1e-10] = 1.
        x = c_features(ctx, t, mean, scale, guarded)
        y = d[t]-k[t]
        est = (Ridge(alpha=alpha*len(t)) if c_solver == "ridge" else
               ElasticNet(alpha=alpha*max(float(y.std()), 1e-10), l1_ratio=ratio,
                          max_iter=20000, tol=1e-5))
        est.fit(x, y)
        return dict(mean=mean.tolist(), scale=scale.tolist(), coef=est.coef_.tolist(),
                    intercept=float(est.intercept_))

    c_grid = [(a, 0.) for a in (.0001, .01, 1.)] if c_solver == "ridge" else list(
        itertools.product((.001, .01, .1), (.5, .9)))
    c, ac = gate("C", lo, m1, c_grid, fit_c, lambda t,m:c_predict(ctx, t, k, m, guarded), k, d, robust)
    kc = c_predict(ctx, all_t, k, c, guarded)

    def fit_w(t, spec):
        family, knots, smooth, direction = spec
        z, weight = w_input(ctx, t, kc, c is not None, guarded)
        keep = weight >= .95
        if keep.sum() < 26 or z[keep].std() < 1e-10:
            return None
        z, rows = z[keep], t[keep]
        lim = np.quantile(z, [.01, .99]).tolist()
        if guarded:
            z = np.clip(z, *lim)
        if np.ptp(z) < 1e-10:
            return None
        _, ct = pipe.w.fit_w_correction(z, d[rows]-kc[rows], z, family=family, knot_count=knots,
                                       smoothness=smooth, direction=direction)
        return dict(contract=ct, limits=lim) if pipe.cert(ct) else None

    w_grid = [(pipe.w.MONOTONE, 4, 10., direction) for direction in (-1, 1)]
    w_grid += [(pipe.w.NATURAL_CUBIC, 6, s, 1) for s in (1., 100.)]
    w, aw = gate("W", m1+EMBARGO, m2, w_grid, fit_w,
                 lambda t,m:w_predict(pipe, ctx, t, kc, c is not None, m, guarded), kc, d, robust)
    kcw = w_predict(pipe, ctx, all_t, kc, c is not None, w, guarded)
    residual = d-kcw

    def fit_a(t, spec):
        x = a_features(pipe, residual, t, spec, m2+D-1)
        _, ct = pipe.a.fit_mature_residual_ar(x, residual[t], x, alpha=h.LEGACY_ALPHA, mu=0.)
        return dict(spec=spec, contract=ct) if pipe.cert(ct) else None

    def predict_a(t, model):
        if model is None:
            return kcw[t]
        x = a_features(pipe, residual, t, model["spec"], m2+D-1)
        return kcw[t]+pipe.a.predict_mature_residual_ar(x, model["contract"])

    a_grid = [("ar", p) for p in h.LEGACY_PROFILES] if a_family == "ar" else [("ema", n) for n in (4, 7)]
    a, aa = gate("A", m2+EMBARGO, hi, a_grid, fit_a, predict_a, kcw, d, robust)
    return dict(C=c, W=w, A=a, upstream_available_from=m2+D-1, gates=[ac, aw, aa])


def predict(pipe, ctx, arm, model, t, a_start):
    _, guarded, _, _, _ = ARMS[arm]
    rows = indices(0, int(t[-1])+1)
    k = c_base(ctx, guarded)
    kc = c_predict(ctx, rows, k, model["C"], guarded)
    kcw = w_predict(pipe, ctx, rows, kc, model["C"] is not None, model["W"], guarded)
    r = ctx["y"][:len(rows)]-ctx["anchor"][:len(rows)]-kcw
    final = kcw[t].copy()
    if model["A"] is not None:
        a = model["A"]
        x = a_features(pipe, r, t, a["spec"], a_start)
        final += pipe.a.predict_mature_residual_ar(x, a["contract"])
    anchor = ctx["anchor"][t]
    return {stage: anchor+p for stage,p in
            (("K",k[t]),("KC",kc[t]),("KCW",kcw[t]),("KCWA",final))}


def fit(entity):
    bound(SCRIPT)
    geo, pipe, data, _, stages = inputs(entity)
    k = stages["K"]
    lo, hi = geo["calibration"]
    contexts = {}
    k_models = {}
    for kind in {x[0] for x in ARMS.values()}:
        k_models[kind] = fit_k(entity, geo, pipe, data, k, kind)
        contexts[kind] = context(entity, pipe, data, k, k_models[kind], hi)
    models = {}
    outer = {}
    folds = lo + np.floor(np.linspace(.65, 1., 5)*(hi-lo)).astype(int)
    folds[-1] = hi
    for arm, spec in ARMS.items():
        x = contexts[spec[0]]
        audit = []
        for start, stop in zip(folds[:-1], folds[1:]):
            end = int(start)-EMBARGO
            model = combo(pipe, x, lo, end, arm)
            t = indices(start, stop)
            p = predict(pipe, x, arm, model, t, model["upstream_available_from"])
            audit.append(dict(fit=[lo,end], score=[int(start),int(stop)], gates=model["gates"],
                              metrics={s:score(x["y"][t], z) for s,z in p.items()}))
        outer[arm] = audit
        models[arm] = combo(pipe, x, lo, hi, arm)
        print("FIT", entity, arm, [g["admitted"] for g in models[arm]["gates"]], flush=True)
    path = DEST/entity/"MODEL.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, dict(entity=entity, K=k_models, arms=models, outer=outer))
    write(path.with_name("FIT.json"),dict(entity=entity, sha256=sha(path), score_rows_not_read=True))


def freeze():
    bound(SCRIPT)
    hashes = {}
    for entity in CASES:
        path = DEST/entity/"MODEL.json"
        assert sha(path) == read(path.with_name("FIT.json"))["sha256"]
        hashes[entity] = sha(path)
    write(DEST/"ALL_FROZEN.json",dict(models=hashes))


def evaluate(entity):
    bound(SCRIPT)
    path = DEST/entity/"MODEL.json"
    assert sha(path) == read(DEST/"ALL_FROZEN.json")["models"][entity]
    saved = read(path)
    geo, pipe, data, rebind, k = inputs(entity, development=True)
    k = k["K"]
    lo, hi = geo["development"]
    t = indices(lo, hi)
    truth = data[t, 0]
    cuts = (int(t[23]), int(t[len(t)//2]), int(t[-1]))
    metrics, masks, states = {}, {}, {}
    for kind, ks in saved["K"].items():
        ctx = context(entity, pipe, data, k, ks, hi)
        dirty = []
        clean_accessor = pipe.accessor
        for cut in cuts:
            data_poison = poisoned(data, cut)
            pipe.accessor = rebind(data_poison)
            dirty.append(context(entity, pipe, data_poison, k, ks, cut+1))
        pipe.accessor = clean_accessor
        masks[kind] = {name:dict(mean=float(col.mean()), zero=int((col==0).sum()),
                                 reduced=int((col<1).sum()))
                       for name,col in zip(ctx["names"],ctx["g"][t].T)}
        for arm,spec in ARMS.items():
            if spec[0] != kind:
                continue
            model = saved["arms"][arm]
            a_start = model["upstream_available_from"]
            p = predict(pipe, ctx, arm, model, t, a_start)
            for cut, evil in zip(cuts, dirty):
                prefix = t[t<=cut]
                q = predict(pipe, evil, arm, model, prefix, a_start)
                for key in p:
                    np.testing.assert_allclose(p[key][:len(prefix)],q[key],rtol=0.,atol=3e-11)
            metrics[arm] = {s:score(truth,z) for s,z in p.items()}
            states[arm] = model["gates"]
            print("SCORE",entity,arm,round(metrics[arm]["KCWA"]["RMSE"],8),flush=True)
    write(path.with_name("RESULT.json"),dict(entity=entity, N=len(t), arms=metrics, trust=masks,
          gates=states, outer=saved["outer"], poison_cuts=list(cuts)))


if __name__ == "__main__":
    command,*args = sys.argv[1:]
    {"register":register, "fit":fit, "freeze":freeze, "evaluate":evaluate}[command](*args)
