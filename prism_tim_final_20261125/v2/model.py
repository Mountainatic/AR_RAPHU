"""PRISM v2 training on the TIM legal prefixes."""

import ctypes
from dataclasses import replace
from functools import lru_cache
from itertools import product
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import core as old

np = old.np
p = old.p
g = p.g
v = p.V
s = p.s
PIPE = p.PIPE
TRUST_WINDOW = 128
HAC_CRITICAL = 1.645


def admission(blocks, delay):
    """Bartlett Newey–West on chronological, disjoint held-out loss differences."""
    loss = np.concatenate(blocks)
    n = len(loss)
    mean = float(loss.mean())
    centered = loss - mean
    lag = min(n - 1, max(delay - 1, int(4 * (n / 100)**(2 / 9))))
    variance = float(centered @ centered / n)
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / n)
    se = float(np.sqrt(max(variance, 0.) / n))
    statistic = mean / se if se else None
    means = [float(block.mean()) for block in blocks]
    wins = sum(value > 0 for value in means)
    significant = statistic > HAC_CRITICAL if statistic is not None else mean > 0
    return dict(n=n, lag=lag, mean=mean, se=se, statistic=statistic,
                block_means=means, admitted=bool(significant and 2 * wins > len(blocks)))


class EndpointBasis(g.NaturalBasis):
    def __init__(self, sample, width):
        self.mean = float(sample.mean())
        self.scale = max(float(sample.std()), 1e-9)
        z = (sample - self.mean) / self.scale
        quantiles = np.arange(1, width + 2) / (width + 2)
        quantiles[[0, -1]] = 0., 1.
        self.knots = np.unique(np.quantile(z, quantiles))
        self.center = self.columns(z).mean(axis=0)
        self.dimension = len(self.center)


def moments(signal, t, edges, basis):
    z = (signal - basis.mean) / basis.scale
    first, second = g.bucket(np.column_stack((z, z*z)), t, edges).transpose(2, 0, 1)
    sigma = np.sqrt(np.maximum(0., second - first*first))
    sigma[:, np.diff(edges) == 1] = 0.
    return first, sigma


class Surface:
    def __init__(self, signal, t, y, edges, width, weighted):
        self.basis = EndpointBasis(signal[t[0] - edges[-1] + 1:t[-1] + 1], width)
        phi = g.bucket(self.basis.transform(signal[:t[-1] + 1]), t, edges)
        self.scale = np.maximum(np.sqrt(np.mean(phi*phi, axis=(0, 1))), 1e-7)
        widths = np.diff(edges)
        measure = widths.astype(float) if weighted else np.ones(len(widths))
        design = phi / self.scale * measure[None, :, None]
        n, self.nt, self.nx = design.shape
        self.n = n
        u = PIPE.u
        self.stat = u.DesignStatistics.from_design(design.reshape(n, -1), y)
        self.linear = u.DesignStatistics.from_design(design[:, :, 0], y)
        self.curve = v.curvature(self.basis, self.scale)
        self.mass = np.diag(measure)
        centers = (np.asarray(edges[:-1]) + edges[1:]) / 2
        diff = np.diff(np.eye(len(widths)), axis=0)
        self.time = (diff.T / np.diff(centers)) @ diff if weighted else u._difference_penalty_cached(self.nt)
        self.export_scale = measure[:, None] / self.scale
        mean, sigma = moments(signal[:t[-1] + 1], t, edges, self.basis)
        lo, hi = np.quantile(mean, [.01, .99], axis=0)
        sigma_mid, sigma_hi = np.quantile(sigma, [.50, .99], axis=0)
        sigma_scale = np.maximum(sigma_hi - sigma_mid, 1e-9 * (1 + sigma_hi))
        self.support = dict(lo=lo.tolist(), hi=hi.tolist(), sigma_max=sigma_hi.tolist(),
                            sigma_scale=sigma_scale.tolist(), widths=widths.tolist())

    def solve(self, spec):
        family, _, alpha = spec
        ridge, lag, amp = np.asarray(alpha) * self.n
        u = PIPE.u
        lag_penalty = ridge * self.mass + lag * self.time
        if family == v.K_FAMILIES[0][0]:
            values, bias, _ = u._statistics_solve(self.linear, lag_penalty)
            theta = values[:, None]
        else:
            penalty = np.kron(lag_penalty, np.eye(self.nx)) + amp * np.kron(self.mass, self.curve)
            values, bias, _ = u._statistics_solve(self.stat, penalty)
            theta = values.reshape(self.nt, self.nx)
            if family.startswith("RANK_"):
                rank = min(int(family.split("_")[1]), self.nt, self.nx)
                theta, bias = self.low_rank(theta, rank, penalty)
        theta = theta * self.export_scale[:, :theta.shape[1]]
        return dict(family=family, basis=self.basis.metadata(), theta=theta.tolist(),
                    intercept=float(bias))

    def low_rank(self, theta, rank, penalty):
        u = PIPE.u
        _, _, right = np.linalg.svd(theta, full_matrices=False)
        right = right[:rank].T
        last = np.inf
        for _ in range(40):
            proj = u._u_transform(right, self.nt)
            left, _, _ = u._statistics_solve(self.stat.project(proj), proj.T @ penalty @ proj)
            left = left.reshape(rank, self.nt).T
            left, _ = np.linalg.qr(left, mode="reduced")
            proj = u._v_transform(left, self.nx)
            values, bias, _ = u._statistics_solve(self.stat.project(proj), proj.T @ penalty @ proj)
            theta = (proj @ values).reshape(self.nt, self.nx)
            objective = u._statistics_mse(self.stat, theta.ravel(), bias)
            objective += (theta.ravel() @ penalty @ theta.ravel()) / self.n
            if abs(last - objective) <= 1e-10 + 1e-6 * abs(objective):
                break
            last = objective
            _, _, right = np.linalg.svd(theta, full_matrices=False)
            right = right[:rank].T
        return theta, bias


def lag_windows(q, kind):
    length = q.L if kind == "level" else q.L - q.mature
    options = {}
    for near, base in p.BASE_WINDOWS.items():
        far = np.rint(near + (np.asarray(base[near+1:])-near)*(q.L-near)/(128-near)).astype(int)
        if near >= length:
            continue
        # A long forecast can end exactly at an original bin boundary.
        interior = tuple(int(i) for i in far if i < length)
        options[near] = (*range(near+1), *interior, length)
    if not options or any(np.any(np.diff(edges) <= 0) for edges in options.values()):
        raise ValueError("No positive-width lag bins for this protocol")
    return options


def choose_k(q, signal, y, kind, weighted):
    options = lag_windows(q, kind)
    candidates = list(product(options, p.ORIGINAL_K_GRID))
    losses = np.zeros(len(candidates))
    for first, end in s.blocks(q.first, q.parent, .48, 3):
        train = np.arange(q.first, first - q.embargo)
        score = np.arange(first, end)
        cache = {}
        for i, (near, spec) in enumerate(candidates):
            key = near, spec[1]
            edges = options[near]
            if key not in cache:
                surface = Surface(signal, train, y[train], edges, spec[1], weighted)
                phi = g.bucket(surface.basis.transform(signal[:end]), score, edges)
                cache[key] = surface, phi
            surface, phi = cache[key]
            err = y[score] - PIPE.u.predict_contract_from_design(phi, surface.solve(spec))
            losses[i] += err @ err
    best = int(np.argmin(losses))
    near, spec = candidates[best]
    edges = options[near]
    t = np.arange(q.first, q.parent)
    surface = Surface(signal, t, y[t], edges, spec[1], weighted)
    return dict(view=kind, edges=edges, contract=surface.solve(spec), support=surface.support,
                selection=dict(spec=spec, near=near, inner_sse=float(losses[best])))


def fit_views(q, a, weighted):
    # Fit the representation before the first inner K validation block.
    legal = replace(q, parent=q.parent - q.H)
    first_validation = s.blocks(q.first, legal.parent, .48, 3)[0][0]
    rep_prefix = replace(q, parent=first_validation - q.embargo)
    reps = p.representations(rep_prefix, a[:rep_prefix.parent])
    anchor, truth = p.target_arrays(q, a[:q.parent], q.parent)
    y = truth - anchor
    views = []
    for rep in reps:
        signal = p.signal(q, a[:legal.parent], rep)
        views.append([choose_k(legal, signal, y, kind, weighted) for kind, signal in
                      (("level", signal), ("delta", p.diff(q, signal)))])
    return dict(representations=reps, views=views, weighted=weighted)


def features(q, a, k, stop, use_sigma):
    t = np.arange(q.L-1, stop)
    raw, trust = [], []
    for rep, pair in zip(k["representations"], k["views"], strict=True):
        signal = p.signal(q, a[:stop], rep)
        for view in pair:
            x = p.diff(q, signal) if view["view"] == "delta" else signal
            ct, edges, support = (view[name] for name in ("contract", "edges", "support"))
            basis = g.NaturalBasis.load(ct["basis"])
            phi = g.bucket(basis.transform(x), t, edges)
            raw.append(PIPE.u.predict_contract_from_design(phi, ct))
            lo, hi = np.asarray(support["lo"]), np.asarray(support["hi"])
            span = np.maximum(hi-lo, 1e-9 * (1 + np.maximum(abs(lo), abs(hi))))
            mu, sigma = moments(x, t, edges, basis)
            error = np.maximum(lo-mu, mu-hi) / span
            if use_sigma:
                error = np.maximum(error, (sigma - support["sigma_max"]) / support["sigma_scale"])
            u = np.clip(error.max(axis=1), 0., 1.)
            trust.append((1-u)**2 * (1+2*u))
    anchor, truth = p.target_arrays(q, a, stop)
    n = len(raw)
    x, rho = np.zeros((stop, n)), np.zeros((stop, n))
    x[t], rho[t] = np.column_stack(raw), np.column_stack(trust)
    return dict(x=x, g=rho, y=truth, anchor=anchor)


@lru_cache(maxsize=1)
def group_solver():
    source = Path(__file__).with_name("group_en.c")
    with TemporaryDirectory(prefix="prism-group-en-") as scratch:
        library = Path(scratch)/"group_en.so"
        subprocess.run(["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-ffp-contract=off", "-fPIC", "-shared", str(source), "-lm",
                        "-o", str(library)], check=True)
        solve = ctypes.CDLL(str(library)).solve_groups
    fp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    solve.argtypes = (ctypes.c_size_t, fp, fp, ctypes.c_double, ctypes.c_double, ctypes.c_double, fp)
    solve.restype = ctypes.c_int
    return solve


def group_fit(x, y, alpha, ratio):
    """FISTA over the two coefficients per view with a positive ridge term."""
    n, groups, width = x.shape
    design = x.reshape(n, groups*width)
    gram = (design.T @ design) / n + alpha * (1-ratio) * np.eye(groups*width)
    xy = design.T @ y / n
    penalty = alpha * ratio
    step = 1. / np.linalg.eigvalsh(gram)[-1]
    tolerance = 2e-7 * max(penalty, np.max(abs(xy)), 1e-10)
    theta = np.empty((groups, width))
    if group_solver()(groups, gram, xy, step, penalty, tolerance, theta):
        raise RuntimeError("Group Elastic Net did not reach the requested tolerance")
    return theta

def smooth(rho):
    end = np.arange(1, len(rho) + 1)
    start = np.maximum(end - TRUST_WINDOW, 0)
    window = np.r_[0., np.cumsum(rho)]
    return np.clip((window[end] - window[start]) / (end - start), 0., 1.)


class Engine(p.c.Engine):
    def __init__(self, q, a, k, stop, use_sigma=True, use_a0=True):
        old.configure(q)
        self.pipe = PIPE
        self.ctx = features(q, a, k, stop, use_sigma)
        self.lo = q.calibration[0]
        self.y = self.ctx["y"] - self.ctx["anchor"]
        self.c_policy = "auto"
        self.q = q
        self.use_a0 = use_a0
        self.use_sigma = use_sigma
        self.w_grid = [(PIPE.w.MONOTONE, 4, penalty, sign)
                       for penalty, sign in product((.08, .3, 1.), (-1, 1))]

    def c_fit(self, t, spec):
        if spec is None or not len(t):
            return None
        x, rho, y = self.ctx["x"][t], self.ctx["g"][t], self.y[t]
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-10] = 1.
        design = np.stack((rho*(x-mean)/scale, rho), axis=2)
        theta = group_fit(design, y, spec[0] * max(float(y.std()), 1e-10), spec[1])
        if not theta.any():
            return None
        return dict(mean=mean.tolist(), scale=scale.tolist(), theta=theta.tolist(), spec=spec)

    def c_eval(self, t, model):
        if model is None:
            zero = np.zeros(len(t))
            return zero, zero, zero
        x, rho = self.ctx["x"][t], self.ctx["g"][t]
        w, bias = np.asarray(model["theta"]).T
        lengths = np.hypot(w, bias)
        out = np.sum(rho*((x-model["mean"])/model["scale"]*w+bias), axis=1)
        return out, out, np.clip(rho @ lengths / lengths.sum(), 0., 1.)

    def gate(self, end, comparison):
        diffs = []
        for first, stop in s.blocks(self.lo, end, .55, 4):
            t = np.arange(first, stop)
            parent, child = comparison(first-self.q.mature+1, t)
            diffs.append((self.y[t]-parent)**2 - (self.y[t]-child)**2)
        return admission(diffs, self.q.mature)

    @lru_cache(maxsize=128)
    def c_route(self, end):
        audit = self.gate(end, lambda fit, t: (np.zeros(len(t)), self.c_eval(t, self.c_select(fit))[0]))
        model = self.c_select(end) if audit["admitted"] else None
        return model, audit

    @lru_cache(maxsize=128)
    def w_select(self, end):
        c, _ = self.c_route(end)
        if c is None:
            return None, None
        return super().w_select(end)

    @lru_cache(maxsize=128)
    def w_route(self, end):
        def compare(fit, t):
            c, w = self.w_select(fit)
            values = self.c_eval(t, c)
            return values[0], self.w_eval(*values, w)
        audit = self.gate(end, compare)
        c, w = self.w_select(end)
        return c, w if audit["admitted"] else None, audit

    def exogenous(self, t, c, w):
        fc, z, weight = self.c_eval(t, c)
        return self.w_eval(fc, z, weight, w), weight

    @lru_cache(maxsize=128)
    def feedback_data(self, end):
        sections = s.blocks(self.lo, end, .30, 8)
        start = sections[0][0]
        residual = self.y[:end].copy()
        for first, stop in sections:
            t = np.arange(first, stop)
            fit = first-self.q.mature+1
            fc, fw, _ = self.w_route(fit)
            residual[t] -= self.exogenous(t, fc, fw)[0]
        return start, residual

    @lru_cache(maxsize=128)
    def feedback_select(self, end, fallback):
        if fallback:
            if not self.use_a0:
                return None
            start = s.blocks(self.lo, end, .30, 8)[0][0]
            y = self.y
        else:
            c, _ = self.c_route(end)
            if c is None:
                return None
            start, y = self.feedback_data(end)
        boundary = start + int(.69 * (end - start))
        train = np.arange(start + self.q.mature, boundary - self.q.mature + 1)
        test = np.arange(boundary, end)
        risks = []
        for spec in v.A_GRID:
            model = self.a_fit(y, start, train, spec)
            if model is not None:
                error = y[test] - self.a_eval(y, start, test, model)
                risks.append((float(error @ error), spec))
        if not risks:
            return None
        return self.a_fit(y, start, np.arange(start + self.q.mature, end), min(risks)[1])

    @lru_cache(maxsize=128)
    def fallback_route(self, end):
        def compare(fit, t):
            c, w, _ = self.w_route(fit)
            q, _, a0, weight = self.readouts(t, c, w, None, self.feedback_select(fit, True), int(t[0]))
            return q, q+(1-weight)*a0
        if not self.use_a0:
            return None, dict(admitted=False, disabled=True)
        audit = self.gate(end, compare)
        return self.feedback_select(end, True) if audit["admitted"] else None, audit

    def train(self, end, issue):
        c, w, _ = self.w_route(end)
        a0, a0_audit = self.fallback_route(end)
        def compare(fit, t):
            parent_c, parent_w, _ = self.w_route(fit)
            parent_a0, _ = self.fallback_route(fit)
            q, a, b, weight = self.readouts(t, parent_c, parent_w, self.feedback_select(fit, False),
                                           parent_a0, int(t[0]))
            baseline = q+(1-weight)*b
            return baseline, baseline+weight*a
        a_audit = self.gate(end, compare)
        a = self.feedback_select(end, False) if a_audit["admitted"] else None
        return dict(C=c, W=w, A=a, A0=a0, issue_from=issue, use_sigma=self.use_sigma,
                    gates=dict(C=self.c_route(end)[1], W=self.w_route(end)[2], A=a_audit, A0=a0_audit))

    def readouts(self, t, c, w, a, a0, start):
        whole = np.arange(start, int(t[-1])+1)
        q, rho = self.exogenous(whole, c, w)
        weight = smooth(rho)
        residual = self.y[:whole[-1]+1].copy()
        residual[start:] -= q
        delta = self.a_eval(residual, start, t, a)
        backup = self.a_eval(self.y, start, t, a0)
        return q[t-start], delta, backup, weight[t-start]

    def predict(self, t, model):
        q, a, a0, weight = self.readouts(t, *(model[m] for m in ("C", "W", "A", "A0")),
                                        model["issue_from"])
        return self.ctx["anchor"][t] + q + weight*a + (1-weight)*a0
