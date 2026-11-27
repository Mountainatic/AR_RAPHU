"""Support-gated Laguerre ECM prototype. Arrays use issue order and FP64."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from scipy.signal import lfilter
from scipy.stats import norm


@dataclass(frozen=True)
class Dynamics:
    near: int
    banks: tuple

    def batch(self, u):
        """Newest-first exact lags followed by orthogonal Laguerre banks."""
        n = len(u)
        padded = np.pad(u, ((self.near, 0),) + ((0, 0),) * (u.ndim - 1))
        columns = [padded[self.near-j:self.near-j+n] for j in range(self.near)]
        delayed = padded[:n]
        for pole, order in self.banks:
            state = lfilter([np.sqrt(1-pole*pole)], [1, -pole], delayed, axis=0)
            for j in range(order):
                columns.append(state)
                if j+1 < order:
                    state = lfilter([-pole, 1], [1, -pole], state, axis=0)
        return np.stack(columns, axis=-1)

    def stream(self, shape):
        return LaguerreStream(self, shape)


class LaguerreStream:
    def __init__(self, dynamics, shape):
        self.ring = np.zeros((dynamics.near, *shape))
        self.at = 0
        self.banks = []
        for pole, order in dynamics.banks:
            row, col = np.indices((order, order))
            lower = row > col
            a = np.eye(order) * pole
            a[lower] = (1-pole*pole) * (-pole)**(row[lower]-col[lower]-1)
            b = np.sqrt(1-pole*pole) * (-pole)**np.arange(order)
            self.banks.append((a, b, np.zeros((order, *shape))))

    def step(self, u):
        delayed = self.ring[self.at].copy()
        self.ring[self.at] = u
        near = self.ring[(self.at-np.arange(len(self.ring))) % len(self.ring)]
        self.at = (self.at+1) % len(self.ring)
        states = [near]
        for a, b, state in self.banks:
            state[:] = np.tensordot(a, state, axes=1) + np.multiply.outer(b, delayed)
            states.append(state)
        return np.moveaxis(np.concatenate(states), 0, -1)


class Basis:
    """Natural spline with an exact linear column and orthogonal curvature columns."""

    def __init__(self, sample):
        self.center = sample.mean()
        self.scale = max(sample.std(), 1e-12)
        z = self.standardize(sample)
        self.knots = np.unique(np.quantile(z, [.05, .35, .65, .95]))
        self.spline = None
        self.rotation = np.empty((0, 0))
        self.curvature = np.zeros((3, 3))
        self.limit = max(12., np.max(np.abs(z))+1)
        if len(self.knots) < 3:
            return
        self.spline = CubicSpline(self.knots, np.eye(len(self.knots)), bc_type="natural")
        linear = np.column_stack((np.ones(len(z)), z))
        raw = self.raw(z)
        self.project = np.linalg.lstsq(linear, raw, rcond=None)[0]
        residual = raw-linear@self.project
        _, singular, vt = np.linalg.svd(residual, full_matrices=False)
        keep = singular[:2] > singular[0]*1e-10 + 1e-12
        self.rotation = vt[:2][keep].T * (np.sqrt(len(z))/singular[:2][keep])
        lo, hi = self.knots[:-1], self.knots[1:]
        # Two-point Gauss integration is exact for the product of linear second derivatives.
        for sign in (-1, 1):
            nodes = (lo+hi)/2 + sign*(hi-lo)/(2*np.sqrt(3))
            curvature = self.spline(nodes, 2) @ self.rotation
            k = self.rotation.shape[1]
            self.curvature[1:1+k, 1:1+k] += (curvature.T*((hi-lo)/2)) @ curvature

    def standardize(self, x):
        with np.errstate(over="ignore"):
            return np.clip((x-self.center)/self.scale, -1e6, 1e6)

    def raw(self, z):
        bound = z.clip(self.knots[0], self.knots[-1])
        return self.spline(bound)+(z-bound)[..., None]*self.spline(bound, 1)

    def transform(self, z):
        # Smooth saturation far beyond training support keeps even stored OOD states bounded.
        excess = np.maximum(abs(z)-self.limit, 0)
        z = np.sign(z)*(np.minimum(abs(z), self.limit) + excess/(1+excess))
        out = np.zeros((*z.shape, 3))
        out[..., 0] = z
        if self.rotation.size:
            nonlinear = (self.raw(z)-self.project[0]-z[..., None]*self.project[1]) @ self.rotation
            out[..., 1:1+self.rotation.shape[1]] = nonlinear
        return out


def moments(z, poles):
    result = []
    for a in poles:
        mu = lfilter([1-a], [1, -a], z, axis=0)
        previous = np.vstack((np.zeros_like(z[:1]), mu[:-1]))
        var = lfilter([a*(1-a)], [1, -a], (z-previous)**2, axis=0)
        result.append(np.stack((mu, np.sqrt(var)), axis=-1))
    return np.stack(result, axis=-2)


def hermite(distance):
    v = np.clip(distance, 0, 1)
    return (1-v)**2 * (1+2*v)


@dataclass(frozen=True)
class Support:
    poles: tuple
    lower: np.ndarray
    upper: np.ndarray
    sigma: np.ndarray
    padding: tuple
    width: float

    @classmethod
    def fit(cls, values, poles, padding, width):
        lo, hi = np.quantile(values[..., 0], [.01, .99], axis=0)
        return cls(poles, lo, hi, np.quantile(values[..., 1], .99, axis=0), padding, width)

    def trust(self, values):
        mu, sigma = values[..., 0], values[..., 1]
        # Use the largest range deviation for each mean; multiply smooth one-sided tapers.
        mean_pad, sigma_pad = self.padding
        limits = self.sigma + sigma_pad
        width = self.width * np.maximum(self.sigma, .25)
        return np.prod(hermite((self.lower-mean_pad-mu)/width)
                       * hermite((mu-self.upper-mean_pad)/width)
                       * hermite((sigma-limits)/width), axis=-1)


@dataclass
class Design:
    raw: np.ndarray
    trust: np.ndarray
    curvature: np.ndarray

    def weighted(self):
        return (self.raw*self.trust[..., None, None]).reshape(len(self.trust), -1)


@dataclass
class ExogenousMap:
    basis: list
    support: Support
    dynamics: Dynamics

    @classmethod
    def fit(cls, views, fit, dynamics, padding, width, poles=(.50, .97)):
        bases = [Basis(x[fit]) for x in views.T]
        z = np.column_stack([basis.standardize(x) for basis, x in zip(bases, views.T)])
        support = Support.fit(moments(z, poles)[fit], poles, padding, width)
        return cls(bases, support, dynamics)

    def design(self, views):
        z = np.column_stack([basis.standardize(x) for basis, x in zip(self.basis, views.T)])
        point = np.stack([basis.transform(x) for basis, x in zip(self.basis, z.T)], axis=1)
        raw = np.swapaxes(self.dynamics.batch(point), -2, -1)
        return Design(raw, self.support.trust(moments(z, self.support.poles)),
                      np.stack([basis.curvature for basis in self.basis]))

    def stream(self):
        return ExogenousStream(self)


class ExogenousStream:
    def __init__(self, mapping):
        self.mapping = mapping
        self.dynamic = mapping.dynamics.stream((len(mapping.basis), 3))
        self.mu = np.zeros((len(mapping.basis), len(mapping.support.poles)))
        self.var = np.zeros_like(self.mu)

    def step(self, views):
        z = np.array([basis.standardize(x) for basis, x in zip(self.mapping.basis, views)])
        point = np.stack([basis.transform(x) for basis, x in zip(self.mapping.basis, z)])
        a = np.asarray(self.mapping.support.poles)
        deviation = z[:, None]-self.mu
        self.var[:] = a*(self.var+(1-a)*deviation**2)
        self.mu[:] += (1-a)*deviation
        trust = self.mapping.support.trust(np.stack((self.mu, np.sqrt(self.var)), axis=-1))
        return self.dynamic.step(point).swapaxes(-2, -1), trust


def ridge(x, y, penalty):
    gram = (x.T@x)/len(y)
    if np.ndim(penalty) == 1:
        gram.flat[::len(gram)+1] += penalty
    else:
        gram += penalty
    return solve(gram, (x.T@y)/len(y), assume_a="pos")


def scale_columns(x):
    return np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-7)


def shrink(groups, threshold):
    norms = np.linalg.norm(groups, axis=1)
    return groups*np.maximum(0, 1-threshold/np.maximum(norms, 1e-300))[:, None]


def screen(x, y, group_shape, penalty, fraction, nonlinear_fraction):
    """FISTA with nested group prox and a checked convex stationarity residual."""
    n, p = x.shape
    count, taps, width = group_shape
    gram = x.T@x/n + penalty
    target = x.T@y/n
    group_gradient = target[:count*taps*width].reshape(count, taps*width)
    strength = fraction*np.linalg.norm(group_gradient, axis=1).max()
    nonlinear = strength*nonlinear_fraction
    stop = group_gradient.size
    value = previous = np.zeros(p)
    # Symmetric PSD Gram. Backtracking corrects the power-iteration estimate.
    axis = np.random.default_rng(843).standard_normal(p)
    for _ in range(24):
        axis = gram@axis
        axis /= max(np.linalg.norm(axis), 1e-30)
    step = .95/max(float(axis@(gram@axis)), 1e-8)
    def proximal(v):
        block = v[:stop].reshape(count, taps, width)
        curved = block[..., 1:].reshape(count, -1)
        block[..., 1:] = shrink(curved, step*nonlinear).reshape(count, taps, width-1)
        v[:stop] = shrink(block.reshape(count, -1), step*strength).ravel()
        return v

    speed = 1.
    scale = max(np.linalg.norm(target, ord=np.inf), 1e-12)
    for iteration in range(1, 4501):
        gradient = gram@value-target
        candidate = proximal(value-step*gradient)
        if iteration == 1:
            diff = candidate-value
            while diff@(gram@diff) > (diff@diff)/step+1e-16:
                step *= .7
                candidate = proximal(value-step*gradient)
                diff = candidate-value
        if iteration % 25 == 0:
            prox_error = candidate-proximal(candidate-step*(gram@candidate-target))
            kkt = np.linalg.norm(prox_error, ord=np.inf)/(step*scale)
            if kkt <= 3e-5:
                return candidate[:stop].reshape(group_shape), dict(
                    group_penalty=strength, nonlinear_group_penalty=nonlinear,
                    iterations=iteration, relative_proximal_kkt=float(kkt))
        next_speed = (1+np.sqrt(1+4*speed**2))/2
        if (value-candidate)@(candidate-previous) > 0:
            next_speed = 1.
        value = candidate + (speed-1)/next_speed*(candidate-previous) if next_speed > 1 else candidate
        previous, speed = candidate, next_speed
    raise RuntimeError(f"Group screening KKT residual {kkt:.3g}")


@dataclass
class Joint:
    theta: np.ndarray
    adjustment: np.ndarray
    weights: np.ndarray
    audit: dict

    def predict(self, design, endo):
        rho = design.trust@self.weights
        output = np.einsum("igtp,gtp,ig->i", design.raw, self.theta, design.trust, optimize=True)
        return output + rho*(endo@self.adjustment), rho


def fit_joint(design, endo, delta, rows, fraction, curve, linear_ridge, relax_ridge):
    shape = design.raw.shape[1:]
    count, taps, width = shape
    flat = design.weighted()
    x = flat[rows]
    scale = scale_columns(x).reshape(shape)
    e = endo[rows]
    se = scale_columns(e)
    s = scale.ravel()
    pooled = design.trust[rows].mean(axis=1)
    # Align grouped proximal coordinates with sample RMS. The curvature matrix
    # is expressed in the original spline coefficients before diagonal rescaling.
    penalty = np.zeros((x.shape[1], x.shape[1]))
    for group, k in enumerate(design.curvature):
        for tap in range(taps):
            i = (group*taps+tap)*width
            d = np.maximum(scale[group, tap], np.sqrt(.02))
            penalty[i:i+width, i:i+width] = curve*k/(d[:, None]*d)
    penalty.flat[::len(penalty)+1] += linear_ridge
    nonlinear_columns = np.arange(len(s)) % width != 0
    penalty[np.flatnonzero(nonlinear_columns), np.flatnonzero(nonlinear_columns)] += curve*.1
    # First-stage adjustment uses pooled trust. The active-group weighted trust
    # is computed once from the screening coefficients, then held fixed in refit.
    theta, audit = screen(np.column_stack((x/s, pooled[:, None]*e/se)), delta[rows],
                          shape, _augment(penalty, len(se), linear_ridge), fraction, .55)
    weights = np.linalg.norm((theta/scale).reshape(count, -1), axis=1)
    active = weights > 0
    if not active.any():
        return Joint(np.zeros(shape), np.zeros(endo.shape[1]), weights,
                     dict(audit, active=[]))
    weights /= weights.sum()
    rho = design.trust[rows]@weights
    cols = np.repeat(active, taps*width)
    relaxed = penalty[np.ix_(cols, cols)]
    relaxed.flat[::len(relaxed)+1] += relax_ridge-linear_ridge
    matrix = np.column_stack((x[:, cols]/s[cols], rho[:, None]*e/se))
    fit = ridge(matrix, delta[rows], _augment(relaxed, len(se), relax_ridge))
    full = np.zeros(len(s))
    full[cols] = fit[:int(cols.sum())]/s[cols]
    return Joint(full.reshape(shape), fit[int(cols.sum()):]/se, weights,
                 dict(audit, active=np.flatnonzero(active).tolist()))


def _augment(penalty, extra, diagonal):
    n = len(penalty)
    result = np.zeros((n+extra, n+extra))
    result[:n, :n] = penalty
    result[n:, n:] = diagonal*np.eye(extra)
    return result


def admission(blocks, mature):
    """One-sided Bartlett HAC on held-out loss differences, plus majority wins."""
    loss = np.concatenate(blocks)
    centered = loss-loss.mean()
    n = len(loss)
    lag = min(n-1, max(mature-1, int(4*(n/100)**(2/9))))
    variance = centered@centered
    for k in range(1, lag+1):
        variance += 2*(1-k/(lag+1))*(centered[k:]@centered[:-k])
    se = np.sqrt(max(variance, 0))/n
    mean = float(loss.mean())
    p = float(norm.sf(mean/se)) if se > 0 else float(mean <= 0)
    means = [float(block.mean()) for block in blocks]
    return dict(n=n, lag=lag, mean=mean, se=float(se), p=p, block_means=means,
                admitted=bool(p < .05 and 2*sum(m > 0 for m in means) > len(means)))



def aligned_deployment(anchor, endo, exogenous, active_trust, innovation=0., *,
                       exo_admitted=True, innovation_admitted=False):
    """Shared law for batch and online predictions in the aligned ECM."""
    if not exo_admitted:
        return anchor+endo
    rho = np.clip(active_trust,0.,1.)
    predict = anchor+exogenous+(1.-rho)*endo
    if innovation_admitted:
        predict = predict+rho*innovation
    return predict


def aligned_exogenous_target(increment, endo, active_trust):
    """Training target conditional on the fixed stage-one trust weights."""
    return increment-(1.-np.clip(active_trust,0.,1.))*endo


def box_jenkins_deployment(safe, process_location, gated_process, active_trust,
                           disturbance=0., *, exo_admitted=True, inn_admitted=False):
    """Mix the safe baseline with support-gated X-only process predictions."""
    if not exo_admitted:
        return safe
    trust = np.clip(active_trust, 0., 1.)
    location = process_location + disturbance if inn_admitted else process_location
    return (1.-trust)*safe + trust*location + gated_process

