"""Export the frozen v2 contracts into a compiler-checked FP64 model header."""

import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import model as v2


np = v2.np
legacy = v2.p.native.legacy
c_value, c_row = legacy.c_value, legacy.c_row
SOURCE = Path(__file__).with_name("stream_v2.c")


def channel_config(q, rep):
    if rep["channel"] == "joint_lift":
        transform = rep["joint_lift_pca"]
        return c_row(2, 3, *transform["mean"], *(np.asarray(transform["pc1_loading"])/transform["scale"]))
    column = q.channels.index(rep["channel"])+1
    return c_row(column, column, 0., 0., 1., 0.)


def view_config(source, view):
    contract, support, edges = (view[key] for key in ("contract", "support", "edges"))
    basis = contract["basis"]
    coef = np.asarray(contract["theta"])
    left, right = np.asarray(edges[:-1]), np.asarray(edges[1:])
    inv_width = 1. / (right-left)
    lo, hi = np.asarray(support["lo"]), np.asarray(support["hi"])
    width = coef.shape[1]
    upper, ref, inv_ref = 0., -1., 1.
    knots, inv_knots = [0.], [0.]
    if width > 1:
        quantiles = np.asarray(basis["knots"])
        upper, ref = quantiles[-1], quantiles[-2]
        active = quantiles[:width-1]
        inv_ref = 1. / (upper-ref)
        knots, inv_knots = np.r_[0., active], np.r_[0., 1./(upper-active)]
    span = np.maximum(hi-lo, 1e-9 * (1 + np.maximum(abs(lo), abs(hi))))
    return c_row(source, int(view["view"] == "delta"), len(left), width, basis["mean"],
                 1./basis["scale"], contract["intercept"], upper, ref, basis["center"][:width],
                 knots, inv_knots, inv_ref, left, right, inv_width, lo, hi, 1./span,
                 coef*inv_width[:, None], support["sigma_max"], 1./np.asarray(support["sigma_scale"]))


def head_config(model):
    config = legacy.a_config(model)
    return c_row(config["nrecent"], config["nema"], config["shift"], config["offset"],
                 config["alpha"], config["weight"], config["bias"])


def header(q, k, model):
    views = [view_config(i, view) for i, pair in enumerate(k["views"]) for view in pair]
    channels = [channel_config(q, rep) for rep in k["representations"]]
    c = model["C"]
    if c is None:
        mean, weights, bias, mass = np.zeros((4, len(views)))
    else:
        mean = c["mean"]
        w, bias = np.asarray(c["theta"]).T
        weights = w / c["scale"]
        mass = np.hypot(w, bias)
        mass /= mass.sum()
    w = legacy.w_config(v2.PIPE, model["W"])
    macros = dict(CHANNELS=len(channels), STRIDE=len(q.channels)+1, VIEWS=len(views),
                  HISTORY=q.L, PREFIX_RING=1 << q.L.bit_length(), DELAY=q.mature,
                  ANCHOR_DELAY=q.D, TRUST_WINDOW=v2.TRUST_WINDOW, USE_SIGMA=int(model["use_sigma"]),
                  W_SEGMENTS=len(w["poly"]), W_ACTIVE=int(model["W"] is not None),
                  A_ACTIVE=int(model["A"] is not None), A0_ACTIVE=int(model["A0"] is not None))
    text = [f"#define {name} {val}" for name, val in macros.items()]
    text.extend((
        "static const Channel channels[]={" + ",\n".join(channels) + "};",
        "static const View views[]={" + ",\n".join(views) + "};",
        "static const Head assisted_head=" + head_config(model["A"]) + ";",
        "static const Head fallback_head=" + head_config(model["A0"]) + ";",
    ))
    fields = dict(c_mean=mean, c_scaled_w=weights, c_bias=bias, c_mass=mass,
                  w_mean=w["mean"], w_inv_scale=w["inv_scale"], w_edges=w["edges"],
                  w_inv_span=1./np.diff(w["edges"]))
    for name, value in fields.items():
        suffix = "[]" if np.asarray(value).ndim else ""
        text.append(f"static const double {name}{suffix}={c_value(value)};")
    text.append(f"static const double w_poly[][5]={c_value(w['poly'])};")
    return "\n".join(text) + "\n"


def replay(q, a, k, model, t):
    first = model["issue_from"]
    if t[0] < first or t[-1] >= len(a) or first < q.L-1:
        raise ValueError("Replay rows are outside the model's issue interval")
    with TemporaryDirectory(prefix="prism-v2-") as scratch:
        path = Path(scratch)
        (path/"model.h").write_text(header(q, k, model), encoding="ascii")
        target = path/"prism_v2.so"
        subprocess.run(["gcc", "-O3", "-std=c11", "-mavx2", "-mfma", "-ffp-contract=off",
                        "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", "-I", str(path),
                        str(SOURCE), "-lm", "-o", str(target)], check=True)
        call = ctypes.CDLL(str(target)).replay_v2
        call.argtypes = (legacy.F64, ctypes.c_size_t, ctypes.c_size_t, legacy.F64)
        call.restype = None
        out = np.empty(int(t[-1])-first+1)
        call(a, first, len(out), out)
        return out[t-first]

