"""Reconstruct original v0.6 predictions and paired loss by interpretable components."""

import sys

from ablation import ROOT, chain, h, np, old, prepare, write


def describe(v):
    return dict(mean=float(v.mean()), rms=float(np.sqrt(np.mean(v**2))), mean_abs=float(np.abs(v).mean()))


def pieces(parts, total):
    recon = sum(parts.values(), np.zeros(len(total)))
    np.testing.assert_allclose(recon, total, atol=2e-12, rtol=1e-10)
    return float(np.max(np.abs(recon-total)))


def k_responses(pipe, nodes, frame, fit_frame):
    responses = []
    root = nodes["K"]
    for rep in root["representations"]:
        contract = rep["contract"]
        x, intervals = pipe._values(frame, rep)
        phi = pipe.u.basis_from_metadata(contract["basis"]).transform(x)
        theta = np.asarray(contract["theta"])
        terms = np.einsum("tbk,bk->tb", phi[:, :, :theta.shape[1]], theta)
        intercept = float(contract["intercept"])
        response = pipe.u.predict_contract(x, contract)
        np.testing.assert_allclose(intercept+terms.sum(1), response, atol=2e-12, rtol=1e-10)
        train, _ = pipe._values(fit_frame, rep)
        train_min, train_max = train.min(0), train.max(0)
        outside = np.mean((x < train_min) | (x > train_max), axis=0)
        limits = np.quantile(x, [.05,.95], axis=0)
        grid = np.linspace(limits[0], limits[1], 41)
        design = pipe.u.basis_from_metadata(contract["basis"]).transform(grid)
        curve = np.einsum("tbk,bk->tb", design[:, :, :theta.shape[1]], theta)
        responses.append(dict(channel=rep["channel"], family=rep["family"], profile=rep["profile"],
             is_reference=rep["channel"] == root["reference_channel"], intercept=intercept,
             response=describe(response), training_response=describe(pipe.u.predict_contract(train,contract)),
             development_bins=[dict(near=n, far=f, response=describe(terms[:, j]),
             training_range=[float(train_min[j]),float(train_max[j])], dev_fraction_outside_train_range=float(outside[j]),
             amplitude_grid=grid[:, j].tolist(), fitted_response=curve[:, j].tolist())
             for j, (n,f) in enumerate(intervals)]))
    return responses


def c_terms(pipe, nodes, frame, total):
    c, k = nodes["C"], nodes["K"]
    if c["routing_status"] != "ADMITTED":
        np.testing.assert_array_equal(total, np.zeros(len(frame)))
        return {}, 0.
    p, raw, ready = pipe._features(k, frame)
    if c["recipe"] == "ADDITIVE_COMPRESSED":
        x = p
        widths = [1]*len(k["representations"])
    else:
        x = np.concatenate([r[:, columns] for r, columns in zip(raw, c["joint_columns"], strict=True)], axis=1)
        widths = [len(columns) for columns in c["joint_columns"]]
    ct = c["contract"]
    terms = np.zeros_like(x)
    terms[ready] = (x[ready]-ct["mean"]) / ct["scale"] * ct["coefficient"]
    parts = {"intercept": np.where(ready, ct["intercept"], 0.)}
    for rep, values in zip(k["representations"], np.split(terms, np.cumsum(widths)[:-1], axis=1), strict=True):
        parts[rep["channel"]] = values.sum(1)
    return parts, pieces(parts, total)


def a_terms(pipe, nodes, frame, upstream, total, y, full):
    model = nodes["A"]
    if model["routing_status"] != "ADMITTED":
        np.testing.assert_array_equal(total, np.zeros(len(frame)))
        return dict(route=model["routing_status"], ready=0, groups={}, lags=[], intercept=None)
    source = h.a_state.build_a_inference_state_from_deployment_residual(
        frame, upstream, deployment_model_id=nodes["W"]["partition_model_id"], h_steps=1)
    x, ready, _ = h.a_state.feature_rows(frame, source, model["profile"], pipe.A_max, model["residual_mean"])
    c = model["contract"]
    terms = np.zeros((len(total), len(c["coefficient"])))
    terms[ready] = (x-c["mean"]) / c["scale"] * c["coefficient"]
    intercept = np.where(ready, c["intercept"], 0.)
    reconstructed = intercept + terms.sum(1)
    np.testing.assert_allclose(reconstructed, total, atol=2e-12, rtol=1e-10)
    loss_factor = 2*(y-full+total)-total
    lags = h.D + h.a_state.offsets(model["profile"], pipe.A_max)

    def entry(part):
        return dict(**describe(part), mean_MSE_reduction=float(np.mean(loss_factor*part)))

    groups = {}
    for lo, hi in ((4,12), (12,20), (20,36), (36,68), (68,132)):
        take = (lo <= lags) & (lags < hi)
        if take.any():
            selected = lags[take]
            groups[f"{selected[0]}-{selected[-1]}"] = entry(terms[:, take].sum(1))
    attributed = entry(intercept)["mean_MSE_reduction"] + sum(v["mean_MSE_reduction"] for v in groups.values())
    actual = np.mean((y-full+total)**2 - (y-full)**2)
    np.testing.assert_allclose(attributed, actual, atol=2e-12, rtol=1e-10)
    return dict(route=model["routing_status"], ready=int(ready.sum()), groups=groups,
                lags=[dict(lag=int(lag), coefficient_per_residual_unit=float(c["coefficient"][j]/c["scale"][j]),
                           **entry(terms[:, j])) for j, lag in enumerate(lags)],
                intercept=entry(intercept), reconstruction_max_abs=float(np.max(np.abs(reconstructed-total))))


def w_curve(pipe, nodes, frame):
    w = nodes["W"]
    if w["routing_status"] != "ADMITTED":
        return dict(route=w["routing_status"])
    q, ready, detail = pipe.response_coordinate(nodes["C"], frame)
    x = np.linspace(*np.quantile(q[ready], [.05,.95]), 101)
    y = pipe.w.predict_w_correction(x, w["contract"])
    return dict(route=w["routing_status"], recipe=w["recipe"], coordinate=detail["interface"],
                q_grid=x.tolist(), correction=y.tolist(), response_range=[float(y.min()), float(y.max())])


def run(entity):
    geo, pipe, data, _, model = prepare(entity)
    nodes = chain(model)
    start, stop = geo["development"]
    frame = h.frame(entity, start, stop, "development", y=data[:, 0])
    x = frame.drop(columns="y_true")
    fit_frame = h.frame(entity, *geo["parent"], "train")
    t = frame.origin.to_numpy()
    y, anchor = data[t,0], data[t-h.D,0]
    k, c, w = (pipe.emit(nodes[name], x) for name in ("K", "C", "W"))
    full, _, _ = old.prediction(pipe, model, frame, anchor)
    with np.load(old.path(entity, "predictions_private.npz")) as frozen:
        np.testing.assert_array_equal(frozen["target_index"], t)
        np.testing.assert_array_equal(full, frozen["v06_kcwa"])
    terms = dict(K=k, C=c-k, W=w-c, A=full-(anchor+w))
    reconstruction = pieces({"anchor":anchor, **terms}, full)
    c_parts, c_error = c_terms(pipe, nodes, x, terms["C"])
    train_c = pipe.emit(nodes["C"], fit_frame)-pipe.emit(nodes["K"], fit_frame)
    train_c_parts, _ = c_terms(pipe,nodes,fit_frame,train_c)
    error = y-anchor
    adjustment = full-anchor
    attribution = {k: float(np.mean((2*error-adjustment)*v)) for k,v in terms.items()}
    risk = np.mean(error**2)-(np.mean((y-full)**2))
    np.testing.assert_allclose(sum(attribution.values()), risk, atol=2e-12, rtol=1e-10)
    means = {k: dict(route=nodes[k]["routing_status"], **describe(v), mean_MSE_reduction=attribution[k]) for k,v in terms.items()}
    a = a_terms(pipe, nodes, frame, w, terms["A"], y, full)
    response = dict(entity=entity, N=len(t), R2_level=old.score(y,full)["R2"],
                    R2_increment=old.score(error,adjustment)["R2"],
                    persistence_MSE_skill=float(risk/np.mean(error**2)),
                    reconstruction_max_abs=reconstruction, stages=means,
                    C_channels={k:describe(v) for k,v in c_parts.items()}, C_reconstruction_max_abs=c_error,
                    C_training_channels={k:describe(v) for k,v in train_c_parts.items()},
                    K_reference=nodes["K"]["reference_channel"], K_channels=k_responses(pipe,nodes,x,fit_frame),
                    W=w_curve(pipe,nodes,x), A=a,
                    scope="Exact prediction and paired squared-error decomposition. Response curves describe the fitted model on observed amplitude ranges.")
    write(ROOT/entity/"EXPLANATION.json",response)
    print("EXPLAINED",entity,means,flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
