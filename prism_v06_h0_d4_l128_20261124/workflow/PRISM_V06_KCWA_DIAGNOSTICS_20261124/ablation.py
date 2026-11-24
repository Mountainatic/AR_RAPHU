"""Refit original v0.6 routing under module deletion and replay paired development rows."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "PRISM_V06_KCWA_H0_D4_L128_BASELINES_20261124"
sys.path.insert(0, str(SOURCE))
import suite as old

h, np = old.hybrid, old.np
read, write, sha = old.read, old.write, old.sha
REFITS = {"no_C": ("C",), "no_W": ("W",), "no_KCW": ("K", "C", "W")}
REG = ROOT / "REGISTRATION.json"


def register():
    record = old.registration()
    for entity in old.ENTITIES:
        old.verify_freeze(entity)
    value = dict(protocol=record["protocol"], reference=sha(SOURCE / "ALL_FROZEN.json"),
                 script=sha(Path(__file__).resolve()), deletions=REFITS,
                 fit="Refit downstream original recipes, cross-fitted parameters and routes using the original training rows; reuse identical upstream checkpoints.",
                 score="Full same-entity development rows with cold-start A; frozen cumulative stage controls; no parameter selection on development.",
                 interval="Descriptive paired circular block bootstrap, block 192, 2500 draws.")
    if REG.exists():
        assert read(REG) == h.json.loads(h.json.dumps(value))
    else:
        write(REG, value)


def bound():
    value = read(REG)
    assert value["script"] == sha(Path(__file__).resolve()) and value["reference"] == sha(SOURCE / "ALL_FROZEN.json")
    return old.registration()


class VariantCheckpoints(old.Checkpoints):
    def __init__(self, output, source, variant, keep):
        self.source = source
        self.keep = keep
        binding = dict(source.binding, ablation=variant, registration=sha(REG))
        super().__init__(output, binding, resume=output.exists())

    def load(self, role, key):
        value = super().load(role, key)
        stage = role.split("_")[1] if role.startswith(("MODEL_", "ADMISSION_")) else role.split("_")[0]
        if value is None and stage in self.keep:
            return self.source.load(role, key)
        return value


def fit(entity, variant):
    start = time.monotonic()
    record = bound()
    pipe, data, _ = old.pipeline(entity, record)
    out = ROOT / entity / variant
    pipe.root = out / "fit"
    pipe.context["resume"] = pipe.root.exists()
    keep = {"no_C": {"K"}, "no_W": {"K", "C"}, "no_KCW": set()}[variant]
    reused = pipe.checkpoints.source
    pipe.checkpoints = VariantCheckpoints(pipe.root, reused, variant, keep)
    tune_w_a = pipe.tune_residual
    removed = REFITS[variant]

    def unavailable(stage, parent, **extra):
        return pipe._unavailable_model(stage, parent, "DOMAIN_NOT_APPLICABLE",
                                       reason=f"REGISTERED_ABLATION_REMOVED_{stage}", **extra)

    if "K" in removed:
        pipe.tune_k = lambda rows, destination: unavailable("K", None)
    if "C" in removed:
        pipe.tune_c = lambda rows, parent, destination: unavailable("C", parent)
    if "W" in removed:
        pipe.tune_residual = lambda stage, rows, parent, destination: (
            unavailable("W", parent) if stage == "W" else tune_w_a(stage, rows, parent, destination))
    frame = h.frame(entity, *record["geometry"][entity]["parent"], "train", y=data[:, 0])
    model = pipe.learn("A", frame)
    nodes = chain(model)
    for stage in removed:
        assert not nodes[stage]["domain"] and nodes[stage]["routing_status"] != "ADMITTED"
    if keep:
        source = chain(read(old.path(entity, "FROZEN_MODEL.json"))["original_kcwa"])
        assert all(h.digest(nodes[k]) == h.digest(source[k]) for k in keep)
    target = out / "FROZEN_MODEL.json"
    write(target, model)
    write(out / "FIT.json", dict(sha256=sha(target), variant=variant, entity=entity,
          routes={k: v["routing_status"] for k, v in nodes.items()}, A_recipe=model.get("recipe"),
          A_gate=model.get("selection"), fit_seconds=time.monotonic()-start,
          reused_checkpoints=reused.hits))
    print("FROZEN", entity, variant, read(out / "FIT.json")["routes"], flush=True)


def chain(model):
    nodes = {}
    while model is not None:
        nodes[model["stage"]] = model
        model = model["parent"]
    return nodes


def freeze():
    bound()
    files = {}
    for entity in old.ENTITIES:
        for variant in REFITS:
            path = ROOT / entity / variant / "FROZEN_MODEL.json"
            assert sha(path) == read(path.with_name("FIT.json"))["sha256"]
            files[str(path.relative_to(ROOT))] = sha(path)
    write(ROOT / "ALL_FROZEN.json", dict(registration=sha(REG), models=files))


def prepare(entity):
    record = bound()
    old.verify_freeze(entity)
    pipe, data, accessor = old.pipeline(entity, record, development=True)
    model = read(old.path(entity, "FROZEN_MODEL.json"))["original_kcwa"]
    return record["geometry"][entity], pipe, data, accessor, model


def predictions(entity, pipe, data, model, variants, start, stop):
    frame = h.frame(entity, start, stop, "development", y=data[:, 0])
    t = frame.origin.to_numpy()
    anchor = data[t-h.D, 0]
    nodes = chain(model)
    p = {"persistence": anchor}
    for name, stage in (("K", "K"), ("KC", "C"), ("KCW", "W")):
        p[name] = anchor + pipe.emit(nodes[stage], frame.drop(columns="y_true"))
    p["KCWA"] = old.prediction(pipe, model, frame, anchor)[0]
    for name, variant in variants.items():
        p[name] = old.prediction(pipe, variant, frame, anchor)[0]
    # Diagnostic control singles out K's direct addend while preserving K
    # representations, the C/W coordinates, and the A coefficients.
    direct_k = p["K"] - anchor
    no_k = p["KCW"] - direct_k
    p["no_K_addend"] = no_k + correct(pipe, frame, model, no_k-anchor)
    return p


def correct(pipe, frame, model, upstream):
    correction = np.zeros(len(frame))
    if model["routing_status"] == "ADMITTED":
        source = h.a_state.build_a_inference_state_from_deployment_residual(
            frame, upstream, deployment_model_id=model["parent"]["partition_model_id"], h_steps=1)
        x, ready, _ = h.a_state.feature_rows(frame, source, model["profile"], pipe.A_max, model["residual_mean"])
        correction[ready] = pipe.a.predict_mature_residual_ar(x, model["contract"])
    return correction


def effect(y, full, candidate):
    rng = np.random.default_rng(20261124)
    block, n, reps = 192, len(y), 2500
    starts = rng.integers(n, size=(reps, (n+block-1)//block))
    widths = np.full(starts.shape[1], block)
    widths[-1] = n-block*(len(widths)-1)

    def risk(p):
        e = (y-p)**2
        cumulative = np.r_[0., np.cumsum(np.r_[e, e])]
        return (cumulative[starts+widths]-cumulative[starts]).sum(1)

    draws = 100*(np.sqrt(risk(candidate)/risk(full))-1)
    point = 100*(np.linalg.norm(y-candidate)/np.linalg.norm(y-full)-1)
    return dict(change_pct=float(point), CI95=np.quantile(draws, [.025, .975]).tolist())


def evaluate(entity):
    g, pipe, data, accessor, model = prepare(entity)
    seal = read(ROOT / "ALL_FROZEN.json")
    assert seal["registration"] == sha(REG)
    variants, fits = {}, {}
    for name in REFITS:
        path = ROOT / entity / name / "FROZEN_MODEL.json"
        fits[name] = read(path.with_name("FIT.json"))
        assert sha(path) == seal["models"][str(path.relative_to(ROOT))] == fits[name]["sha256"]
        variants[name] = read(path)
    start, stop = g["development"]
    t = np.arange(start, stop)
    out = predictions(entity, pipe, data, model, variants, start, stop)
    with np.load(old.path(entity, "predictions_private.npz")) as reference:
        np.testing.assert_array_equal(reference["target_index"], t)
        np.testing.assert_array_equal(reference["v06_kcwa"], out["KCWA"])
        np.testing.assert_array_equal(reference["kcw"], out["KCW"])
    np.testing.assert_array_equal(out["persistence"], data[t-h.D, 0])
    for cut in (start+140, stop-2):
        poison = data.copy()
        poison[cut-h.D+1:, 0] += 1200.
        poison[cut+1:, 1:] -= 1300.
        pipe.accessor = accessor(poison)
        trial = predictions(entity, pipe, poison, model, variants, start, cut+1)
        for name, p in out.items():
            np.testing.assert_allclose(trial[name], p[:len(trial[name])], atol=1e-10, rtol=1e-12)
    np.savez_compressed(ROOT / entity / "ablation_predictions_private.npz", target_index=t, **out)
    y = data[t, 0]
    results = {k: dict(**old.score(y, p), relative_to_KCWA=effect(y, out["KCWA"], p)) for k, p in out.items()}
    write(ROOT / entity / "ABLATION.json", dict(entity=entity, N=len(y), results=results, fits=fits,
          causal_prefix_checks=2))
    print("SCORED", entity, {k: v["RMSE"] for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    {"register": register, "fit": fit, "freeze": freeze, "evaluate": evaluate}[action](*args)
