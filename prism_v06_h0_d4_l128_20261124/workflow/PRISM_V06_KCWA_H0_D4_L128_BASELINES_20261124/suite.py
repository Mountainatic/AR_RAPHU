"""Extend the matched KCW experiment with the registered October v0.6 A stage."""

import csv
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "PRISM_V06_KCW_QC32_H0_D4_L128_SUITE_20261124"
sys.path[:0] = [str(SOURCE), str(SOURCE / "baselines")]

import run as hybrid
import sru
import protocol as baseline
from canonical.cz_crossrod_l128_20261008.checkpoint import Checkpoints

np = hybrid.np
read, write, sha = hybrid.read, hybrid.write, hybrid.sha
ENTITIES = baseline.RODS
BASELINES = ("Persistence", "AR", "ARX", *baseline.NAMES)
ARMS = ("v06_kcwa", "kcw", "kcw_qc32", "kcw_fixed_a", "persistence_qc32")


def path(entity, name):
    return ROOT / entity / name


def register():
    common = read(SOURCE / "REGISTRATION.json")
    sru_reg = sru.bound_registration()
    paths = [Path(__file__).resolve(), SOURCE / "REGISTRATION.json", sru.REG,
             SOURCE / "SUITE_RESULTS.json", baseline.ROOT / "COMPLETE.json", baseline.ROOT / "ALL_FROZEN.json"]
    for entity in ENTITIES:
        paths.extend(SOURCE / entity / name for name in ("FIT.json", "FROZEN_MODEL.json"))
        assert sha(SOURCE / entity / "FROZEN_MODEL.json") == read(SOURCE / entity / "FIT.json")["model_sha256"]
    spec = dict(name="October v0.6 KCWA, H=0,D=4,L=128", protocol=common["protocol"],
                geometry={**common["data"], sru.ENTITY: sru_reg["data"]},
                sources={str(p): sha(p) for p in paths},
                A="Original October A registry, parameter cross-fitting, outer paired admission and empty residual state at the score boundary.",
                KCW="Read-only, hash-validated reuse of the exact October KCW fits and internal fold checkpoints from the matched experiment.",
                matched_fixed_A="Separate fixed-coefficient calibration control, with the same later calibration rows as QC32.",
                baselines="Replay all eight frozen matched baselines on the identical development row indices.")
    target = ROOT / "REGISTRATION.json"
    if target.exists():
        assert read(target) == spec
    else:
        write(target, spec)


def registration():
    record = read(ROOT / "REGISTRATION.json")
    assert all(sha(Path(p)) == digest for p, digest in record["sources"].items())
    common = read(SOURCE / "REGISTRATION.json")
    assert all(sha(Path(p)) == digest for p, digest in common["sources"].items())
    return record


class ReadThroughCheckpoints(Checkpoints):
    def __init__(self, output, source):
        super().__init__(output, source.binding, resume=output.exists())
        self.source = source

    def load(self, role, key):
        stored = super().load(role, key)
        if stored is None and not role.endswith("_A") and not role.startswith("A_"):
            return self.source.load(role, key)
        return stored


def pipeline(entity, record, *, development=False):
    if entity == sru.ENTITY:
        sr = sru.bound_registration()
        pipe = sru.SRUPipeline(sru.values(sr), sr)
        data = sru.values(sr, development=development)
        make_accessor = sru.SRUAccessor
    else:
        pipe = hybrid.OctoberPipeline(entity, hybrid.values(entity))
        data = hybrid.values(entity, evaluate=development)

        def make_accessor(values):
            return hybrid.CurrentProcessAccessor(entity, values)

    assert record["protocol"] == dict(H=0, D=4, L=128)
    out = path(entity, "october_fit")
    pipe.root = out
    pipe.context["resume"] = out.exists()
    pipe.checkpoints = ReadThroughCheckpoints(out, pipe.checkpoints)
    pipe.a_input_cache = {}
    if development:
        pipe.accessor = make_accessor(data)
    return pipe, data, make_accessor


def fit(entity):
    started = time.monotonic()
    record = registration()
    pipe, data, _ = pipeline(entity, record)
    original = read(SOURCE / entity / "FROZEN_MODEL.json")
    geo = record["geometry"][entity]
    frame = hybrid.frame(entity, *geo["parent"], "train", y=data[:, 0])
    model = pipe.learn("A", frame)
    assert hybrid.digest(model["parent"]) == hybrid.digest(original["kcw"])
    f, _, q = hybrid.upstream(entity, data, pipe, model["parent"], *geo["calibration"], "calibration")
    fixed = hybrid.select_fixed_model(pipe, hybrid.stream(f, data, q, "CROSS_FITTED_PREQUENTIAL"))
    if "fixed_a" in original:
        assert hybrid.digest(fixed) == hybrid.digest(original["fixed_a"])
    target = path(entity, "FROZEN_MODEL.json")
    write(target, dict(original_kcwa=model, matched_fixed_a=fixed))
    result = dict(entity=entity, model_sha256=sha(target), training_rows=geo["parent"],
                  kcwa_source="10 月 v0.6 逐层实现 6ba6f1b", original_kcw_exact=True,
                  A_candidates=len(pipe.A_specs), A_profiles=[list(p) for p in pipe.A_profiles],
                  route=model["routing_status"], recipe=model.get("recipe"),
                  selection=model.get("selection"), A_final_fit_rows=model.get("final_fit_rows"),
                  matched_fixed_a_profile=fixed["profile"], reused_checkpoints=pipe.checkpoints.source.hits,
                  elapsed_seconds=time.monotonic()-started)
    write(path(entity, "FIT.json"), result)
    print("FROZEN", entity, result["route"], result["recipe"], flush=True)


def freeze():
    registration()
    files = {}
    for entity in ENTITIES:
        m, f = path(entity, "FROZEN_MODEL.json"), path(entity, "FIT.json")
        assert sha(m) == read(f)["model_sha256"]
        files[entity] = {"model": sha(m), "fit": sha(f)}
    write(ROOT / "ALL_FROZEN.json", dict(registration=sha(ROOT / "REGISTRATION.json"), files=files))


def verify_freeze(entity):
    seal = read(ROOT / "ALL_FROZEN.json")
    assert sha(ROOT / "REGISTRATION.json") == seal["registration"]
    assert sha(path(entity, "FROZEN_MODEL.json")) == seal["files"][entity]["model"]
    assert sha(path(entity, "FIT.json")) == seal["files"][entity]["fit"]


def prediction(pipe, model, frame, anchor):
    """Cold-start block replay with the registered mature-residual index rule."""
    no_y = frame.drop(columns="y_true")
    parent = pipe.emit(model["parent"], no_y)
    if model["routing_status"] != "ADMITTED":
        return parent + anchor, parent + anchor, np.zeros(len(frame), dtype=bool)
    source = hybrid.a_state.build_a_inference_state_from_deployment_residual(
        frame, parent, deployment_model_id=model["parent"]["partition_model_id"], h_steps=1)
    child, native, _ = pipe.predict(model, no_y, a_state=source)
    latency = hybrid.D + max(hybrid.a_state.offsets(model["profile"], pipe.A_max))
    np.testing.assert_array_equal(native, np.arange(len(frame)) >= latency)
    return child + anchor, parent + anchor, native


def fixed_prediction(entity, pipe, data, models, start, stop):
    model, fixed = models["original_kcwa"], models["matched_fixed_a"]
    warm = hybrid.D + max(hybrid.a_state.offsets(fixed["profile"], pipe.A_max))
    f, _, q = hybrid.upstream(entity, data, pipe, model["parent"], start-warm, stop, "replay")
    g = hybrid.stream(f, data, q, "DEPLOYED_W_ONLINE")
    return q[warm:] + hybrid.fixed_prediction(pipe, g.iloc[warm:], g.iloc[:warm], fixed)


def evaluate(entity):
    rec = registration()
    verify_freeze(entity)
    pipe, data, accessor = pipeline(entity, rec, development=True)
    models = read(path(entity, "FROZEN_MODEL.json"))
    model = models["original_kcwa"]
    start, stop = rec["geometry"][entity]["development"]
    origins = np.arange(start, stop)
    f = hybrid.frame(entity, start, stop, "development", y=data[:, 0])
    p, kcw, native = prediction(pipe, model, f, data[origins-hybrid.D, 0])
    fixed = fixed_prediction(entity, pipe, data, models, start, stop)
    reference_rows = 192
    if model["routing_status"] == "ADMITTED":
        ref, mask, _ = pipe.online_a(model, f.iloc[:reference_rows])
        np.testing.assert_array_equal(mask, native[:reference_rows])
        np.testing.assert_allclose(ref + data[origins[:reference_rows]-hybrid.D, 0],
                                   p[:reference_rows], rtol=1e-12, atol=1e-10)
    for cut in (start, start+4, start+140, stop-2):
        poison = data.copy()
        poison[cut-hybrid.D+1:, 0] += 1200.
        poison[cut+1:, 1:] -= 1300.
        pipe.accessor = accessor(poison)
        ev = hybrid.frame(entity, start, cut+1, "development", y=poison[:, 0])
        trial, _, _ = prediction(pipe, model, ev, poison[origins[:len(ev)]-hybrid.D, 0])
        np.testing.assert_allclose(trial, p[:len(ev)], rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(fixed_prediction(entity, pipe, poison, models, start, cut+1),
                                   fixed[:len(ev)], rtol=1e-12, atol=1e-10)
    name = "predictions.npz" if entity == sru.ENTITY else "predictions_private.npz"
    with np.load(SOURCE / entity / name) as archive:
        np.testing.assert_array_equal(archive["target_index"], origins)
        np.testing.assert_array_equal(archive["kcw"], kcw)
        if "kcw_fixed_a" in archive:
            np.testing.assert_array_equal(archive["kcw_fixed_a"], fixed)
        arms = dict(v06_kcwa=p, kcw=kcw, kcw_fixed_a=fixed,
                    kcw_qc32=archive["kcw_qc32"], persistence_qc32=archive["persistence_qc32"])
    np.savez_compressed(path(entity, "predictions_private.npz"), target_index=origins, **arms)
    result = dict(causal_prefix_checks=4, corrected_rows=int(native.sum()),
                  canonical_online_replay_rows=reference_rows if native.any() else 0,
                  development=[start, stop])
    write(path(entity, "EVALUATION.json"), result)
    print("SCORED", entity, {k: score(data[origins, 0], v)["RMSE"] for k, v in arms.items()}, flush=True)


def replay_baseline(mode, entity):
    verify_freeze(entity)
    baseline.ensure_bound("neural.py" if mode == "neural" else "cpu.py")
    reference = baseline.ROOT / entity
    output = ROOT / "baseline_replay" / entity
    output.mkdir(exist_ok=True, parents=True)
    if mode == "neural":
        sys.path.insert(0, str(baseline.model_source(entity)))
        import baseline_models
        assert Path(baseline_models.__file__).resolve() == (baseline.model_source(entity) / "baseline_models.py").resolve()
        links = ("neural",)
        function = runpy.run_path(str(baseline.REFERENCE / "neural.py"))["eval_models"]
    else:
        assert mode == "linear"
        links = ("linear.json", "linear.pkl")
        function = runpy.run_path(str(baseline.REFERENCE / "cpu.py"))["evaluate_linear"]
    for name in links:
        link = output / name
        if not link.exists():
            link.symlink_to(reference / name, target_is_directory=name == "neural")
        assert link.resolve() == (reference / name).resolve()
    function.__globals__["ROOT"] = output.parent
    function(entity)
    archive = f"{mode}_evaluation.npz"
    with np.load(output / archive) as fresh, np.load(reference / archive) as prior:
        assert fresh.files == prior.files
        for key in fresh.files:
            np.testing.assert_array_equal(fresh[key], prior[key])
    print("BASELINES_REPLAYED", entity, mode, flush=True)

def score(y, p):
    assert p.shape == y.shape and np.isfinite(p).all()
    e = y-p
    mse = np.mean(e**2)
    return dict(RMSE=float(np.sqrt(mse)), MAE=float(np.mean(np.abs(e))),
                R2=float(1-mse/np.var(y)), bias=float(e.mean()))


def summarize():
    rec = registration()
    results = []
    for entity in ENTITIES:
        verify_freeze(entity)
        origins = baseline.issues(entity, evaluate=True)
        y = baseline.values(entity, evaluate=True)[origins, 0]
        with (np.load(path(entity, "predictions_private.npz")) as old,
              np.load(ROOT / "baseline_replay" / entity / "linear_evaluation.npz") as linear,
              np.load(ROOT / "baseline_replay" / entity / "neural_evaluation.npz") as neural):
            for archive in (old, linear, neural):
                np.testing.assert_array_equal(archive["target_index"], origins)
            arms = {k: score(y, old[k]) for k in ARMS}
            baselines = {k: score(y, (linear if k in linear else neural)[k]) for k in BASELINES}
        best = min(baselines, key=lambda k: baselines[k]["RMSE"])
        results.append(dict(entity=entity, N=len(y), fit=read(path(entity,"FIT.json")),
                            evaluation=read(path(entity,"EVALUATION.json")), ARMS=arms,
                            baselines=baselines, best_baseline=best, baseline_replay_exact=True))
    write(ROOT / "RESULTS.json", dict(status="COMPLETED", name=rec["name"], protocol=rec["protocol"], datasets=results))
    target = ROOT / "SCORES.csv"
    temp = target.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("entity", "N", "H", "D", "L", "arm", "RMSE", "MAE", "R2", "bias"))
        for row in results:
            for arm, metrics in (row["ARMS"] | row["baselines"]).items():
                writer.writerow((row["entity"], row["N"], hybrid.H, hybrid.D, hybrid.L, arm, *metrics.values()))
    temp.replace(target)
    print("VERIFIED_ALL", len(results), "entities", flush=True)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    {"register": register, "fit": fit, "freeze": freeze, "evaluate": evaluate,
     "baseline": replay_baseline, "summarize": summarize}[action](*args)
