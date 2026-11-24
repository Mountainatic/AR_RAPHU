"""Train and score the same-rod October KCW / November QC32 experiment."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("AR_RAPHU_RUNTIME_MANAGER", "uv")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get("CZ_HYBRID_SERVER_BASE", ROOT.parent))
CODE = ROOT / "october_source"
CANONICAL = CODE / "canonical/cz_crossrod_l128_20261008"
NOVEMBER = BASE / "PRISM_CZ_D0_DEBUTANIZER_D4_20261122"
A_CODE = NOVEMBER / "CZ/runtime/delta_CZ"
DRIVER = BASE / "PRISM_PAPER_DRIVER_A_20261030"
CACHE = BASE / "cz_sru_matched_20261024"
sys.path[:0] = [str(CODE), str(CODE / "src"), str(A_CODE), str(DRIVER)]

from canonical.cz_crossrod_l128_20261008 import a_state, stagewise_tuning
from canonical.cz_crossrod_l128_20261008.canonical_engine import CanonicalPipeline
from canonical.cz_crossrod_l128_20261008.invariants import digest
from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
from prism_benchmark.cpu_data import BaseAccessor
from prism_benchmark.v211_support import SUPPORT_CONTRACT
import mature_a

NAME = "v0.6 KCW + QC32-RLS A，H=0,D=4,L=128"
H, D, L = 0, 4, 128
WARM = 32 + D
FIRST = 192
SPEC = dict(family="QC32_RLS", p=1, window=0, forget=.995, gain=1.)
COLUMNS = ("main_heater_power", "crystal_lift", "crucible_lift",
           "crystal_rotation", "crucible_rotation")
GEOMETRY = {
    "CZ_Rod1": (6636, 12061, 16082),
    "CZ_Rod2": (6810, 12376, 16501),
}
ARMS = ("persistence", "kcw", "persistence_qc32", "kcw_qc32", "kcw_fixed_a")
LEGACY_PROFILES = ((1, 4), (1, 16), (2, 32), (4, 64))
LEGACY_ALPHA = .0002310129700083158


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register():
    data = {}
    for rod, (parent_end, train_end, dev_end) in GEOMETRY.items():
        reference = read(CACHE / rod / "REGISTRATION.json")
        blocks = reference["data_files"] if "data_files" in reference else next(
            value for value in reference.values()
            if isinstance(value, dict) and {"train", "development"}.issubset(value)
            and isinstance(value["train"], dict) and "export_sha256" in value["train"]
        )
        assert all(v["columns"] == ["diameter", *COLUMNS] for v in blocks.values()
                   if isinstance(v, dict) and "columns" in v)
        data[rod] = dict(
            parent=[FIRST, parent_end],
            calibration=[parent_end + L + 2 * D, train_end - D + 1],
            development=[train_end, dev_end],
            data_sha256={p: blocks[p]["export_sha256"] for p in ("train", "development")},
        )
    november = read(NOVEMBER / "CZ/CPU_REGISTRATION.json")["code_sha256"]
    assert all(sha(A_CODE / name) == november[str(A_CODE / name)]
               for name in ("mature_a.py", "rls_core.py"))
    sources = [
        Path(__file__).resolve(), A_CODE / "mature_a.py", A_CODE / "rls_core.py",
        DRIVER / "warm_history_a.py",
        *sorted(CANONICAL.glob("*.py")), *sorted(CANONICAL.glob("*REGISTRY.json")),
        CANONICAL / "CANONICAL_PROTOCOL.json",
        CODE / "configs/cz_h1_k_temporal_registry_v2/UNIT_TESTS.xml",
        CODE / "scripts/preflight_cz_h1_k_registry_v2.py",
    ]
    reg = dict(name=NAME, protocol=dict(H=H, D=D, L=L), data=data, arms=ARMS,
               kcw_source="10 月 v0.6 逐层 CZ 实现（6ba6f1b）",
               k_grid="October 20 regularization and basis widths; all 19 October CZ profiles; five K families",
               C_W_grid="Full October CZ registries with original routing",
               qc32_source="11 月 22 日现行版（CZ matched protocol source binding）",
               qc32=SPEC, qc32_route="Always applied in each named QC32 arm",
               fixed_a=dict(profiles=LEGACY_PROFILES, alpha=LEGACY_ALPHA, mu=0.,
                            profile_selection="two forward calibration folds, no online coefficient update"),
               calibration_policy="one frozen parent, contiguous held-out training stream per rod",
               scoring_policy="all existing development target rows, train history for features only",
               final_test_access=False, sources={str(path): sha(path) for path in sources})
    dest = ROOT / "REGISTRATION.json"
    if dest.exists():
        assert read(dest) == json.loads(json.dumps(reg))
    else:
        write(dest, reg)


def values(rod, evaluate=False):
    reg = read(ROOT / "REGISTRATION.json")["data"][rod]
    paths = CACHE / rod / "data"
    parts = ("train", "development") if evaluate else ("train",)
    arrays = []
    for part in parts:
        path = paths / f"{part}.npz"
        assert sha(path) == reg["data_sha256"][part]
        with np.load(path) as archive:
            array = archive["values"]
        start, stop = (0, reg["development"][0]) if part == "train" else reg["development"]
        assert array.shape == (stop - start, 1 + len(COLUMNS))
        assert np.isfinite(array).all()
        arrays.append(array)
    return np.vstack(arrays)


class CurrentProcessAccessor(BaseAccessor):
    """Translate the old exclusive process origin so each window ends at t."""

    def __init__(self, rod, data):
        self.columns = frozenset(COLUMNS)
        # The historical accessor queries [origin-far, origin-near).
        # Stored row coordinates s-1 therefore map back to physical raw row s.
        self.entities = {rod: (np.arange(len(data)) - 1, dict(zip(COLUMNS, data[:, 1:].T)))}
        self.pca = None
        self.warm_prefixes(COLUMNS)

    def has_columns(self, columns):
        return super().has_columns(set(columns) - {"joint_lift"})


class OctoberPipeline(StagewisePipeline):
    def __init__(self, rod, data):
        self.data = data
        protocol = read(CANONICAL / "CANONICAL_PROTOCOL.json")
        protocol["task"] = dict(H=H, D=D, L=L, evaluation="same-rod development")
        protocol["execution_authorization"]["target_test_inference"] = False
        reg = read(ROOT / "REGISTRATION.json")
        registration = dict(
            experiment=sha(ROOT / "REGISTRATION.json"), rod=rod,
            canonical_protocol_sha256={p.name: sha(p) for p in CANONICAL.glob("*REGISTRY.json")},
        )
        output = ROOT / rod / "october_fit"
        context = dict(registration=registration, protocol=protocol, policy_id=digest(registration),
                       shared=str(CACHE), authority=str(CODE), case_id=rod, resume=output.exists(),
                       release_sha=digest(registration), C1_manifest=reg["data"][rod])
        # Start a new fit under H=0,D=4. The old cross-rod constructor imports K
        # checkpoints whose targets and input visibility belong to its own protocol.
        CanonicalPipeline.__init__(self, context, output, mode="formal_development")
        self.fixed_cache, self.fixed_stream_cache, self.local_chain_cache = {}, {}, {}
        self.active_parents = {}
        self.K_specs = [
            spec for spec in self.K_specs if spec[1] == 4 and spec[4] == .001
            and ((spec[2] == "LINEAR_DISTRIBUTED_LAG" and spec[5] == 0.)
                 or (spec[2] != "LINEAR_DISTRIBUTED_LAG" and spec[3] == 4 and spec[5] == .001))
        ]
        assert len(self.K_specs) == 95

    def _make_accessor(self, shared, split):
        assert split == "train"
        return CurrentProcessAccessor(self.context["case_id"], self.data)

    def tune_k(self, frame, destination):
        print("Tuning K", self.context["case_id"], len(frame), flush=True)
        return stagewise_tuning.tune_k(self, frame, destination)


def frame(rod, start, stop, split, y=None):
    t = np.arange(start, stop)
    keys = [f"{rod}:t{i}" for i in t]
    out = pd.DataFrame(dict(
        entity_id=rod, origin=t, target_stop_exclusive=t + 1,
        latest_available_target_index=t-D, dependency_start=t-(L-1),
        dependency_stop_exclusive=t+1, causal_history_floor=-1, anchor_history_steps=D,
        sample_support_contract=SUPPORT_CONTRACT, sample_id=keys, view_sample_id=keys,
        base_origin_id=keys, split=split,
    ))
    if y is not None:
        out["y_true"] = y[t] - y[t-D]
    return out


def upstream(rod, data, pipe, model, start, stop, split):
    f = frame(rod, start, stop, split)
    t = f.origin.to_numpy()
    persistence = data[t-D, 0]
    return f, persistence, persistence + pipe.emit(model, f)


def stream(f, data, q, role):
    return f.assign(issued_q=q, residual=data[f.origin.to_numpy(), 0] - q,
                    residual_source_role=role, split_id="contiguous")


def fit_qc32(f):
    x, native, audit = mature_a.dynamic_features(f, SPEC["p"], SPEC["family"])
    assert x.shape[1] == 12
    np.testing.assert_array_equal(audit["latest_feature_target_end"][native],
                                  f.latest_available_target_index.to_numpy()[native])
    model = mature_a.initialize(f, np.arange(len(f)), SPEC, {(SPEC["family"], SPEC["p"]): (x, native, audit)})
    assert len(model["theta"]) == 13
    return model


def legacy_state(f):
    return a_state._state(f, a_state.INFERENCE_DEPLOYED, "frozen_october_kcw", 1, 1)


def fit_fixed_model(pipe, f, profile):
    state = legacy_state(f)
    _, native, _ = a_state.feature_rows(f, state, profile, pipe.A_max, 0.)
    mean = float(f.residual.to_numpy()[native].mean())
    x, _, _ = a_state.feature_rows(f, state, profile, pipe.A_max, mean)
    _, model = pipe.a.fit_mature_residual_ar(x, f.residual.to_numpy()[native], x,
                                           alpha=LEGACY_ALPHA, mu=0.)
    assert pipe.cert(model)
    return dict(profile=profile, residual_mean=mean, contract=model)


def fixed_prediction(pipe, f, history, model):
    state = legacy_state(pd.concat([history, f], ignore_index=True))
    x, native, _ = a_state.feature_rows(f, state, model["profile"], pipe.A_max, model["residual_mean"])
    result = np.zeros(len(f))
    result[native] = pipe.a.predict_mature_residual_ar(x, model["contract"])
    return result


def select_fixed_model(pipe, f):
    scores = []
    for profile in LEGACY_PROFILES:
        errors = []
        for lo, hi in ((.6, .8), (.8, 1.)):
            begin, end = int(lo * len(f)), int(hi * len(f))
            train = f.iloc[:begin-L-64-D]
            valid = f.iloc[begin:end]
            model = fit_fixed_model(pipe, train, profile)
            prediction = fixed_prediction(pipe, valid, f.iloc[begin-64-D:begin], model)
            errors.append(valid.residual.to_numpy() - prediction)
        scores.append(float(np.mean(np.concatenate(errors)**2)))
    model = fit_fixed_model(pipe, f, LEGACY_PROFILES[int(np.argmin(scores))])
    return dict(model, profile_calibration_mse=scores)


def route_summary(model):
    stages = {}
    while model is not None:
        stage = model["stage"]
        selection = model.get("selection") or {}
        summary = dict(route=model["routing_status"], recipe=model.get("recipe"),
                       selection=selection if stage != "K" else {})
        if stage == "K":
            summary.update(reference=model.get("reference_channel"),
                           channels=[{k: r[k] for k in ("channel", "profile", "family", "m_tau", "m_x")}
                                     for r in model["representations"]])
        if stage == "C":
            summary["alpha"] = model.get("selected_alpha")
        stages[stage] = summary
        model = model["parent"]
    return stages


def train(rod):
    reg = read(ROOT / "REGISTRATION.json")["data"][rod]
    data = values(rod)
    started = time.monotonic()
    pipe = OctoberPipeline(rod, data)
    t = frame(rod, *reg["parent"], "train", y=data[:, 0])
    start, stop = reg["calibration"]
    assert int(t.target_stop_exclusive.max()) < start-(L-1)
    model = pipe.learn("W", t)
    f, persistence, q = upstream(rod, data, pipe, model, start, stop, "calibration")
    calibration = stream(f, data, q, "CROSS_FITTED_PREQUENTIAL")
    frozen = dict(kcw=model, kcw_qc32=fit_qc32(calibration),
                  persistence_qc32=fit_qc32(stream(f, data, persistence, "CROSS_FITTED_PREQUENTIAL")),
                  fixed_a=select_fixed_model(pipe, calibration))
    assert all(frozen[k]["fit_last_target_end"] == reg["development"][0]-D
               for k in ("kcw_qc32", "persistence_qc32"))
    dest = ROOT / rod / "FROZEN_MODEL.json"
    write(dest, frozen)
    write(ROOT / rod / "FIT.json", dict(rod=rod, model_sha256=sha(dest), routes=route_summary(model),
          k_candidates_per_channel=len(pipe.K_specs), C_candidates=len(pipe.C_specs), W_candidates=len(pipe.W_specs),
          QC32_fit_rows={k: frozen[k]["fit_rows"] for k in ("kcw_qc32", "persistence_qc32")},
          fixed_a_profile=frozen["fixed_a"]["profile"], elapsed_seconds=time.monotonic()-started,
          fit_calls=pipe.fit_calls))
    print("FROZEN", rod, flush=True)


def freeze():
    models = {}
    for rod in GEOMETRY:
        model = ROOT / rod / "FROZEN_MODEL.json"
        assert sha(model) == read(ROOT / rod / "FIT.json")["model_sha256"]
        models[rod] = sha(model)
    write(ROOT / "ALL_FROZEN.json", dict(models=models, registration=sha(ROOT / "REGISTRATION.json")))


def predictions(rod, data, pipe, model, start, stop):
    f, persistence, q = upstream(rod, data, pipe, model["kcw"], start-WARM, stop, "replay")
    result = {"persistence": persistence[WARM:], "kcw": q[WARM:]}
    updates = {}
    for arm, parent in (("persistence_qc32", persistence), ("kcw_qc32", q)):
        g = stream(f, data, parent, "DEPLOYED_W_ONLINE")
        active, history = g.iloc[WARM:], g.iloc[:WARM]
        correction, audit = mature_a.predict_spec(active, model[arm], SPEC, history=history)
        latest = active.latest_available_target_index.to_numpy()
        np.testing.assert_array_equal(audit["latest_feature_label"], latest)
        assert np.all(audit["latest_parameter_label"] <= latest)
        assert audit["history_parameter_updates"] == 0
        assert audit["updates"] == max(len(active)-D, 0)
        result[arm] = parent[WARM:] + correction
        updates[arm] = audit["updates"]
    g = stream(f, data, q, "DEPLOYED_W_ONLINE")
    # The longest fixed-A profile can need more history than QC32.
    old_warm = D + max(a_state.offsets(model["fixed_a"]["profile"], pipe.A_max))
    old_f, _, old_q = upstream(rod, data, pipe, model["kcw"], start-old_warm, start, "replay")
    history = stream(old_f, data, old_q, "DEPLOYED_W_ONLINE")
    result["kcw_fixed_a"] = q[WARM:] + fixed_prediction(pipe, g.iloc[WARM:], history, model["fixed_a"])
    return result, updates


def evaluate(rod):
    seal = read(ROOT / "ALL_FROZEN.json")
    assert sha(ROOT / "REGISTRATION.json") == seal["registration"]
    model_path = ROOT / rod / "FROZEN_MODEL.json"
    assert sha(model_path) == seal["models"][rod]
    model = read(model_path)
    start, stop = read(ROOT / "REGISTRATION.json")["data"][rod]["development"]
    data = values(rod, evaluate=True)
    pipe = OctoberPipeline(rod, values(rod))
    pipe.accessor = CurrentProcessAccessor(rod, data)
    result, updates = predictions(rod, data, pipe, model, start, stop)
    for cut in (start, start+4, start+72, stop-2):
        poison = data.copy()
        poison[cut-D+1:, 0] += 1200.
        poison[cut+1:, 1:] -= 1300.
        pipe.accessor = CurrentProcessAccessor(rod, poison)
        trial, _ = predictions(rod, poison, pipe, model, start, cut+1)
        for arm in ARMS:
            np.testing.assert_allclose(trial[arm], result[arm][:cut-start+1], rtol=1e-12, atol=1e-10)
    t = np.arange(start, stop)
    y = data[t, 0]
    np.testing.assert_array_equal(result["persistence"], data[t-D, 0])
    error = {arm: float(np.sqrt(np.mean((y-result[arm])**2))) for arm in ARMS}
    np.savez_compressed(ROOT / rod / "predictions_private.npz", target_index=t, **result)
    write(ROOT / rod / "RESULT.json", dict(rod=rod, N=len(t), development=[start, stop], RMSE=error,
          KCW_gain_over_persistence_QC32=dict(absolute=error["persistence_qc32"]-error["kcw_qc32"],
          percent=100*(1-error["kcw_qc32"]/error["persistence_qc32"])),
          QC32_gain_over_fixed_A_pct=100*(1-error["kcw_qc32"]/error["kcw_fixed_a"]),
          qc32_online_updates=updates, causal_prefix_checks=4, fit=read(ROOT / rod / "FIT.json")))
    print("SCORED", rod, error, flush=True)


def summarize():
    result = [read(ROOT / rod / "RESULT.json") for rod in GEOMETRY]
    write(ROOT / "RESULTS.json", dict(status="COMPLETED", name=NAME, protocol=dict(H=H,D=D,L=L),
                                    qc32=SPEC, results=result))
    with (ROOT / "SCORES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rod", "N", "H", "D", "L", "arm", "RMSE"))
        for rod in result:
            for arm in ARMS:
                writer.writerow((rod["rod"], rod["N"], H, D, L, arm, rod["RMSE"][arm]))


if __name__ == "__main__":
    actions = dict(register=register, train=train, freeze=freeze, evaluate=evaluate, summarize=summarize)
    action, *args = sys.argv[1:]
    actions[action](*args)
