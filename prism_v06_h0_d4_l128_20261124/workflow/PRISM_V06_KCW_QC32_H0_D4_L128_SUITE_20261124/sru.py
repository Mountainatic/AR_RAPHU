"""Fit and score the October KCW and November QC32 hybrid on SRU H₂S."""

from pathlib import Path
import sys
import time

import run as hybrid
from baselines import protocol
from canonical.cz_crossrod_l128_20261008 import canonical_engine
from prism_benchmark.cpu_data import BaseAccessor

np = hybrid.np
ENTITY = "SRU_H2S"
COLUMNS = ("u1", "u2", "u3", "u4", "u5")
OUT = hybrid.ROOT / ENTITY
REG = OUT / "REGISTRATION.json"
MODEL = OUT / "FROZEN_MODEL.json"
FIT = OUT / "FIT.json"


def register():
    _, train_end, dev_end = protocol.GEOMETRY[ENTITY]
    parent_end = protocol.GEOMETRY[ENTITY][0]
    data = dict(parent=[hybrid.FIRST, parent_end],
                calibration=[parent_end+hybrid.L+2*hybrid.D, train_end-hybrid.D+1],
                development=[train_end, dev_end], data_sha256=protocol.file_hashes(ENTITY))
    registration = dict(name=hybrid.NAME, protocol=dict(H=hybrid.H, D=hybrid.D, L=hybrid.L),
                        data=data, target="H2S (y1)", process_channels=COLUMNS,
                        cz_common_source_sha256=hybrid.sha(hybrid.ROOT / "REGISTRATION.json"),
                        source_sha256={str(p): hybrid.sha(p) for p in
                                       (Path(__file__).resolve(), Path(protocol.__file__).resolve())},
                        kcw="October v0.6 matched CZ registry and stagewise routing, raw SRU channels without PCA",
                        A=dict(source="November 22 live QC32 implementation", **hybrid.SPEC),
                        history="Calibrate on a held-out contiguous training stream of the frozen KCW or persistence.",
                        development="Score every issue on [6048,8064), with chronological delayed residual feedback.")
    if REG.exists():
        assert hybrid.read(REG) == hybrid.json.loads(hybrid.json.dumps(registration))
    else:
        hybrid.write(REG, registration)


def bound_registration():
    registration = hybrid.read(REG)
    common = hybrid.ROOT / "REGISTRATION.json"
    assert hybrid.sha(common) == registration["cz_common_source_sha256"]
    sources = registration["source_sha256"] | hybrid.read(common)["sources"]
    assert all(hybrid.sha(Path(p)) == digest for p, digest in sources.items())
    return registration


class SRUAccessor(BaseAccessor):
    def __init__(self, data):
        self.columns = frozenset(COLUMNS)
        self.pca = None
        # v0.6 windows have an exclusive right edge. Physical row t is stored at t-1.
        self.entities = {ENTITY: (np.arange(len(data))-1, dict(zip(COLUMNS, data[:, 1:].T)))}
        self.warm_prefixes(COLUMNS)


class SRUPipeline(hybrid.OctoberPipeline):
    def __init__(self, data, reg):
        canonical_engine.CHANNELS = list(COLUMNS)
        self.data = data
        source = hybrid.read(hybrid.CANONICAL / "CANONICAL_PROTOCOL.json")
        source["task"] = dict(**reg["protocol"], evaluation="SRU H2S development")
        source["execution_authorization"]["target_test_inference"] = False
        registration = dict(experiment=hybrid.sha(REG), rod=ENTITY,
                            canonical_protocol_sha256={p.name: hybrid.sha(p) for p in
                                                       hybrid.CANONICAL.glob("*REGISTRY.json")})
        output = OUT / "october_fit"
        context = dict(registration=registration, protocol=source, policy_id=hybrid.digest(registration),
                       shared=str(protocol.SRU), authority=str(hybrid.CODE), case_id=ENTITY,
                       resume=output.exists(), release_sha=hybrid.digest(registration), C1_manifest=reg["data"])
        canonical_engine.CanonicalPipeline.__init__(self, context, output, mode="formal_development")
        self.fixed_cache, self.fixed_stream_cache, self.local_chain_cache, self.active_parents = {}, {}, {}, {}
        self.K_specs = [spec for spec in self.K_specs if spec[1] == 4 and spec[4] == .001
                        and ((spec[2] == "LINEAR_DISTRIBUTED_LAG" and spec[5] == 0.)
                             or (spec[2] != "LINEAR_DISTRIBUTED_LAG" and spec[3] == 4 and spec[5] == .001))]
        assert len(self.K_specs) == 95

    def _make_accessor(self, shared, split):
        assert split == "train"
        return SRUAccessor(self.data)

    def fold_accessor(self, fit):
        return self.accessor


def values(reg, development=False):
    _, train_end, dev_end = protocol.GEOMETRY[ENTITY]
    parts = [("train", train_end)]
    if development:
        parts.append(("development", dev_end-train_end))
    data = []
    for part, count in parts:
        path = protocol.SRU / "data" / f"{part}.npz"
        assert hybrid.sha(path) == reg["data"]["data_sha256"][part]
        with np.load(path) as archive:
            raw = archive["values"]
        assert raw.shape == (count, len(COLUMNS)+1) and np.isfinite(raw).all()
        data.append(raw)
    return np.vstack(data)


def fit():
    started = time.monotonic()
    reg = bound_registration()
    data = values(reg)
    pipe = SRUPipeline(data, reg)
    rows = hybrid.frame(ENTITY, *reg["data"]["parent"], "train", y=data[:, 0])
    start, stop = reg["data"]["calibration"]
    assert rows.target_stop_exclusive.max() < start-(hybrid.L-1)
    kcw = pipe.learn("W", rows)
    f, persistence, q = hybrid.upstream(ENTITY, data, pipe, kcw, start, stop, "calibration")
    models = {"kcw": kcw}
    for arm, parent in (("persistence_qc32", persistence), ("kcw_qc32", q)):
        models[arm] = hybrid.fit_qc32(hybrid.stream(f, data, parent, "CROSS_FITTED_PREQUENTIAL"))
        assert models[arm]["fit_last_target_end"] == reg["data"]["development"][0]-hybrid.D
    hybrid.write(MODEL, models)
    hybrid.write(FIT, dict(registration=hybrid.sha(REG), model_sha256=hybrid.sha(MODEL),
                           routes=hybrid.route_summary(kcw), fit_calls=pipe.fit_calls,
                           elapsed_seconds=time.monotonic()-started))
    print("FROZEN", ENTITY, flush=True)


def predict(data, pipe, model, start, stop):
    warm = hybrid.WARM
    f, persistence, q = hybrid.upstream(ENTITY, data, pipe, model["kcw"], start-warm, stop, "replay")
    result = {"persistence": persistence[warm:], "kcw": q[warm:]}
    updates = {}
    for arm, parent in (("persistence_qc32", persistence), ("kcw_qc32", q)):
        g = hybrid.stream(f, data, parent, "DEPLOYED_W_ONLINE")
        active = g.iloc[warm:]
        correction, audit = hybrid.mature_a.predict_spec(active, model[arm], hybrid.SPEC, history=g.iloc[:warm])
        latest = active.latest_available_target_index.to_numpy()
        np.testing.assert_array_equal(audit["latest_feature_label"], latest)
        assert np.all(audit["latest_parameter_label"] <= latest)
        assert audit["history_parameter_updates"] == 0
        assert audit["updates"] == max(len(active)-hybrid.D, 0)
        result[arm] = parent[warm:] + correction
        updates[arm] = audit["updates"]
    return result, updates


def evaluate():
    reg = bound_registration()
    receipt = hybrid.read(FIT)
    assert hybrid.sha(REG) == receipt["registration"] and hybrid.sha(MODEL) == receipt["model_sha256"]
    model = hybrid.read(MODEL)
    start, stop = reg["data"]["development"]
    data = values(reg, development=True)
    pipe = SRUPipeline(values(reg), reg)
    pipe.accessor = SRUAccessor(data)
    output, updates = predict(data, pipe, model, start, stop)
    for cut in (start, start+4, start+72, stop-2):
        poison = data.copy()
        poison[cut-hybrid.D+1:, 0] += 1200.
        poison[cut+1:, 1:] -= 1300.
        pipe.accessor = SRUAccessor(poison)
        trial, _ = predict(poison, pipe, model, start, cut+1)
        for arm, predicted in output.items():
            np.testing.assert_allclose(trial[arm], predicted[:cut-start+1], rtol=1e-12, atol=1e-10)
    origin = np.arange(start, stop)
    np.testing.assert_array_equal(output["persistence"], data[origin-hybrid.D, 0])
    y = data[origin, 0]
    rmse = {arm: float(np.sqrt(np.mean((y-predicted)**2))) for arm, predicted in output.items()}
    np.savez_compressed(OUT / "predictions.npz", target_index=origin, **output)
    hybrid.write(OUT / "RESULT.json", dict(entity=ENTITY, N=len(origin), development=[start, stop],
                 RMSE=rmse, qc32_online_updates=updates, causal_prefix_checks=4, fit=receipt))
    print("SCORED", ENTITY, rmse, flush=True)


if __name__ == "__main__":
    {"register": register, "fit": fit, "evaluate": evaluate}[sys.argv[1]]()
