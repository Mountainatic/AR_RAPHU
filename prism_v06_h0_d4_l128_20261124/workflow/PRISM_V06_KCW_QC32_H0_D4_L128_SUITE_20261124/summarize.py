"""Verify matched development rows and publish aggregate scores only."""

import csv
from pathlib import Path

import run as hybrid
from baselines import protocol

np = hybrid.np
ROOT = hybrid.ROOT
ARMS = ("persistence", "kcw", "persistence_qc32", "kcw_qc32")
BASELINES = ("Persistence", "AR", "ARX", *protocol.NAMES)
PRIOR = ROOT.parent / "PRISM_CZ_V06_KCW_QC32_H0_D4_L128_20261124"


def metrics(y, p):
    assert p.shape == y.shape and np.isfinite(p).all()
    e = y - p
    return dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))),
                R2=float(1-np.mean(e**2)/np.var(y)), bias=float(e.mean()))


def main():
    base = protocol.ROOT
    completed = hybrid.read(base / "COMPLETE.json")
    assert completed["freeze"] == hybrid.sha(base / "ALL_FROZEN.json")
    assert completed["registration"] == hybrid.sha(base / "REGISTRATION.json")
    protocol.ensure_bound("suite.py")
    cz_seal = hybrid.read(ROOT / "ALL_FROZEN.json")
    assert cz_seal["registration"] == hybrid.sha(ROOT / "REGISTRATION.json")
    records = []
    for entity in protocol.RODS:
        is_sru = entity == "SRU_H2S"
        origins = protocol.issues(entity, evaluate=True)
        data = protocol.values(entity, evaluate=True)
        y = data[origins, 0]
        h_path = ROOT / entity / ("predictions.npz" if is_sru else "predictions_private.npz")
        audit = hybrid.read(ROOT / entity / "RESULT.json")
        model_path = ROOT / entity / "FROZEN_MODEL.json"
        assert hybrid.sha(model_path) == audit["fit"]["model_sha256"]
        if not is_sru:
            assert hybrid.sha(model_path) == cz_seal["models"][entity]
        paths = [base / entity / f"{kind}_evaluation.npz" for kind in ("linear", "neural")]
        with np.load(h_path) as h, np.load(paths[0]) as linear, np.load(paths[1]) as neural:
            for archive in (h, linear, neural):
                np.testing.assert_array_equal(archive["target_index"], origins)
            for archive in (linear, neural):
                np.testing.assert_array_equal(archive["origin"], origins)
            np.testing.assert_array_equal(h["persistence"], linear["Persistence"])
            results = {k: metrics(y, h[k]) for k in ARMS}
            baseline_results = {k: metrics(y, (linear if k in linear.files else neural)[k]) for k in BASELINES}
            if not is_sru:
                results["kcw_fixed_a"] = metrics(y, h["kcw_fixed_a"])
        for arm, score in results.items():
            np.testing.assert_allclose(score["RMSE"], audit["RMSE"][arm], rtol=1e-13, atol=0)
        old_max_delta = None
        prior_result = PRIOR / entity / "RESULT.json"
        if not is_sru and prior_result.exists():
            previous = hybrid.read(prior_result)["RMSE"]
            old_max_delta = max(abs(results[arm]["RMSE"]-previous[arm]) for arm in results)
        best = min(baseline_results, key=lambda k: baseline_results[k]["RMSE"])
        freeze = base / entity
        neural_selection = {k: hybrid.read(freeze / "neural" / k / "FROZEN.json")["selected"] for k in protocol.NAMES}
        records.append(dict(entity=entity, N=len(origins), development=audit["development"], hybrid=results,
                            baselines=baseline_results, best_baseline=best, routes=audit["fit"]["routes"],
                            hybrid_rmse_change_vs_persistence_pct=100*(results["kcw_qc32"]["RMSE"] / results["persistence"]["RMSE"]-1),
                            hybrid_rmse_change_vs_best_baseline_pct=100*(results["kcw_qc32"]["RMSE"] / baseline_results[best]["RMSE"]-1),
                            prior_cz_replay_max_rmse_delta=old_max_delta,
                            causal_prefix_checks=audit["causal_prefix_checks"], qc32_online_updates=audit["qc32_online_updates"],
                            linear_selection=hybrid.read(freeze / "linear.json")["selected"],
                            neural_selection=neural_selection,
                            verified_artifact_sha256={str(p.relative_to(ROOT)): hybrid.sha(p)
                                                      for p in (model_path, h_path, *paths)}))
    output = dict(status="COMPLETED", name=hybrid.NAME, protocol=dict(H=hybrid.H, D=hybrid.D, L=hybrid.L),
                  qc32=hybrid.SPEC, development_only=True, datasets=records,
                  registration_sha256=dict(cz=hybrid.sha(ROOT / "REGISTRATION.json"),
                                           sru=hybrid.sha(ROOT / "SRU_H2S/REGISTRATION.json"),
                                           baseline=completed["registration"]))
    hybrid.write(ROOT / "SUITE_RESULTS.json", output)
    dest = ROOT / "SUITE_SCORES.csv"
    tmp = dest.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("entity", "N", "H", "D", "L", "arm", "RMSE", "MAE", "R2", "bias"))
        for row in records:
            entries = dict(row["hybrid"], **{k: v for k, v in row["baselines"].items() if k != "Persistence"})
            for arm, score in entries.items():
                writer.writerow((row["entity"], row["N"], hybrid.H, hybrid.D, hybrid.L, arm,
                                 *(score[k] for k in ("RMSE", "MAE", "R2", "bias"))))
    tmp.replace(dest)
    print("VERIFIED_AND_SCORED", len(records), "entities", flush=True)


if __name__ == "__main__":
    main()
