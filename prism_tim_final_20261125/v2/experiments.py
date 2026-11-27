"""Fit sealed PRISM v2 models and evaluate only the TIM E1–E3 protocols."""

import argparse
from dataclasses import replace
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("PRISM_TIM_ARCHIVE", ROOT.parent))
sys.path.append(str(ARCHIVE))

import core as old
import model as v2


np, sha, read, write = old.np, old.sha, old.read, old.write
ALIASES = {"rod1": old.RODS[0], "rod2": old.RODS[1], "sru": old.MAIN[1]}
CASES = tuple((name, h) for name in old.RODS for h in old.HORIZONS) + ((old.MAIN[1], 0),)
ABLATIONS = ("full", "equal_bins", "mean_only", "no_A0")
SEEDS = range(5)


def task(name, h):
    q = old.x.TASKS[name]
    return replace(q, H=h)


def register(out):
    old.verify_code()
    receipt = dict(
        base_commit="f111106ffef38587869276281f32abe1fd87983e",
        archived_code=sha(old.LOCK),
        archived_results=sha(old.ROOT/"RESULTS_LOCK.json"),
        archived_fit=sha(old.SUITE/"ALL_FROZEN.json"),
        frozen_data_receipt=sha(old.x.TAIL/"SOURCES.json"),
        source={name: sha(ROOT/name) for name in ("model.py", "group_en.c", "experiments.py")},
        cases=[list(case) for case in CASES],
        experiments=["E1", "E2", "E3"],
        same_entity_time_holdout=True,
        moment_support="01/99 percentiles of box means; 99th percentile of box standard deviation",
        sigma_scale="99th minus median, floored at 1e-9 times one plus the upper bound",
        trust="C group norm weighted; moving average of the last 128 issued rows",
        e1_primary="CZ Rod1 excludes the known overlap; SRU H2S uses the historical-contact tail",
        e2=list(ABLATIONS),
        stress="Constant and balanced-pair pulses, each 4 train standard deviations for 512 rows",
    )
    path = out/"REGISTRATION.json"
    if path.is_file():
        if read(path) != receipt:
            raise ValueError("Experiment registration does not match the current inputs")
    else:
        write(path, receipt)
    return sha(path)


def fit(out, name, h):
    receipt = register(out)
    dst = out/"frozen"/f"{name}_H{h}.json"
    if dst.is_file():
        if read(dst)["registration"] != receipt:
            raise ValueError("Stale model receipt: " + str(dst))
        return
    q = task(name, h)
    a = old.data(q)
    stop = q.calibration[1]
    is_main = (name in old.MAIN and h == old.x.TASKS[name].H)
    is_drift = name == old.RODS[1] and h == old.x.TASKS[name].H
    variants = ABLATIONS if is_main else ("full", "mean_only") if is_drift else ("full",)
    views = {}
    arms = {}
    for variant in variants:
        weighted = variant != "equal_bins"
        if weighted not in views:
            views[weighted] = v2.fit_views(q, a, weighted)
        k = views[weighted]
        engine = v2.Engine(q, a, k, stop, use_sigma=variant != "mean_only", use_a0=variant != "no_A0")
        arms[variant] = dict(k=k, model=engine.train(stop, q.train))
        active = [m for m in ("C", "W", "A", "A0") if arms[variant]["model"][m] is not None]
        print(name, h, variant, "active:", ",".join(active), flush=True)
    write(dst, dict(registration=receipt, protocol=[q.H, q.D, q.L],
                    last_fit_origin=stop-1, last_fit_target=stop-1+q.H, arms=arms))


def freeze(out):
    registration = register(out)
    paths = [out/"frozen"/f"{name}_H{h}.json" for name, h in CASES]
    if any(read(path)["registration"] != registration for path in paths):
        raise ValueError("At least one frozen model does not match this registration")
    path = out/"FROZEN.json"
    old.x.bind(path, dict(registration=registration, files={p.name: sha(p) for p in paths}))
    return read(path)["files"]


def comparable(q, t, full=True):
    target = t + q.H
    clean = np.ones(len(t), dtype=bool)
    if q.name in old.RODS:
        lo, hi = old.x.OVERLAP[q.name]
        clean = (target < lo) | (target >= hi)
    mask = t >= q.dev if full else target < q.dev
    return clean & mask


def eval_e1(q, a, t, pred):
    import e1 as archived_e1

    dst = old.SUITE/q.name
    with np.load(dst/"linear/PREDICTIONS.npz") as archive:
        np.testing.assert_array_equal(archive["origins"], t)
        baseline = {m: [archive[m]] for m in ("Persistence", "AR", "ARX")}
        baseline["ElasticNet"] = [archive[f"ElasticNet_{s}"] for s in SEEDS]
    baseline["LightGBM"] = archived_e1.lgb_predict(q, a, t)
    neural = dst/"neural/PREDICTIONS.npz" if q.H else old.ROOT/"E1/Transformer/PREDICTIONS.npz"
    with np.load(neural) as archive:
        np.testing.assert_array_equal(archive["origins"], t)
        prefix = "Transformer" if q.H else "seed"
        baseline["Transformer"] = [archive[f"{prefix}_{s}"] for s in SEEDS]
    with np.load(dst/"prism/PREDICTIONS.npz") as archive:
        np.testing.assert_array_equal(archive["origins"], t)
        baseline["PRISM_archive"] = [archive["PRISM"]]
    baseline["PRISM_v2"] = [pred]
    scores, comparisons = [], []
    for label, mask in (("development", comparable(q, t, False)), ("tail", comparable(q, t))):
        result = old.summarize(q, a, t, baseline, mask)
        scores.extend(dict(data=q.name, split=label, N=int(mask.sum()), method=m, repeats=len(v["runs"]),
                           **v["mean"], **{f"{k}_sd": value for k, value in v["sd"].items()})
                      for m, v in result.items())
        if label != "tail":
            continue
        truth = a[t[mask]+q.H, 0]
        errors = (pred[mask]-truth)**2
        lag = max(q.mature-1, int(4*(len(truth)/100)**(2/9)))
        for name, estimates in baseline.items():
            if name == "PRISM_v2":
                continue
            gain = np.mean([(z[mask]-truth)**2 for z in estimates], axis=0) - errors
            comparisons.append(dict(data=q.name, baseline=name, **archived_e1.dm(gain, lag), p_Holm=None))
    archived_e1.holm(comparisons)
    return scores, comparisons


def eval_stress(q, a, t, arms):
    mask = comparable(q, t)
    test = t[mask]
    scale = a[:q.parent, 1:].std(axis=0)
    perturbed = {"clean": a}
    for stress in ("constant", "oscillation"):
        dirty = a.copy()
        # Use one contiguous pulse on the retained tail and leave target observations untouched.
        window = np.arange(test[0], test[0] + 512)
        if not np.isin(window+q.H, test+q.H).all():
            raise ValueError("Stress window intersects excluded targets")
        # Randomize the orientation within each balanced pair to avoid cancellation in the delayed view.
        sign = np.random.default_rng(20261125).choice((-1, 1), size=len(window)//2)
        balanced = np.column_stack((sign, -sign)).reshape(-1, 1)
        pulse = 1 if stress == "constant" else balanced
        dirty[window, 1:] += 4 * scale * pulse
        perturbed[stress] = dirty

    truth, anchor = a[test+q.H, 0], a[test-q.D, 0]
    impact = np.arange(len(test)) < 512
    entries = []
    for variant in ("full", "mean_only"):
        arm = arms[variant]
        for stress, source in perturbed.items():
            e = v2.Engine(q, source, arm["k"], int(t[-1])+1, use_sigma=arm["model"]["use_sigma"])
            pred = e.predict(test, arm["model"])
            qhat, _, _, rho = e.readouts(test, *(arm["model"][k] for k in ("C", "W", "A", "A0")), q.train)
            view_trust = e.ctx["g"][test]
            for label, selected in (("pulse", impact), ("tail", np.ones(len(test), dtype=bool))):
                entries.append(dict(data=q.name, variant=variant, stress=stress, split=label,
                                    N=int(selected.sum()), mean_trust=float(rho[selected].mean()),
                                    mean_abs_Q=float(abs(qhat[selected]).mean()),
                                    mean_view_trust=float(view_trust[selected].mean()),
                                    zero_view_trust_fraction=float((view_trust[selected] == 0.).mean()),
                                    **old.score(truth[selected], pred[selected], anchor[selected])))
    return entries


def evaluate(out):
    import native_v2

    models = freeze(out)
    scores, dm, ablation, horizons, parity = [], [], [], [], []
    stress = []
    for name, h in CASES:
        filename = f"{name}_H{h}.json"
        path = out/"frozen"/filename
        if sha(path) != models[filename]:
            raise ValueError("Frozen model changed: " + filename)
        q = task(name, h)
        a = old.data(q, full=True)
        t = q.test_origins()
        arms = read(path)["arms"]
        estimates = {}
        for variant, arm in arms.items():
            model = arm["model"]
            e = v2.Engine(q, a, arm["k"], int(t[-1])+1, use_sigma=model["use_sigma"])
            py = e.predict(t, model)
            native = native_v2.replay(q, a, arm["k"], model, t)
            np.testing.assert_allclose(native, py, atol=1e-8, rtol=0)
            parity.append(dict(data=name, H=h, variant=variant, n=len(t),
                               max_abs=float(abs(native-py).max())))
            estimates[variant] = native
        if name in old.MAIN and h == old.x.TASKS[name].H:
            rows, tests = eval_e1(q, a, t, estimates["full"])
            scores.extend(rows)
            dm.extend(tests)
            for variant, estimate in estimates.items():
                for label, mask in (("development", comparable(q, t, False)), ("tail", comparable(q, t))):
                    selected = arms[variant]["model"]
                    ablation.append(dict(data=name, split=label, variant=variant, N=int(mask.sum()),
                                         **{k: selected[k] is not None for k in ("C", "W", "A", "A0")},
                                         **old.score(a[t[mask]+h, 0], estimate[mask], a[t[mask]-q.D, 0])))
        if name in old.RODS:
            target = t + h
            baseline = {"PRISM_v2": estimates["full"], "Persistence": a[t, 0]}
            archive = old.parent(h)/name/"linear/PREDICTIONS.npz"
            if archive.is_file():
                with np.load(archive) as linear:
                    np.testing.assert_array_equal(linear["origins"], t)
                    baseline["ARX"] = linear["ARX"]
            for seg, lo, hi in (("development", q.train+max(old.HORIZONS), q.dev),
                                ("tail", q.dev+max(old.HORIZONS), q.total)):
                overlap_lo, overlap_hi = old.x.OVERLAP[name]
                mask = (target >= lo) & (target < hi) & ((target < overlap_lo) | (target >= overlap_hi))
                for method, estimate in baseline.items():
                    horizons.append(dict(data=name, H=h, split=seg, N=int(mask.sum()), method=method,
                                         **old.score(a[target[mask], 0], estimate[mask], a[t[mask], 0])))
            if name == old.RODS[1] and h == old.x.TASKS[name].H:
                stress = eval_stress(q, a, t, arms)
        print("evaluated", name, h, flush=True)
    tables = {"E1/METRICS": scores, "E1/DM": dm, "E2/ABLATION": ablation,
              "E3/HORIZON": horizons, "E3/ROD2_PRESSURE": stress, "NATIVE_PARITY": parity}
    for path, rows in tables.items():
        file = out/(path+".csv")
        file.parent.mkdir(parents=True, exist_ok=True)
        old.csv_table(file, rows)
    write(out/"RESULTS.json", dict(frozen=sha(out/"FROZEN.json"),
          native_sources={name: sha(ROOT/name) for name in ("native_v2.py", "stream_v2.c")},
          tables={path+".csv": sha(out/(path+".csv")) for path in tables}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cmd = cli.add_subparsers(dest="action", required=True)
    cmd.add_parser("register")
    cmd.add_parser("evaluate")
    training = cmd.add_parser("fit")
    training.add_argument("entity", choices=ALIASES)
    training.add_argument("h", type=int)
    args = cli.parse_args()
    if args.action == "fit":
        name = ALIASES[args.entity]
        if (name, args.h) not in CASES:
            cli.error("This dataset and horizon are outside the E1–E3 registration")
        fit(args.out, name, args.h)
    elif args.action == "evaluate":
        evaluate(args.out)
    else:
        register(args.out)


if __name__ == "__main__":
    main()
