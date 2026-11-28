"""Refine the two remaining GRU search boundaries using the original train folds."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import extend_public_baselines as initial


def run(a):
    sys.path[:0] = [str(a.source), str(a.baseline_code), str(a.tep_code)]
    import benchmark_addendum as baseline
    from laguerre_screen import digest, save_json
    import torch

    q = baseline.Protocol(a)
    q.verify_source()
    source = a.initial / "FIT.json"
    registered = baseline.read(a.initial / "REGISTRATION.json")
    old_file = baseline.read(source)
    old = old_file["fit"]
    assert old_file["registration"] == digest(a.initial / "REGISTRATION.json")
    assert all(digest(a.initial / n) == v for n, v in old_file["weights"].items())
    assert initial.stable_registration(a, q, baseline, registered)
    if a.task == "TEP_D2":
        assert old["width_upper_boundary"]
    else:
        assert old["stop_at_epoch_cap"] and old["epochs_evaluated"] - old["epochs"] < initial.PATIENCE

    a.out.mkdir(parents=True, exist_ok=False)
    identity = {**registered, "refinement_source": digest(Path(__file__)),
                "first_gru_fit": digest(source)}
    save_json(a.out / "REGISTRATION.json", identity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    ids = q.rows
    x = q.sequence(q.train, ids)
    target = (q.train.truth - q.train.anchor())[ids]
    folds = [(tr - ids[0], val - ids[0]) for tr, val in q.folds]
    validation = sum(len(val) for _, val in folds)
    if q.tep:
        traces = {int(width): trace for width, trace in old["traces"].items()}
        widths = (512, 640, 768, 1024, 1280)
        candidates = list(widths[:2])
        cap = initial.EPOCH_CAP
    else:
        traces = {}
        widths = (*old["grid"], 128, 160, 192)
        candidates = list(old["grid"])
        cap = 256
    for width in candidates:
        learners = [initial.NeuralFold(torch, baseline, x, target, tr, val, width, 0, device)
                    for tr, val in folds]
        trace = []
        priority, last = float("inf"), 0
        for epoch in range(1, cap + 1):
            metric = sum(learner.epoch() for learner in learners) / validation
            trace.append(metric)
            if metric < priority * (1. - 1e-4):
                priority, last = metric, epoch
            if epoch >= initial.MIN_EPOCH and epoch - last >= initial.PATIENCE:
                break
        del learners
        traces[width] = trace
        best = min(traces, key=lambda n: min(traces[n]))
        print(a.task, width, "best epoch", int(np.argmin(trace)) + 1,
              "trained", len(trace), "mse", min(trace), flush=True)
        at_right = best == max(traces)
        if width == candidates[-1] and at_right and len(candidates) < len(widths):
            candidates.append(widths[len(candidates)])
    width = min(traces, key=lambda n: min(traces[n]))
    epochs = int(np.argmin(traces[width])) + 1
    for seed in q.seed:
        model, _ = baseline.fit_gru(x, target, np.arange(len(target)), width, epochs, seed, device)
        torch.save(model, a.out / f"GRU_{seed}.pt")
    decision = {**old, "width":width, "epochs":epochs, "epochs_evaluated":len(traces[width]),
                "grid":sorted(traces), "traces":traces, "epoch_cap":cap,
                "width_upper_boundary":width==max(traces), "device":device,
                "optimum_at_last_epoch":epochs==len(traces[width]),
                "stop_at_epoch_cap":len(traces[width])==cap,
                "first_gru_fit":digest(source), "refinement_source":digest(Path(__file__))}
    save_json(a.out / "FIT.json", dict(registration=digest(a.out / "REGISTRATION.json"),
              baseline="gru", fit=decision,
              weights={f.name:digest(f) for f in a.out.glob("GRU_*.pt")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("TEP_D2", "SO2"))
    for arg in ("source", "lock", "model", "data", "out", "archived", "baseline_code",
                "initial", "tep_code"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--tep-helper-digest", required=True)
    run(parser.parse_args())
