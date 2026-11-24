"""Check published scores and exercise the released timing contract on synthetic inputs."""

import csv
import math
import statistics
import sys

from study import HYBRID, RELEASE


def check_scores():
    output = RELEASE / "results"
    with (output / "RUN_METRICS.csv").open(encoding="utf-8", newline="") as handle:
        detail = list(csv.DictReader(handle))
    with (output / "MEAN_METRICS.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(detail) == 285 and len(summary) == 57
    for row in summary:
        selected = [v for v in detail if (v["entity"], v["model"]) == (row["entity"], row["model"])]
        assert {int(v["seed"]) for v in selected} == set(range(5)) and len(selected) == 5
        assert all(v["N"] == row["N"] for v in selected)
        for name in ("RMSE", "MAE", "MSE", "R2"):
            values = [float(v[name]) for v in selected]
            assert math.isclose(statistics.mean(values), float(row[f"{name}_mean"]), rel_tol=1e-14, abs_tol=1e-15)
            assert math.isclose(statistics.stdev(values), float(row[f"{name}_sd"]), rel_tol=1e-13, abs_tol=1e-15)
        assert all(math.isclose(float(v["RMSE"])**2, float(v["MSE"]), rel_tol=1e-14) for v in selected)


def check_causality():
    sys.path.insert(0, str(RELEASE / "workflow" / HYBRID))
    import run as model

    np = model.np
    entity = "CZ_Rod1"
    t = np.arange(800)
    rng = np.random.default_rng(7)
    y = np.sin(t / 31.) + rng.normal(0., .08, len(t))
    x = np.c_[t, np.sin(t / 13.), np.cos(t / 17.), np.sin(t / 19.), np.cos(t / 21.)]
    raw = np.c_[y, x]

    issue = model.frame(entity, 200, 201, "train", y=y)
    assert (issue.latest_available_target_index == 196).all()
    np.testing.assert_array_equal(issue.y_true, [y[200]-y[196]])
    accessor = model.CurrentProcessAccessor(entity, raw)
    current = accessor.block_means(issue, model.COLUMNS[0], [(0, 1), (0, 128)])
    np.testing.assert_array_equal(current, [[200., (73.+200.)/2.]])

    def stream(data, start, stop):
        frame = model.frame(entity, start, stop, "replay")
        i = frame.origin.to_numpy()
        q = data[i-model.D, 0] + 0.1 * data[i, 2]
        return model.stream(frame, data, q, "DEPLOYED_W_ONLINE")

    fitted = model.fit_qc32(stream(raw, 192, 500))
    assert len(fitted["theta"]) == 13 and fitted["fit_last_target_end"] == 499

    def replay(data, stop):
        first = 510
        active = stream(data, first, stop)
        history = stream(data, first-model.WARM, first)
        correction, audit = model.mature_a.predict_spec(active, fitted, model.SPEC, history=history)
        np.testing.assert_array_equal(audit["latest_feature_label"], active.origin.to_numpy()-model.D)
        assert np.all(audit["latest_parameter_label"] <= active.origin.to_numpy()-model.D)
        return active.issued_q.to_numpy()+correction

    cut = 620
    baseline = replay(raw, 650)
    changed = raw.copy()
    changed[cut-model.D+1:, 0] += 200.
    changed[cut+1:, 1:] -= 100.
    np.testing.assert_array_equal(replay(changed, cut+1), baseline[:cut+1-510])


if __name__ == "__main__":
    check_scores()
    check_causality()
    print("Verified the release tables, lag endpoints and causal QC32 replay.")
