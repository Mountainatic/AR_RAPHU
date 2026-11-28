"""Export the manuscript tables from validated final score panels."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent
CASES = ("TEP_D1", "TEP_D2", "H2S", "SO2", "Rod1", "Rod2")
METHODS = ("PRISM", "Persistence", "AR", "DPLS", "ARX", "ElasticNet", "ElasticNet_clipped",
           "LightGBM", "GRU", "Transformer")
LABEL = {"ElasticNet": "Elastic Net", "ElasticNet_clipped": "Clipped Elastic Net"}
NAME = {"TEP_D1": r"TEP ($D=1$)", "TEP_D2": r"TEP ($D=2$)", "H2S": r"SRU ($\mathrm{H_2S}$)",
        "SO2": r"SRU ($\mathrm{SO_2}$)", "Rod1": r"CZ Rod1 ($H=16$)", "Rod2": r"CZ Rod2 ($H=16$)"}
CLEAN_NOTE = (r"$R^2=1-\sum_t(y_t-\hat y_t)^2/\sum_t(y_t-\bar y_{\mathrm{tail}})^2$, "
              r"where $\bar y_{\mathrm{tail}}$ is the mean of the scored targets in each tail. "
              r"Stars indicate two-sided Bartlett HAC Diebold--Mariano tests against PRISM using "
              r"mean per-seed squared loss with Holm correction across nine baselines "
              r"(\({}^*p<.05\), \({}^{**}p<.01\), \({}^{***}p<.001\)). "
              r"\(+\) denotes higher baseline loss and \(-\) denotes lower baseline loss.")
FAULT_NOTE = ("Shift MAE is the mean absolute difference between a method's perturbed prediction "
              "and its own clean prediction on the same issues.")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stars(test):
    p = test["p_holm"]
    significance = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    return significance + ("+" if test["mean_loss_gain"] > 0 else "-") if significance else ""


def math_cell(value, decimals, marker=""):
    superscript = "^{" + r"\ast" * (len(marker)-1) + marker[-1] + "}" if marker else ""
    return rf"\({value:.{decimals}f}{superscript}\)"


def latex(title, label, groups, columns, data, note):
    def row(cells):
        return " & ".join(cells) + r" \\"
    heading = [""] + [r"\multicolumn{3}{c}{" + g + "}" for g in groups]
    return "\n".join([r"\begin{table*}[!t]", r"\centering", r"\caption{" + title + "}",
                      r"\label{" + label + "}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\hline",
                      row(heading), r"\cline{2-4}\cline{5-7}", row(["Method"] + columns * 2), r"\hline",
                      *[row(c) for c in data], r"\hline", r"\end{tabular}", r"\par\smallskip",
                      r"\parbox{0.96\textwidth}{\footnotesize " + note + "}", r"\end{table*}", ""])


def render(panels, cases=CASES):
    tables = []
    for keys, title, label in ((CASES[:2], "Tennessee Eastman", "tep"),
                               (CASES[2:4], "sulfur recovery", "sru"),
                               (CASES[4:], "CZ crystal growth after target-overlap masking", "cz")):
        if not all(case in cases for case in keys):
            continue
        data = []
        for method in METHODS:
            cells = [LABEL.get(method, method)]
            for case in keys:
                arm = panels[case]["clean"][method]
                cells += [math_cell(arm["rmse"], 6, arm["rmse_dm_stars"]),
                          math_cell(arm["mae"], 6), math_cell(arm["r2"], 4)]
            data.append(cells)
        tables.append(latex("Clean-tail results on " + title, "tab:clean-" + label,
                            [NAME[c] for c in keys], ["RMSE", "MAE", r"$R^2$"], data, CLEAN_NOTE))
    for case in cases:
        data = []
        for method in METHODS:
            cells = [LABEL.get(method, method)]
            for scenario in ("oscillation", "bias"):
                arm = panels[case][scenario][method]
                cells += [math_cell(arm[k], 6) for k in ("rmse", "mae", "output_shift_mae")]
            data.append(cells)
        tables.append(latex(NAME[case] + " performance in matched 512-point sensor-fault windows",
                            "tab:fault-" + case.lower().replace("_", "-"),
                            ["Oscillation", r"$4\sigma$ constant bias"], ["RMSE", "MAE", "Shift MAE"],
                            data, FAULT_NOTE))
    return "% Import into the manuscript with \\input.\n" + "\n".join(tables)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_fit(base, case, archive_sha, release_sha, result):
    if case.startswith("Rod"):
        assert result["preserved_result"] == archive_sha
        return {"reused": True}
    lock = read(base / "TEST_LOCK.json")
    assert lock["old_result"] == archive_sha and lock["source_lock"] == release_sha
    assert digest(base / "TEST_LOCK.json") == result["test_lock"]
    selected = {}
    for kind in ("dpls", "gru"):
        folder = base / kind
        fit = read(folder / "FIT.json")
        reg = read(folder / "REGISTRATION.json")
        assert digest(folder / "FIT.json") == lock["fits"][kind]
        assert fit["registration"] == digest(folder / "REGISTRATION.json")
        assert reg["study"] == digest(OUT / "extend_public_baselines.py")
        if "refinement_source" in reg:
            assert reg["refinement_source"] == digest(OUT / "refine_boundary_gru.py")
            assert reg["first_gru_fit"] == digest(base / "gru_initial" / "FIT.json")
            assert reg["first_gru_fit"] == fit["fit"]["first_gru_fit"]
        selected[kind] = fit["fit"]
    d, g = selected["dpls"], selected["gru"]
    return dict(dpls_components=d["components"], dpls_grid_max=max(d["tried"]),
                dpls_at_upper_boundary=d["upper_boundary"], gru_width=g["width"],
                gru_width_max=max(g["grid"]), gru_epochs=g["epochs"],
                gru_epochs_evaluated=g["epochs_evaluated"],
                gru_width_at_upper_boundary=g["width_upper_boundary"],
                gru_optimum_at_last_epoch=g["optimum_at_last_epoch"],
                gru_stop_at_epoch_cap=g["stop_at_epoch_cap"])


def main(release, previous, subset):
    cases = CASES[4:] if subset == "cz" else CASES
    algorithm = read(release)
    release_sha = digest(release)
    traces = read(previous / "RECEIPT.json")
    assert traces["source_lock"] == release_sha
    assert all(digest(release.parent / "source" / k) == v for k, v in algorithm["code"].items())
    clean, fault, tests, panels, selection, score_hashes = [], [], [], {}, {}, {}
    variances = {}
    for case in cases:
        base = OUT / case
        result = read(base / "RESULTS.json")
        archive = previous / (f"{case}_AGGREGATES.json" if case.startswith("Rod") else f"{case}/RESULTS.json")
        old_sha = traces["dataset_result_sha256"][case]
        assert digest(archive) == old_sha
        old = read(archive)
        assert result["commit"] == algorithm["commit"]
        assert result["clean_count"] == result["panels"]["clean"]["count"]
        variances[case] = result["tail_variance"]
        assert variances[case] > 0.
        selection[case] = validate_fit(base, case, old_sha, release_sha, result)
        score_hashes[case] = digest(base / "RESULTS.json")
        panels[case] = {}
        for scenario, panel in result["panels"].items():
            historical = old["panels"][scenario]
            assert panel["issue_sha256"] == historical["issue_sha256"]
            assert set(panel["metrics"]) == set(METHODS)
            assert set(panel["dm"]) == set(METHODS) - {"PRISM"}
            assert all(arm == historical["metrics"][m] for m, arm in panel["metrics"].items()
                       if case.startswith("Rod") or m not in ("DPLS", "GRU"))
            fixed = ("n", "bartlett_lag", "mean_loss_gain", "t", "log_p")
            assert all(item[k] == historical["dm"][m][k]
                       for m, item in panel["dm"].items()
                       if case.startswith("Rod") or m not in ("DPLS", "GRU") for k in fixed)
            compare = sorted(panel["dm"].values(), key=lambda row: row["log_p"])
            adjusted = -math.inf
            for i, item in enumerate(compare):
                adjusted = max(adjusted, min(0., item["log_p"] + math.log(len(compare) - i)))
                assert math.isclose(adjusted / math.log(10), item["log10_p_holm"], abs_tol=1e-10)
            rows = panels[case][scenario] = {}
            for method in METHODS:
                arm = panel["metrics"][method]
                metric = dict(rmse=arm["rmse"], mae=arm["mae"])
                if scenario == "clean":
                    # Both RMSE squared and the population target variance contain 1/n.
                    marker = "" if method == "PRISM" else stars(panel["dm"][method])
                    metric.update(r2=1. - arm["rmse"]**2 / variances[case], rmse_dm_stars=marker)
                    clean.append(dict(dataset=case, method=method, **metric))
                else:
                    assert panel["count"] == 512
                    metric["output_shift_mae"] = arm["shift_mae"]
                    fault.append(dict(dataset=case, condition=scenario, method=method, **metric))
                rows[method] = metric
            for method, item in panel["dm"].items():
                tests.append(dict(dataset=case, condition=scenario, baseline=method, n=item["n"],
                                  bartlett_lag=item["bartlett_lag"], mean_loss_gain=item["mean_loss_gain"],
                                  dm_t=item["t"], p_holm=item["p_holm"] or None,
                                  log10_p_holm=item["log10_p_holm"]))
    assert [len(table) for table in (clean, fault, tests)] == [len(cases)*n for n in (10,20,27)]
    target = OUT / "cz_tables" if subset == "cz" else OUT
    target.mkdir(exist_ok=True)
    tables = {"CLEAN_THREE_METRICS.csv": clean, "FAULT_THREE_METRICS.csv": fault, "DM_HAC_HOLM.csv": tests}
    for name, entries in tables.items():
        write_csv(target / name, entries)
    tex_name = "IEEE_TABLES.tex"
    (target / tex_name).write_text(render(panels, cases), encoding="utf-8", newline="\n")
    receipt = dict(commit=algorithm["commit"], source_lock=release_sha,
                   parent_receipt=digest(previous / "RECEIPT.json"), selection=selection,
                   tail_population_variances=variances, case_results=score_hashes,
                   tables={name: digest(target / name) for name in (*tables, tex_name)})
    (target / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8", newline="\n")
    print(f"Verified and exported {len(clean)} clean rows, {len(fault)} fault rows, "
          f"and {len(tests)} paired tests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--subset", choices=("cz", "all"), default="all")
    a = parser.parse_args()
    main(a.release, a.previous, a.subset)
