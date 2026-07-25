"""Truth-blind selection and post-selection recovery metrics for M2."""
from __future__ import annotations

import itertools
import numpy as np


def pairwise_jaccard(supports):
    values = []
    for left, right in itertools.combinations(map(set, supports), 2):
        values.append(len(left & right) / max(1, len(left | right)))
    return float(np.mean(values)) if values else 1.0


def support_scores(supports, truth=(0, 1, 2)):
    target = set(truth); rows = []
    for support in supports:
        chosen = set(support); tp = len(chosen & target)
        precision = tp / max(1, len(chosen)); recall = tp / len(target)
        rows.append({"active_count": len(chosen), "precision": precision,
                     "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall),
                     "exact": chosen == target, "false_positives": sorted(chosen-target),
                     "false_negatives": sorted(target-chosen)})
    return rows


def one_se_select(rows):
    """Validation-only one-standard-error rule."""
    groups = {}
    for row in rows: groups.setdefault(str(row["config_id"]), []).append(row)
    stats = []
    for config_id, records in groups.items():
        values = np.asarray([float(r["refit_val_rmse"]) for r in records])
        stats.append({"config_id": config_id, "mean": float(values.mean()),
                      "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values)>1 else 0.,
                      "count": float(np.mean([r["active_count"] for r in records])),
                      "jaccard": pairwise_jaccard([r["terminal_support"] for r in records]),
                      "s0": float(records[0]["s0"])})
    minimum = min(stats, key=lambda x: x["mean"])
    eligible = [x for x in stats if x["mean"] <= minimum["mean"] + minimum["se"]]
    return sorted(eligible, key=lambda x: (x["count"], -x["jaccard"], -x["s0"]))[0]


def support_aware_delay(learned_q, true_q, supports, variables=(0, 1, 2)):
    """Penalize every missed true variable instead of silently dropping it."""
    learned_q=np.asarray(learned_q); true_q=np.asarray(true_q); L=true_q.shape[-1]
    uniform=np.full(L,1/L); tau=np.arange(L)
    uniform_w1s={j:float(np.abs(np.cumsum(uniform)-np.cumsum(true_q[j])).sum()) for j in variables}
    rows=[]
    for seed, support in enumerate(supports):
        for j in variables:
            if j not in support:
                rows.append({"seed":seed,"variable":j,"selected":False,"metric_scope":"support_aware",
                             "mean_delay_mae":L-1,"std_delay_mae":L-1,"w1":uniform_w1s[j],
                             "peak_error":L-1,"boundary_mass":1.,"q_correlation":0.})
            else:
                q=learned_q[seed,j]; h=true_q[j]
                qm=float(q@tau);hm=float(h@tau);qs=float(np.sqrt(q@(tau-qm)**2));hs=float(np.sqrt(h@(tau-hm)**2))
                rows.append({"seed":seed,"variable":j,"selected":True,"metric_scope":"support_aware",
                    "mean_delay_mae":abs(float(q@tau-h@tau)),
                    "std_delay_mae":abs(qs-hs),
                    "w1":float(np.abs(np.cumsum(q)-np.cumsum(h)).sum()),
                    "peak_error":abs(int(q.argmax())-int(h.argmax())),
                    "boundary_mass":float(q[-3:].sum()),
                    "q_correlation":float(np.corrcoef(q,h)[0,1]) if q.std()>1e-12 and h.std()>1e-12 else 0.})
    return rows, float(np.mean(list(uniform_w1s.values())))
