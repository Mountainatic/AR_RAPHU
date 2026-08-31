from __future__ import annotations

"""Diagnostic-only SRU v2.2 wrapper: choose C ridge by best mean OOF loss.

This is NOT a publication candidate and does not change the frozen benchmark
contract.  It isolates one suspected failure mode in the current strict SRU
adapter: the ordinary one-SE C selector prefers the largest ridge alpha inside
the one-SE set, which can collapse an otherwise active K input path and reject
the whole D/M/S branch.

Everything except C ridge selection is inherited unchanged from the strict
runner, including strict K numerical admission, persistence fallback, Gamma_CT,
DeltaW, A, train/test boundary, and no-test-selection rule.
"""

import math
from typing import Any, Callable, Hashable, Mapping

import numpy as np

# Importing the strict wrapper first installs its K-channel numerical admission
# and persistence-only branch fallback hooks into the base module.
import run_prism_v22_sru_full_strict  # noqa: F401
import run_prism_v22_sru_full as base
from prism_benchmark.v2_selection import OneSESelection


def best_mean_c_select(
    fold_losses: Mapping[Hashable, list[float]],
    complexity_key: Callable[[Hashable], tuple[Any, ...]],
    *,
    neutral: Hashable | None = None,
    minimum_usable_folds: int = 3,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> OneSESelection:
    """Select the minimum mean validation loss; diagnostic replacement for C only."""
    if neutral is not None:
        raise RuntimeError("best-mean diagnostic selector was invoked outside C")
    usable: dict[Hashable, np.ndarray] = {}
    for candidate, values in fold_losses.items():
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) >= int(minimum_usable_folds):
            usable[candidate] = array
    if not usable:
        raise ValueError("no C alpha has the minimum number of finite folds")
    means = {
        candidate: float(np.mean(values, dtype=np.float64))
        for candidate, values in usable.items()
    }
    errors = {
        candidate: float(np.std(values, ddof=1) / math.sqrt(len(values)))
        for candidate, values in usable.items()
    }
    best = min(usable, key=lambda candidate: (means[candidate], complexity_key(candidate)))
    threshold = means[best] + errors[best]
    acceptable = tuple(
        candidate
        for candidate in usable
        if means[candidate] < threshold
        or np.isclose(means[candidate], threshold, rtol=rtol, atol=atol)
    )
    # The diagnostic intervention is exactly this line: retain the predictive
    # minimum instead of regularizing all the way to the simplest one-SE member.
    selected = best
    return OneSESelection(
        selected=selected,
        best=best,
        best_mean=means[best],
        best_se=errors[best],
        threshold=threshold,
        acceptable=acceptable,
        usable_fold_count={candidate: len(values) for candidate, values in usable.items()},
        means=means,
        standard_errors=errors,
    )


base.one_se_select = best_mean_c_select


if __name__ == "__main__":
    raise SystemExit(base.main())
