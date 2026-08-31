from __future__ import annotations

"""Diagnostic-only SRU v2.2 wrapper: best-mean C and best-mean guarded A.

C uses the already-created best-mean diagnostic intervention. W remains exactly
as selected by the current guarded local selector. Only A changes: among the
registered candidates, choose the minimum-mean OOF candidate if it passes the
same practical activation guard against A exact-zero; otherwise retain neutral.

This isolates whether the apparent W->A degradation comes from model overlap or
from A's own one-SE complexity preference.
"""

from dataclasses import replace

import run_prism_v22_sru_c_best_alpha_diagnostic  # installs strict K + best-mean C
import run_prism_v22_sru_full as base


_original_select_a = base._select_a
_original_guarded = base.guarded_local_one_se_select


def _best_mean_guarded(*args, **kwargs):
    result = _original_guarded(*args, **kwargs)
    neutral = kwargs["neutral"]
    best = result.best_candidate
    selected = neutral
    if best != neutral:
        audit = result.activation_audit.get(str(best))
        if audit is None:
            audit = result.activation_audit.get(best)
        if audit and bool(audit.get("pass", False)):
            selected = best
    losses = args[0] if args else kwargs["fold_losses"]
    return replace(
        result,
        final_selected_candidate=selected,
        final_selected_fold_losses=tuple(float(v) for v in losses[selected]),
    )


def _select_a_best_mean(w_oof, config):
    prior = base.guarded_local_one_se_select
    base.guarded_local_one_se_select = _best_mean_guarded
    try:
        return _original_select_a(w_oof, config)
    finally:
        base.guarded_local_one_se_select = prior


base._select_a = _select_a_best_mean


if __name__ == "__main__":
    raise SystemExit(base.main())
