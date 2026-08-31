from __future__ import annotations

"""Diagnostic-only SRU v2.2: best-mean C, W identity, best-mean guarded A.

This is the matched counterpart to the best-mean-A-with-W diagnostic. It asks
whether W still adds value when A receives the same best-mean selection rule.
"""

from dataclasses import replace

import run_prism_v22_sru_c_best_alpha_w_identity_diagnostic  # strict K + best C + no W
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
