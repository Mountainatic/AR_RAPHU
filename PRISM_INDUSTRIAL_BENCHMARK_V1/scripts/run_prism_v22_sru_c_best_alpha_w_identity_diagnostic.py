from __future__ import annotations

"""Diagnostic-only SRU v2.2 wrapper: best-mean C alpha with W forced identity.

Purpose: separate K/Gamma -> A interaction from W -> A interaction after the
C over-regularization failure was identified.  The strict K admission and the
best-mean C-alpha diagnostic are inherited unchanged.  Only the W candidate
set is forced to the identity correction.
"""

import run_prism_v22_sru_c_best_alpha_diagnostic  # noqa: F401
import run_prism_v22_sru_full as base


def identity_only_w_candidates(config: dict, direction: int, monotone_allowed: bool):
    return [base.IDENTITY]


base._w_candidates = identity_only_w_candidates


if __name__ == "__main__":
    raise SystemExit(base.main())
