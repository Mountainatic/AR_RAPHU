from __future__ import annotations

"""Audited PRISM v2.2 SRU full-KWA runner with corrected inherited C semantics.

This runner deliberately reuses the already-audited diagnostic implementation
that restores the frozen v2.1.1 C contract:

* C ridge is NUMERICAL_STABILITY_ONLY;
* choose the smallest registered ridge passing numerical certificates;
* require the inherited OOF input-path-preservation gate;
* if compressed C collapses, fall back to BEST_ACTIVE_K_CHANNEL instead of
  rejecting the whole temporal branch.

Strict K numerical admission, Gamma_CT, W, A, SRU split, target transform and
no-test-selection behavior remain unchanged.

Important publication status: the SRU lockbox had already been accessed before
this implementation correction was identified.  Results from this runner are
therefore implementation-correction/descriptive evidence, not a clean new
confirmatory lockbox result.
"""

import argparse
import json
from pathlib import Path

# Import installs the strict K hooks plus the v2.1.1-compatible C hooks.
import run_prism_v22_sru_c_v211_contract_diagnostic  # noqa: F401
import run_prism_v22_sru_full as base


CORRECTION_ID = "PRISM_V2_2_SRU_C_CONTRACT_CORRECTION_20260831_V1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PRISM v2.2 SRU full KWA with corrected inherited C semantics"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = base.run(
        args.data.resolve(), args.config.resolve(), args.output.resolve()
    )
    result["implementation_correction"] = {
        "correction_id": CORRECTION_ID,
        "status": "POST_LOCKBOX_IMPLEMENTATION_CORRECTION_DESCRIPTIVE",
        "corrected_component": "C",
        "inherited_semantics": {
            "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
            "ridge_selection": "SMALLEST_STABLE_REGISTERED_ALPHA",
            "input_path_gate": {
                "minimum_variance_ratio_to_target": 1e-8,
                "minimum_fraction_of_best_active_k_variance_ratio": 0.10,
                "maximum_mse_ratio_vs_best_active_k": 1.02,
                "minimum_nonintercept_coefficient_abs": 1e-10,
            },
            "fallback": "BEST_ACTIVE_K_CHANNEL",
        },
        "best_mean_c_alpha_diagnostic_adopted": False,
        "thresholds_tuned_from_test": False,
        "test_lockbox_previously_accessed_before_correction": True,
        "confirmatory_use": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": result["status"],
                "correction_id": CORRECTION_ID,
                "publication_status": result["implementation_correction"]["status"],
                "nested_routes": result["nested_routes"],
                "gamma_weights": result["gamma_ct"]["weights"],
                "W_selected": result["W"]["selected"],
                "A_selected": result["A"]["selected"],
                "test_target_used_for_selection": result["test_target_used_for_selection"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
