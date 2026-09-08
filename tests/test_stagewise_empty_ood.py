from __future__ import annotations

import pandas as pd

from prism_benchmark.v211_support import SUPPORT_CONTRACT, require_native_support_contract


def test_empty_ood_has_no_observable_contract_value() -> None:
    frame = pd.DataFrame({"sample_support_contract": pd.Series(dtype="object")})
    # This documents why the metadata freezer must branch before applying the
    # strict non-empty support validator.
    try:
        require_native_support_contract(frame)
    except RuntimeError as error:
        assert "observed support contracts []" in str(error)
    else:
        raise AssertionError("empty support must not masquerade as a validated non-empty split")
    assert frame.empty
    assert SUPPORT_CONTRACT == "NATIVE_K_COMMON_ASSEMBLY_R1"
