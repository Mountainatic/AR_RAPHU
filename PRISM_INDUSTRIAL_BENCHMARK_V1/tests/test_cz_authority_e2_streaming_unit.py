from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_cz_authority_e2_streaming_unit.py"
)
SPEC = importlib.util.spec_from_file_location("cz_e2_streaming_unit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_authority_record_classifier_distinguishes_retained_failure() -> None:
    assert MODULE._classify_authority_records([{"status": "PASS"}] * 8) == "PASS"
    retained = [{"status": "PASS"}] * 7 + [
        {"status": "SOLVER_FAILED_RETAINED"}
    ]
    assert (
        MODULE._classify_authority_records(retained)
        == "COMPLETED_WITH_RETAINED_FAILURES"
    )


def test_authority_record_classifier_rejects_unknown_failure() -> None:
    records = [{"status": "PASS"}] * 7 + [{"status": "UNEXPECTED_FAILURE"}]
    assert MODULE._classify_authority_records(records) == "FAILED"
