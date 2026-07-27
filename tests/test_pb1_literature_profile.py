from __future__ import annotations

from ar_raphu.datasets.literature_profile import literature_profile_gaps


def _profile() -> dict:
    return {
        "profile_status": "FROZEN",
        "paper": {
            "doi": "10.1016/j.ifacol.2024.08.574",
            "arxiv": "2405.10779v2",
        },
        "companion_code": {
            "commit": "d9c4972f6bd59cfdde23a014bc57579e2af957aa",
            "file_sha256": {"models.py": "expected"},
        },
    }


def test_pinned_literature_profile_passes_exact_evidence() -> None:
    profile = _profile()
    assert literature_profile_gaps(
        profile,
        observed_refs={profile["companion_code"]["commit"]},
        observed_file_sha256={"models.py": "expected"},
    ) == []


def test_code_drift_is_a_hard_profile_gap() -> None:
    profile = _profile()
    gaps = literature_profile_gaps(
        profile,
        observed_refs=set(),
        observed_file_sha256={"models.py": "changed"},
    )
    assert "EXTERNAL_BASELINE_COMMIT_UNPINNED" in gaps
    assert "EXTERNAL_BASELINE_FILE_HASH_MISMATCH:models.py" in gaps
