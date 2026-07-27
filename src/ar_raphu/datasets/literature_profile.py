"""Validation helpers for pinned external PB1 baseline profiles."""

from __future__ import annotations

from typing import Any


def literature_profile_gaps(
    profile: dict[str, Any],
    *,
    observed_refs: set[str],
    observed_file_sha256: dict[str, str],
) -> list[str]:
    gaps: list[str] = []
    paper = profile.get("paper", {})
    code = profile.get("companion_code", {})
    commit = str(code.get("commit", ""))
    if profile.get("profile_status") != "FROZEN":
        gaps.append("PROFILE_NOT_FROZEN")
    if paper.get("doi") != "10.1016/j.ifacol.2024.08.574":
        gaps.append("PAPER_DOI_MISMATCH")
    if paper.get("arxiv") != "2405.10779v2":
        gaps.append("PAPER_ARXIV_VERSION_MISMATCH")
    if len(commit) != 40 or commit not in observed_refs:
        gaps.append("EXTERNAL_BASELINE_COMMIT_UNPINNED")
    for path, expected in code.get("file_sha256", {}).items():
        if observed_file_sha256.get(path) != expected:
            gaps.append(f"EXTERNAL_BASELINE_FILE_HASH_MISMATCH:{path}")
    if not code.get("file_sha256"):
        gaps.append("EXTERNAL_BASELINE_FILE_HASHES_MISSING")
    return sorted(gaps)
