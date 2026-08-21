import hashlib
import zipfile

import pandas as pd

from prism_benchmark.public5_level_r2_reporting import (
    PACKAGE_ROOT_NAME,
    add_dual_rankings,
    select_frozen_comparison_rows,
    verify_zip_checksums,
)


def _row(model: str, level_r2: float, skill: float, frozen_rank: int) -> dict:
    return {
        "target_head": "HEAD",
        "information_set": "dynamic",
        "availability_scenario": "record_time",
        "proxy_policy": "primary",
        "split": "test",
        "view_role": "primary",
        "model": model,
        "model_source": "PRISM" if model.startswith("PRISM") else "CPU_BASELINE",
        "reconstruction_status": "PASS",
        "r2_level_reconstructed": level_r2,
        "persistence_skill": skill,
        "frozen_delta_rank": frozen_rank,
    }


def test_dual_rankings_preserve_disagreement() -> None:
    frame = pd.DataFrame(
        [
            _row("RIDGE", 0.90, 0.10, 2),
            _row("XGBOOST", 0.80, 0.20, 1),
        ]
    )

    ranked = add_dual_rankings(frame).set_index("model")

    assert ranked.loc["RIDGE", "rank_by_level_r2"] == 1
    assert ranked.loc["RIDGE", "rank_by_persistence_skill"] == 2
    assert ranked.loc["XGBOOST", "rank_by_level_r2"] == 2
    assert ranked.loc["XGBOOST", "rank_by_persistence_skill"] == 1


def test_plot_selection_uses_original_frozen_rank_not_level_r2() -> None:
    frame = pd.DataFrame(
        [
            _row("PRISM_V2_1_1_PHYSICS_FIRST", 0.75, 0.30, 3),
            _row("PRISM_V2_1_1_JOINT_KWA", 0.78, 0.32, 2),
            _row("RIDGE", 0.99, 0.15, 2),
            _row("XGBOOST", 0.80, 0.20, 1),
            _row("PERSISTENCE", 0.70, 0.00, 5),
        ]
    )

    selected = select_frozen_comparison_rows(frame, "HEAD", "dynamic")

    assert "XGBOOST" in set(selected["model"])
    assert "RIDGE" not in set(selected["model"])
    assert "PERSISTENCE" in set(selected["model"])


def test_zip_checksum_verification_reads_packaged_members(tmp_path) -> None:
    payload = b"frozen-report\n"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "report.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{PACKAGE_ROOT_NAME}/results/result.csv", payload)
        archive.writestr(
            f"{PACKAGE_ROOT_NAME}/SHA256SUMS.txt",
            f"{digest}  results/result.csv\n",
        )

    verify_zip_checksums(path)
