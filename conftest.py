from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
STAGE1_BUNDLE_ROOT = PROJECT_ROOT / "STAGE1_DUAL_SOLVER_V20_bundle"


@pytest.fixture(autouse=True)
def use_bundle_working_directory(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    test_path = Path(str(request.node.fspath)).resolve()
    if STAGE1_BUNDLE_ROOT in test_path.parents:
        monkeypatch.chdir(STAGE1_BUNDLE_ROOT)
