from __future__ import annotations

import importlib.util
from pathlib import Path


def test_package_forbidden_sets_are_explicit() -> None:
    path = Path(__file__).parents[1] / "scripts/build_cpu_bundle.py"
    spec = importlib.util.spec_from_file_location("build_cpu_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "raw_sources" in module.FORBIDDEN_PARTS
    assert "base_data" in module.FORBIDDEN_PARTS
    assert ".xlsx" in module.FORBIDDEN_SUFFIXES
