from __future__ import annotations

from ar_raphu.runtime_environment import detect_runtime_manager


def test_accepts_frozen_conda_environment() -> None:
    assert (
        detect_runtime_manager(
            {"CONDA_DEFAULT_ENV": "Env_pytorch"},
            prefix="/env",
            base_prefix="/base",
        )
        == "conda:Env_pytorch"
    )


def test_accepts_explicit_uv_virtual_environment() -> None:
    assert (
        detect_runtime_manager(
            {"AR_RAPHU_RUNTIME_MANAGER": "uv"},
            prefix="/project/.venv",
            base_prefix="/usr",
        )
        == "uv"
    )


def test_rejects_uv_flag_without_virtual_environment() -> None:
    assert (
        detect_runtime_manager(
            {"AR_RAPHU_RUNTIME_MANAGER": "uv"},
            prefix="/usr",
            base_prefix="/usr",
        )
        is None
    )


def test_rejects_unmanaged_python() -> None:
    assert (
        detect_runtime_manager({}, prefix="/usr", base_prefix="/usr") is None
    )
