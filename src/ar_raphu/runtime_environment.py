"""Runtime-environment gate for reproducible local and AutoDL execution."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


def detect_runtime_manager(
    environ: Mapping[str, str] | None = None,
    *,
    prefix: str | None = None,
    base_prefix: str | None = None,
) -> str | None:
    """Return the authorized runtime manager, or ``None`` when unauthorized."""

    env = os.environ if environ is None else environ
    if env.get("CONDA_DEFAULT_ENV") == "Env_pytorch":
        return "conda:Env_pytorch"
    if env.get("AR_RAPHU_RUNTIME_MANAGER") == "uv":
        active_prefix = sys.prefix if prefix is None else prefix
        system_prefix = sys.base_prefix if base_prefix is None else base_prefix
        if active_prefix != system_prefix:
            return "uv"
    return None


def require_runtime_environment() -> str:
    """Require either the frozen local Conda env or explicit AutoDL uv env."""

    manager = detect_runtime_manager()
    if manager is None:
        raise RuntimeError(
            "Run in Conda Env_pytorch, or in an activated uv virtual "
            "environment with AR_RAPHU_RUNTIME_MANAGER=uv."
        )
    return manager
