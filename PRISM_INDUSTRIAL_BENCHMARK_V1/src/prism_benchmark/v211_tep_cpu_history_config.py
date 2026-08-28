from __future__ import annotations

from pathlib import Path
from typing import Any

from .v211_history_override import (
    TEP_CPU_HISTORY_EXTENSION_PROTOCOL_ID,
    TEP_NOWCAST_PROTOCOL_ID,
    load_tep_history_override,
)
from .v211_metro_config import load_metro_config


PROTOCOL = "tep_cpu_history_l256"
CONFIG_RELATIVE_PATH = Path("configs/tep_cpu_history_extension_20260826.json")
NOWCAST_CONFIG_RELATIVE_PATH = Path(
    "configs/tep_cpu_nowcast_history_extension_20260828.json"
)


def load_tep_cpu_history_algorithm_config(project: Path) -> dict[str, Any]:
    """Load the isolated TEP extension without mutating older frozen protocols.

    The published representative descriptor binds hashes of the original K/A/J
    implementation.  This extension intentionally uses a distinct protocol ID
    and seals its implementation through the run's git commit and input
    manifest, so older representative runs continue to validate unchanged.
    """

    override = load_tep_history_override(
        (project / CONFIG_RELATIVE_PATH).resolve()
    )
    if override is None:  # pragma: no cover - the path above is explicit
        raise RuntimeError("TEP history override unexpectedly disabled")
    inherited = load_metro_config(project)
    merged = dict(inherited)
    merged.update(
        {
            "protocol_id": TEP_CPU_HISTORY_EXTENSION_PROTOCOL_ID,
            "evidence_class": "TEP_ONLY_CPU_HISTORY_EXTENSION_DEVELOPMENT",
            "active_datasets": ["tep"],
            "active_heads": ["TEP_G_REP_H1__H1__W2"],
            "history_override_config_sha256": override.config_sha256,
            "history_aware_steps": list(override.history_steps),
            "common_support_history_steps": int(
                override.common_support_history_steps
            ),
            "write_shared_data": False,
            "rebuild_or_resplit_c1": False,
            "test_access_before_freeze": False,
            "ood_access_before_freeze": False,
        }
    )
    merged["resource"] = {
        **inherited["resource"],
        "workers": 1,
        "blas_threads": 1,
    }
    return merged


def load_tep_cpu_nowcast_history_algorithm_config(project: Path) -> dict[str, Any]:
    override = load_tep_history_override(
        (project / NOWCAST_CONFIG_RELATIVE_PATH).resolve()
    )
    if override is None:  # pragma: no cover
        raise RuntimeError("TEP nowcast history override unexpectedly disabled")
    inherited = load_metro_config(project)
    merged = dict(inherited)
    merged.update(
        {
            "protocol_id": TEP_NOWCAST_PROTOCOL_ID,
            "evidence_class": "TEP_ONLY_CPU_STRICT_PAST_NOWCAST_DEVELOPMENT",
            "active_datasets": ["tep"],
            "active_heads": [override.head_id],
            "history_override_config_sha256": override.config_sha256,
            "history_aware_steps": list(override.history_steps),
            "common_support_history_steps": int(
                override.common_support_history_steps
            ),
            "write_shared_data": False,
            "rebuild_or_resplit_c1": False,
            "test_access_before_freeze": False,
            "ood_access_before_freeze": False,
        }
    )
    merged["resource"] = {
        **inherited["resource"],
        "workers": 1,
        "blas_threads": 1,
    }
    return merged
