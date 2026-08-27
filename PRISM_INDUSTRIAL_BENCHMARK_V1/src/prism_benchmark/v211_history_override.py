from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TEP_CPU_HISTORY_EXTENSION_PROTOCOL_ID = "TEP_CPU_HISTORY_EXTENSION_L256_V1"
TEP_CPU_HISTORY_EXTENSION_STATUS = "FROZEN_BY_USER_BEFORE_FORMAL_DEVELOPMENT"
TEP_DATASET = "tep"
TEP_TASK = "TEP_G_REP_H1"
TEP_HEAD = "TEP_G_REP_H1__H1__W2"
FROZEN_HISTORY_STEPS = (128, 256)
FROZEN_COMMON_SUPPORT_HISTORY_STEPS = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_positive_ints(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{field} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RuntimeError(f"{field} must contain only JSON integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise RuntimeError(f"{field} must contain only positive values")
    if tuple(sorted(set(result))) != result:
        raise RuntimeError(f"{field} must be strictly increasing and unique")
    return result


@dataclass(frozen=True)
class TEPHistoryOverride:
    config_path: Path
    config_sha256: str
    history_steps: tuple[int, ...]
    common_support_history_steps: int
    dpls_maximum_lags_per_channel: int
    dpls_maximum_joint_configurations: int
    hammerstein_profile_cap: int
    state_delta_steps: tuple[int, ...]
    fail_if_history_unavailable: tuple[int, ...]

    @property
    def positive_h_history_multipliers(self) -> tuple[int, ...]:
        """The registered TEP head has h=1, so steps equal multipliers."""

        return self.history_steps

    def require_view(self, view: Any) -> None:
        head = view.head
        if str(head.dataset) != TEP_DATASET:
            raise RuntimeError("TEP history override cannot be applied outside TEP")
        if str(head.task_id) != TEP_TASK or str(head.head_id) != TEP_HEAD:
            raise RuntimeError("TEP history override cannot be applied to an unregistered task/head")
        if int(head.h_steps) != 1 or int(head.w_steps) != 2:
            raise RuntimeError("TEP history override requires the registered H1/W2 head")

    def audit(self) -> dict[str, Any]:
        return {
            "protocol_id": TEP_CPU_HISTORY_EXTENSION_PROTOCOL_ID,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "history_steps": list(self.history_steps),
            "common_support_history_steps": int(
                self.common_support_history_steps
            ),
            "dpls_maximum_lags_per_channel": int(
                self.dpls_maximum_lags_per_channel
            ),
            "dpls_maximum_joint_configurations": int(
                self.dpls_maximum_joint_configurations
            ),
            "hammerstein_profile_cap": int(self.hammerstein_profile_cap),
            "state_delta_steps": list(self.state_delta_steps),
            "fail_if_history_unavailable": list(
                self.fail_if_history_unavailable
            ),
        }


def load_tep_history_override(
    config_path: Path | str | None,
) -> TEPHistoryOverride | None:
    """Load the explicit TEP-only extension contract.

    ``None`` intentionally means the published contracts are untouched.  A
    non-``None`` path must be absolute so a worker cannot resolve a different
    scientific contract after its working directory changes.
    """

    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        raise RuntimeError("TEP history override config path must be absolute")
    if not path.is_file():
        raise RuntimeError(f"TEP history override config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("TEP history override config must be a JSON object")
    if payload.get("protocol_id") != TEP_CPU_HISTORY_EXTENSION_PROTOCOL_ID:
        raise RuntimeError("TEP history override protocol_id mismatch")
    if payload.get("status") != TEP_CPU_HISTORY_EXTENSION_STATUS:
        raise RuntimeError("TEP history override is not frozen")
    if payload.get("active_datasets") != [TEP_DATASET]:
        raise RuntimeError("TEP history override must be restricted to TEP")
    if payload.get("active_tasks") != [TEP_TASK]:
        raise RuntimeError("TEP history override task binding mismatch")
    if payload.get("active_heads") != [TEP_HEAD]:
        raise RuntimeError("TEP history override head binding mismatch")
    histories = _strict_positive_ints(
        payload.get("history_aware_steps"), "history_aware_steps"
    )
    if histories != FROZEN_HISTORY_STEPS:
        raise RuntimeError("TEP history override changed the frozen history grid")
    common_support = payload.get("common_support_history_steps")
    if (
        isinstance(common_support, bool)
        or not isinstance(common_support, int)
        or int(common_support) != FROZEN_COMMON_SUPPORT_HISTORY_STEPS
        or int(common_support) != max(histories)
    ):
        raise RuntimeError("TEP history override common support must equal L256")
    methods = payload.get("history_method_overrides")
    if not isinstance(methods, Mapping):
        raise RuntimeError("history_method_overrides must be a JSON object")
    dpls_maximum_lags = methods.get("dpls_maximum_lags_per_channel")
    dpls_maximum_joint_configurations = methods.get(
        "dpls_maximum_joint_configurations"
    )
    profile_cap = methods.get("hammerstein_profile_cap")
    if (
        isinstance(dpls_maximum_lags, bool)
        or not isinstance(dpls_maximum_lags, int)
        or dpls_maximum_lags != 16
    ):
        raise RuntimeError("DPLS maximum lags per channel must remain frozen at 16")
    if (
        isinstance(dpls_maximum_joint_configurations, bool)
        or not isinstance(dpls_maximum_joint_configurations, int)
        or dpls_maximum_joint_configurations != 16
    ):
        raise RuntimeError("DPLS maximum joint configurations must equal 16")
    if (
        isinstance(profile_cap, bool)
        or not isinstance(profile_cap, int)
        or profile_cap < len(histories)
    ):
        raise RuntimeError("Hammerstein profile cap must cover every history")
    if methods.get("hammerstein_profile_cap_must_cover_every_registered_history") is not True:
        raise RuntimeError("Hammerstein history coverage guard must be enabled")
    if methods.get("hammerstein_profile_strategy") != (
        "REGISTERED_HISTORIES_ONLY_WITH_STATE_DELTAS"
    ):
        raise RuntimeError("Hammerstein profile extension strategy changed")
    if methods.get("hammerstein_maximum_configurations_per_method") != 24:
        raise RuntimeError("Hammerstein maximum configuration count must equal 24")
    state_deltas = _strict_positive_ints(
        methods.get("state_delta_steps"), "state_delta_steps"
    )
    if state_deltas != (1, 2, 4):
        raise RuntimeError("TEP state delta grid changed")
    unavailable = _strict_positive_ints(
        methods.get("fail_if_history_unavailable"),
        "fail_if_history_unavailable",
    )
    if unavailable != (128, 256):
        raise RuntimeError("TEP unavailable-history fail guard changed")
    channel_classes = methods.get("k_channel_class")
    if channel_classes != {"xmv": "FAST", "xmeas": "MEDIUM"}:
        raise RuntimeError("TEP K channel-class binding changed")
    return TEPHistoryOverride(
        config_path=path.resolve(),
        config_sha256=_sha256_file(path),
        history_steps=histories,
        common_support_history_steps=int(common_support),
        dpls_maximum_lags_per_channel=int(dpls_maximum_lags),
        dpls_maximum_joint_configurations=int(
            dpls_maximum_joint_configurations
        ),
        hammerstein_profile_cap=int(profile_cap),
        state_delta_steps=state_deltas,
        fail_if_history_unavailable=unavailable,
    )


def validate_positive_h_history_multipliers(
    values: Sequence[int],
) -> tuple[int, ...]:
    """Validate an in-memory override used by low-level profile unit tests."""

    raw = list(values)
    return _strict_positive_ints(raw, "positive_h_history_multipliers")
