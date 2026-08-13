from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from prism_benchmark.neurobem_literature import LiteratureTrajectory
from .prism_adapter import FrozenPrismAdapter, fit_local_adapter, geometry_features
from .monitor import ManifoldTemplate
from .model_bank import BankEntry


@dataclass(frozen=True)
class ReidDecision:
    promoted: bool
    gain: float
    samples: int
    entry: BankEntry | None


def causal_reidentify(frame, arrived_end: int, alarm_index: int, route: str, incumbent: FrozenPrismAdapter, config: dict) -> ReidDecision:
    history = int(config["history"])
    fit_rows = int(config["reid_fit_samples"])
    valid_rows = int(config["reid_validation_samples"])
    required = fit_rows + valid_rows
    start = max(0, alarm_index - history + 1)
    if arrived_end - alarm_index + 1 < required:
        return ReidDecision(False, 0.0, arrived_end - alarm_index + 1, None)
    available = frame.iloc[start:arrived_end + 1].reset_index(drop=True)
    cut = min(len(available) - valid_rows, history + fit_rows)
    fit_frame, valid_frame = available.iloc[:cut].reset_index(drop=True), available.iloc[cut-history:].reset_index(drop=True)
    trajectory = LiteratureTrajectory("CAUSAL_POST_ALARM", "online", fit_frame)
    candidate = fit_local_adapter([trajectory], f"REID_{alarm_index}", config["w_family"], fit_rows, config["ridge_grid"], config["max_condition"], config["max_kkt"], history)
    y, old, _ = incumbent.one_step(route, valid_frame)
    _, new, _ = candidate.one_step(route, valid_frame)
    old_loss = float(np.mean(np.square(y - old)))
    new_loss = float(np.mean(np.square(y - new)))
    gain = (old_loss - new_loss) / max(old_loss, 1e-12)
    if not np.isfinite(gain) or gain < float(config["reid_promotion_margin"]):
        return ReidDecision(False, gain, required, None)
    features = geometry_features(fit_frame)
    template = ManifoldTemplate.fit(features, int(config["geometry_rank"]))
    threshold = float(np.quantile(template.projection_score(features), config["monitor_quantile"]))
    return ReidDecision(True, gain, required, BankEntry(candidate, template, threshold))

