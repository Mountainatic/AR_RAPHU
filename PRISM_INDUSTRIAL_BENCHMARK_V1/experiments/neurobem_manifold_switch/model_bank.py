from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .monitor import ManifoldTemplate
from .prism_adapter import FrozenPrismAdapter


@dataclass
class BankEntry:
    adapter: FrozenPrismAdapter
    template: ManifoldTemplate
    acceptance_threshold: float


class ModelBank:
    def __init__(self, entries: list[BankEntry]):
        if len(entries) < 2:
            raise ValueError("MODEL_BANK_REQUIRES_AT_LEAST_TWO_LOCAL_MODELS")
        self.entries = entries

    def choose(self, feature: np.ndarray) -> tuple[BankEntry | None, dict[str, float]]:
        scores = {e.adapter.model_id: float(e.template.projection_score(feature[None])[0]) for e in self.entries}
        selected = min(self.entries, key=lambda e: scores[e.adapter.model_id] / max(e.acceptance_threshold, 1e-12))
        accepted = scores[selected.adapter.model_id] <= selected.acceptance_threshold
        return (selected if accepted else None), scores

    def add(self, entry: BankEntry) -> None:
        self.entries.append(entry)

