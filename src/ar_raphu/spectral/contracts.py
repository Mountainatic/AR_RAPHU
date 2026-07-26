"""Machine-checked scientific target/model contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    scientific_question: str
    target_contains_ar: bool
    model_contains_ar: bool
    target_contains_x: bool
    model_contains_x: bool
    truth_used_for_training: bool
    truth_used_for_evaluation: bool
    support_used_for_training: str
    hyperparameter_selection_metric: str
    rank_inputs_used_for_selection: bool
    test_used_for_selection: bool
    experiment_role: str = "FORMAL_STRUCTURE_RECOVERY"

    def validate(self) -> None:
        if not self.scientific_question.strip():
            raise ValueError("scientific_question must be non-empty.")
        if self.truth_used_for_training:
            raise ValueError("Truth cannot be used for formal training.")
        if self.support_used_for_training not in {"all", "oracle"}:
            raise ValueError("support_used_for_training must be all or oracle.")
        if self.hyperparameter_selection_metric != "validation_prediction_loss_only":
            raise ValueError("Hyperparameters must use validation prediction loss only.")
        if self.rank_inputs_used_for_selection or self.test_used_for_selection:
            raise ValueError("Rank inputs and test data cannot select hyperparameters.")
        if self.target_contains_ar and not self.model_contains_ar:
            if self.experiment_role != "ORACLE_COMPONENT_DIAGNOSTIC":
                raise ValueError(
                    "An AR-free model cannot formally fit a target containing AR."
                )
        if self.experiment_role == "FORMAL_STRUCTURE_RECOVERY":
            if not self.model_contains_x:
                raise ValueError("Formal structure recovery must contain X.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)
