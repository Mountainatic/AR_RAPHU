"""FAST-D dense eigensolver ridge-ARX screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg

from ar_raphu.cz_real.linear import TrainScaler, regression_metrics, window_designs

from .residualization import FAST_TASKS, FastTask, build_fast_folds, target_indices


def _symmetric_eigh(
    matrix: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    if device.startswith("cuda"):
        import torch

        tensor = torch.as_tensor(
            np.asarray(matrix, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        values, vectors = torch.linalg.eigh(tensor)
        torch.cuda.synchronize(tensor.device)
        return values.cpu().numpy(), vectors.cpu().numpy()
    return scipy.linalg.eigh(
        np.asarray(matrix, dtype=np.float64),
        check_finite=False,
        driver="evd",
    )


@dataclass(slots=True)
class DenseRidgePath:
    feature_mean: np.ndarray
    target_mean: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_rhs: np.ndarray

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        device: str = "cpu",
    ) -> "DenseRidgePath":
        x = np.asarray(matrix, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if device.startswith("cuda"):
            import torch

            x_tensor = torch.as_tensor(
                x, dtype=torch.float64, device=device
            )
            y_tensor = torch.as_tensor(
                y, dtype=torch.float64, device=device
            )
            x_mean_tensor = x_tensor.mean(dim=0)
            target_mean_tensor = y_tensor.mean()
            centered_x_tensor = x_tensor - x_mean_tensor
            centered_y_tensor = y_tensor - target_mean_tensor
            gram_tensor = (
                centered_x_tensor.T @ centered_x_tensor
            ) / len(centered_x_tensor)
            rhs_tensor = (
                centered_x_tensor.T @ centered_y_tensor
            ) / len(centered_x_tensor)
            values_tensor, vectors_tensor = torch.linalg.eigh(gram_tensor)
            torch.cuda.synchronize(x_tensor.device)
            x_mean = x_mean_tensor.cpu().numpy()
            target_mean = float(target_mean_tensor.cpu())
            values = values_tensor.cpu().numpy()
            vectors = vectors_tensor.cpu().numpy()
            rhs = rhs_tensor.cpu().numpy()
            del (
                x_tensor,
                y_tensor,
                centered_x_tensor,
                centered_y_tensor,
                gram_tensor,
                rhs_tensor,
                values_tensor,
                vectors_tensor,
            )
            torch.cuda.empty_cache()
        else:
            x_mean = x.mean(axis=0)
            target_mean = float(y.mean())
            centered_x = x - x_mean
            centered_y = y - target_mean
            gram = centered_x.T @ centered_x / len(centered_x)
            rhs = centered_x.T @ centered_y / len(centered_x)
            values, vectors = _symmetric_eigh(gram, device=device)
        return cls(
            feature_mean=x_mean,
            target_mean=target_mean,
            eigenvalues=np.maximum(values, 0.0),
            eigenvectors=vectors,
            projected_rhs=vectors.T @ rhs,
        )

    def coefficients(self, alpha: float) -> np.ndarray:
        alpha = float(alpha)
        if alpha == 0.0:
            threshold = max(
                float(self.eigenvalues.max()) * 1.0e-12,
                np.finfo(np.float64).eps,
            )
            inverse = np.where(
                self.eigenvalues > threshold,
                1.0 / np.maximum(self.eigenvalues, threshold),
                0.0,
            )
        else:
            inverse = 1.0 / (self.eigenvalues + alpha)
        return self.eigenvectors @ (inverse * self.projected_rhs)

    def predict(self, matrix: np.ndarray, alpha: float) -> np.ndarray:
        coefficients = self.coefficients(alpha)
        return (
            np.asarray(matrix, dtype=np.float64) - self.feature_mean
        ) @ coefficients + self.target_mean


def _task_fold_designs(
    x: np.ndarray,
    y: np.ndarray,
    task: FastTask,
) -> list[dict[str, object]]:
    rows = []
    for fold in build_fast_folds(len(y), task):
        train_targets = target_indices(
            start=0, stop=fold.effective_train_stop, task=task
        )
        validation_targets = target_indices(
            start=fold.validation_start,
            stop=fold.validation_stop,
            task=task,
        )
        scaler = TrainScaler.fit(x, y, fold.effective_train_stop)
        train_x, train_y_history = window_designs(
            x,
            y,
            targets=train_targets,
            horizon=task.horizon,
            L_x=task.L_x,
            L_y=task.L_y,
            scaler=scaler,
        )
        validation_x, validation_y_history = window_designs(
            x,
            y,
            targets=validation_targets,
            horizon=task.horizon,
            L_x=task.L_x,
            L_y=task.L_y,
            scaler=scaler,
        )
        rows.append(
            {
                "fold": fold,
                "train_targets": train_targets,
                "validation_targets": validation_targets,
                "train_ar": train_y_history,
                "validation_ar": validation_y_history,
                "train_arx": np.column_stack((train_x, train_y_history)),
                "validation_arx": np.column_stack(
                    (validation_x, validation_y_history)
                ),
            }
        )
    return rows


def _select_alpha(
    paths: list[DenseRidgePath],
    validation_matrices: list[np.ndarray],
    validation_targets: list[np.ndarray],
    alphas: tuple[float, ...],
) -> tuple[float, list[float]]:
    losses = []
    for alpha in alphas:
        fold_losses = [
            float(np.mean((path.predict(matrix, alpha) - target) ** 2))
            for path, matrix, target in zip(
                paths,
                validation_matrices,
                validation_targets,
                strict=True,
            )
        ]
        losses.append(float(np.mean(fold_losses)))
    index = min(range(len(alphas)), key=lambda item: (losses[item], item))
    return alphas[index], losses


def linear_increment_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_grid: Iterable[float],
    device: str = "cpu",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    alphas = tuple(float(value) for value in ridge_grid)
    output: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    for task in FAST_TASKS:
        fold_rows = _task_fold_designs(x, y, task)
        ar_paths = [
            DenseRidgePath.fit(
                row["train_ar"], y[row["train_targets"]], device=device
            )
            for row in fold_rows
        ]
        arx_paths = [
            DenseRidgePath.fit(
                row["train_arx"], y[row["train_targets"]], device=device
            )
            for row in fold_rows
        ]
        validation_targets = [
            y[row["validation_targets"]] for row in fold_rows
        ]
        selected_ar, ar_losses = _select_alpha(
            ar_paths,
            [row["validation_ar"] for row in fold_rows],
            validation_targets,
            alphas,
        )
        selected_arx, arx_losses = _select_alpha(
            arx_paths,
            [row["validation_arx"] for row in fold_rows],
            validation_targets,
            alphas,
        )
        selections[task.name] = {
            "AR_alpha": selected_ar,
            "ARX_alpha": selected_arx,
            "AR_mean_validation_MSE_by_alpha": dict(zip(map(str, alphas), ar_losses)),
            "ARX_mean_validation_MSE_by_alpha": dict(
                zip(map(str, alphas), arx_losses)
            ),
        }
        for index, row in enumerate(fold_rows):
            target = validation_targets[index]
            origins = row["validation_targets"] - task.horizon
            persistence = y[origins]
            ar_prediction = ar_paths[index].predict(
                row["validation_ar"], selected_ar
            )
            arx_prediction = arx_paths[index].predict(
                row["validation_arx"], selected_arx
            )
            persistence_metrics = regression_metrics(target, persistence)
            ar_metrics = regression_metrics(target, ar_prediction)
            arx_metrics = regression_metrics(target, arx_prediction)
            delta = (
                ar_metrics["MSE_mm2"] - arx_metrics["MSE_mm2"]
            ) / max(ar_metrics["MSE_mm2"], np.finfo(np.float64).eps)
            output.append(
                {
                    "task": task.name,
                    "Lx": task.L_x,
                    "Ly": task.L_y,
                    "horizon": task.horizon,
                    "fold": row["fold"].fold,
                    "purge_gap": row["fold"].purge_gap,
                    "AR_ridge_alpha": selected_ar,
                    "ARX_ridge_alpha": selected_arx,
                    "persistence_RMSE_mm": persistence_metrics["RMSE_mm"],
                    "AR_RMSE_mm": ar_metrics["RMSE_mm"],
                    "ARX_RMSE_mm": arx_metrics["RMSE_mm"],
                    "AR_MSE_mm2": ar_metrics["MSE_mm2"],
                    "ARX_MSE_mm2": arx_metrics["MSE_mm2"],
                    "delta_X_given_AR": delta,
                    "direction_positive": bool(delta > 0.0),
                }
            )
    return output, selections
