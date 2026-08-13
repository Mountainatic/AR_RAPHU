from __future__ import annotations

import numpy as np
from prism_benchmark.neurobem_literature import quaternion_conjugate, quaternion_multiply, normalize_quaternion


def state_errors(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    relative = quaternion_multiply(quaternion_conjugate(normalize_quaternion(target[:, 3:7])), normalize_quaternion(prediction[:, 3:7]))
    angle = 2.0 * np.arctan2(np.linalg.norm(relative[:, 1:], axis=1), np.abs(relative[:, 0]))
    return {
        "velocity": np.linalg.norm(target[:, :3] - prediction[:, :3], axis=1),
        "attitude": angle,
        "body_rate": np.linalg.norm(target[:, 7:10] - prediction[:, 7:10], axis=1),
    }


def divergence_time(errors: dict[str, np.ndarray], finite: np.ndarray, thresholds: dict[str, float], persistence: int) -> int | None:
    bad = (~finite) | (errors["velocity"] > thresholds["velocity"]) | (errors["attitude"] > thresholds["attitude"]) | (errors["body_rate"] > thresholds["body_rate"])
    run = 0
    for i, value in enumerate(bad):
        run = run + 1 if value else 0
        if run >= persistence:
            return i - persistence + 1
    return None

