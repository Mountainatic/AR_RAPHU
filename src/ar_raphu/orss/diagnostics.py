"""Serializable ORSS diagnostics and numerical comparison helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).reshape(-1))
    denominator = torch.linalg.vector_norm(right.reshape(-1)).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return float((numerator / denominator).item())


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return sha256(array.view(np.uint8)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

