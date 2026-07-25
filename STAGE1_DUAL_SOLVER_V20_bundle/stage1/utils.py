"""Utilities for Stage1: config management, seed setting, metrics calculation.

Follows the same style as the project's Config class and utils module.
"""
import random
import numpy as np
import torch
import os


class Stage1Config:
    """Configuration for Stage1TargetDelayKAN training and evaluation."""

    # Data
    WINDOW_SIZE: int = 32
    HORIZON: int = 1

    # Model
    HIDDEN_SCORE: int = 8
    HIDDEN_KAN: int = 8
    KAN_GRID_SIZE: int = 7
    KAN_SPLINE_ORDER: int = 3
    EPSILON: float = 0.5
    DELTA: float = 1e-3
    ACTIVE_THRESHOLD: float = 1e-6

    # Training
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5
    MAX_EPOCHS: int = 100
    PATIENCE: int = 15
    SEED: int = 42
    LAMBDA_GROUP: float = 0.01
    LAMBDA_SMOOTH: float = 0.0

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed_stage1(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
