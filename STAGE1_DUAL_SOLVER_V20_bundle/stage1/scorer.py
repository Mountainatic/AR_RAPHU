"""Bounded lag scorer for sample-dependent local lag correction (Section 2.2).

Each variable j has a small scalar scoring network S_j: R -> R
that reads only the variable's own value at a given lag:
    score[b,j,tau] = S_j(x_lag[b,j,tau])
    r[b,j,tau] = epsilon * tanh(score[b,j,tau])

Constraints:
  - S_j cannot read other variables
  - S_j cannot read labels
  - No free lag-specific bias to offset Gamma prior
  - epsilon is a global hyperparameter (not a learnable parameter)
"""

import torch
import torch.nn as nn


class BoundedLagScorer(nn.Module):
    """Per-variable scalar scorer for sample-dependent lag correction.

    Structure: Linear(1, hidden) -> Tanh -> Linear(hidden, 1)

    Args:
        num_variables: N, number of input variables
        hidden_size: hidden dimension for each scorer, default 8
        epsilon: global bound for local correction, default 0.5
    """
    def __init__(self, num_variables: int, hidden_size: int = 8,
                 epsilon: float = 0.5):
        super().__init__()
        self.num_variables = num_variables
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        # Each variable has its own independent scorer
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            for _ in range(num_variables)
        ])

    def forward(self, x_lag: torch.Tensor) -> torch.Tensor:
        """Compute bounded score correction r.

        Args:
            x_lag: [B, N, L] where lag=0 means current time.
                   Must already be flipped from input format.

        Returns:
            r: [B, N, L] bounded correction in [-epsilon, +epsilon].
        """
        B, N, L = x_lag.shape
        scores = []
        for j in range(N):
            xj = x_lag[:, j, :].reshape(B * L, 1)  # [B*L, 1]
            out = self.branches[j](xj).reshape(B, L)  # [B, L]
            scores.append(out)
        score = torch.stack(scores, dim=1)  # [B, N, L]

        r = self.epsilon * torch.tanh(score)
        return r

    def extra_repr(self) -> str:
        return f"num_vars={self.num_variables}, hidden={self.hidden_size}, epsilon={self.epsilon}"
