"""Discrete Gamma prior for per-variable lag distributions (Section 2.1).

Each variable j uses a discrete Gamma-shaped prior:
    log_pi_raw[j,tau] = (alpha_j - 1) * log(tau + delta) - (tau + delta) / beta_j
    pi[j,:] = softmax(log_pi_raw[j,:])

Supports two parametrizations:
  - 'alpha_beta': original alpha/beta parameters
  - 'mean_std': mean [0, L-1] and std [0.5, L/2], internally converted to alpha/beta
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteGammaPrior(nn.Module):
    """Discrete Gamma-shaped prior over lags for each variable.

    Args:
        num_variables: N, number of input variables
        max_lag: L, maximum lag / window size
        delta: small constant for log(tau+delta) stability, default 1e-3
        parametrization: 'mean_std' (default) or 'alpha_beta'
        alpha_min/max: bounds for alpha (alpha_beta mode)
        beta_min/max: bounds for beta (alpha_beta mode)
        mean_min/max: bounds for mean (mean_std mode), default [0, L-1]
        std_min/max: bounds for std (mean_std mode), default [0.5, L/2]
    """
    def __init__(self, num_variables: int, max_lag: int,
                 delta: float = 1e-3,
                 parametrization: str = 'mean_std',
                 alpha_min: float = 0.5, alpha_max: float = 10.0,
                 beta_min: float = 0.1, beta_max: float = None,
                 mean_min: float = None, mean_max: float = None,
                 std_min: float = 0.5, std_max: float = None):
        super().__init__()
        self.num_variables = num_variables
        self.max_lag = max_lag
        self.delta = delta
        self.parametrization = parametrization

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max if beta_max is not None else float(max_lag)

        self.mean_min = mean_min if mean_min is not None else 0.0
        self.mean_max = mean_max if mean_max is not None else float(max_lag - 1)
        self.std_min = std_min
        self.std_max = std_max if std_max is not None else float(max_lag) / 2.0

        if parametrization == 'mean_std':
            self.raw_mean = nn.Parameter(torch.zeros(num_variables))
            self.raw_std = nn.Parameter(torch.zeros(num_variables))
            # Not used in mean_std mode but kept for checkpoint compatibility
            self.register_buffer('raw_alpha', torch.zeros(num_variables))
            self.register_buffer('raw_beta', torch.zeros(num_variables))
        else:
            self.raw_alpha = nn.Parameter(torch.zeros(num_variables))
            self.raw_beta = nn.Parameter(torch.zeros(num_variables))
            self.register_buffer('raw_mean', torch.zeros(num_variables))
            self.register_buffer('raw_std', torch.zeros(num_variables))

        taus = torch.arange(max_lag, dtype=torch.float32)
        self.register_buffer("taus", taus)
        self.reset_parameters()

    def reset_parameters(self):
        if self.parametrization == 'mean_std':
            nn.init.constant_(self.raw_mean, 0.0)
            nn.init.constant_(self.raw_std, 0.0)
        else:
            nn.init.constant_(self.raw_alpha, 0.0)
            nn.init.constant_(self.raw_beta, 0.0)

    @torch.no_grad()
    def initialize_from_mean_std(self, mean, std):
        """Initialize bounded mean/std parameters and return actual discrete stats.

        Inputs are requested physical lag moments, never truth-derived internally.
        The returned values describe the resulting *discrete* Gamma distribution.
        """
        if self.parametrization != 'mean_std':
            raise RuntimeError("initialize_from_mean_std requires parametrization='mean_std'")
        mean = torch.as_tensor(mean, dtype=self.raw_mean.dtype, device=self.raw_mean.device)
        std = torch.as_tensor(std, dtype=self.raw_std.dtype, device=self.raw_std.device)
        if mean.numel() != self.num_variables or std.numel() != self.num_variables:
            raise ValueError("mean/std must have one value per variable")
        def inv_sigmoid(x):
            x = x.clamp(1e-5, 1 - 1e-5)
            return torch.log(x / (1 - x))
        mn = ((mean.reshape(-1) - self.mean_min) / (self.mean_max - self.mean_min)).clamp(1e-5, 1-1e-5)
        sd = ((std.reshape(-1) - self.std_min) / (self.std_max - self.std_min)).clamp(1e-5, 1-1e-5)
        self.raw_mean.copy_(inv_sigmoid(mn)); self.raw_std.copy_(inv_sigmoid(sd))
        q = self(); taus = self.taus
        actual_mean = (q * taus).sum(-1)
        actual_std = (q * (taus - actual_mean[:, None]).square()).sum(-1).sqrt()
        return {"mean": actual_mean, "std": actual_std, "peak": q.argmax(-1),
                "boundary_mass": q[:, -3:].sum(-1)}

    def get_alpha_beta(self):
        """Return bounded alpha, beta, converting from mean/std if needed."""
        if self.parametrization == 'mean_std':
            mean = self.mean_min + (self.mean_max - self.mean_min) * torch.sigmoid(self.raw_mean)
            std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(self.raw_std)
            # Convert: for continuous Gamma, mean=alpha*beta, var=alpha*beta^2
            # beta = var/mean = std^2/mean, alpha = mean/beta = mean^2/std^2
            safe_mean = mean + self.delta
            beta = std.pow(2) / safe_mean
            alpha = safe_mean / beta
            # Clamp to bounds
            beta = torch.clamp(beta, self.beta_min, self.beta_max)
            alpha = torch.clamp(alpha, self.alpha_min, self.alpha_max)
        else:
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)
            beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_beta)
        return alpha, beta

    def forward(self) -> torch.Tensor:
        """Compute the discrete Gamma prior pi.

        Returns:
            pi: [N, L] where each row sums to 1.
        """
        alpha, beta = self.get_alpha_beta()  # [N]
        N, L = self.num_variables, self.max_lag
        taus = self.taus  # [L]
        alpha = alpha.unsqueeze(-1)  # [N, 1]
        beta = beta.unsqueeze(-1)    # [N, 1]
        log_pi_raw = (alpha - 1.0) * torch.log(taus + self.delta) - (taus + self.delta) / beta
        log_pi = F.log_softmax(log_pi_raw, dim=-1)
        pi = torch.exp(log_pi)
        return pi

    def compute_prior_stats(self) -> tuple:
        """Compute discrete prior mean and variance of delay for each variable.

        Returns:
            prior_mean: [N] actual discrete mean delay tau under pi
            prior_var:  [N] actual discrete variance
        """
        pi = self.forward()  # [N, L]
        taus = self.taus     # [L]
        prior_mean = (pi * taus).sum(dim=-1)          # [N]
        prior_var = (pi * (taus - prior_mean.unsqueeze(-1)) ** 2).sum(dim=-1)  # [N]
        return prior_mean, prior_var

    def extra_repr(self) -> str:
        return (f"num_vars={self.num_variables}, max_lag={self.max_lag}, "
                f"param={self.parametrization}")
