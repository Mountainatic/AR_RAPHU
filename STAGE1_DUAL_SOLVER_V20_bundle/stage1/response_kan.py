"""Per-variable response functions for Stage1.

Contains:
- UnivariateKANResponse: per-variable KAN (default)
- TruthResponseOracle: ground-truth functions from truth_functions.py (audit only)
- MLPResponseOracle: per-variable small MLP (audit only)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from layers import KANLinear


class UnivariateKANResponse(nn.Module):
    """Per-variable univariate KAN response functions.

    The public ``branches`` ModuleList and its state-dict layout are preserved
    for checkpoint compatibility and group pruning.  The default forward path
    stacks those parameters at runtime and evaluates all variables in two
    batched KAN operations, avoiding N separate CUDA launch chains.
    """
    def __init__(self, num_variables: int, hidden_kan: int = 8,
                 grid_size: int = 7, spline_order: int = 3,
                 input_grid_ranges=None, second_layer_grid_range=(-3, 3),
                 active_mask=None, execution_mode: str = "auto"):
        super().__init__()
        if execution_mode not in {"auto", "vectorized", "legacy"}:
            raise ValueError("execution_mode must be 'auto', 'vectorized', or 'legacy'")
        self.num_variables = num_variables
        self.hidden_kan = hidden_kan
        self.execution_mode = execution_mode
        if input_grid_ranges is None:
            input_grid_ranges = [(-3, 3)] * num_variables
        if len(input_grid_ranges) != num_variables:
            raise ValueError("input_grid_ranges must have one range per variable")
        self.input_grid_ranges = [tuple(map(float, r)) for r in input_grid_ranges]
        self.second_layer_grid_range = tuple(map(float, second_layer_grid_range))
        self.register_buffer("active_mask", (
            torch.ones(num_variables, dtype=torch.bool) if active_mask is None
            else active_mask.to(torch.bool)))
        self.branches = nn.ModuleList([
            nn.Sequential(
                KANLinear(1, hidden_kan, grid_size=grid_size, spline_order=spline_order,
                          grid_range=self.input_grid_ranges[j]),
                KANLinear(hidden_kan, 1, grid_size=grid_size, spline_order=spline_order,
                          grid_range=self.second_layer_grid_range),
            ) for j in range(num_variables)
        ])

    @staticmethod
    def _batched_b_splines(x: torch.Tensor, grids: torch.Tensor,
                           spline_order: int) -> torch.Tensor:
        """B-spline bases for x ``[B,N,T,I]`` and per-variable grids ``[N,G]``."""
        x_ext = x.unsqueeze(-1)  # [B,N,T,I,1]
        grid = grids.view(1, grids.shape[0], 1, 1, grids.shape[1])
        bases = ((x_ext >= grid[..., :-1]) & (x_ext < grid[..., 1:])).to(x.dtype)
        for k in range(1, spline_order + 1):
            left_grid = grid[..., :-(k + 1)]
            left_top = grid[..., k:-1]
            right_top = grid[..., k + 1:]
            right_grid = grid[..., 1:-k]
            left = (x_ext - left_grid) / (left_top - left_grid) * bases[..., :-1]
            right = (right_top - x_ext) / (right_top - right_grid) * bases[..., 1:]
            bases = left + right
        return bases

    @classmethod
    def _batched_layer(cls, x: torch.Tensor, layers) -> torch.Tensor:
        """Apply corresponding KANLinear layers to each variable without changing keys."""
        first = layers[0]
        in_features = first.in_features
        out_features = first.out_features
        order = first.spline_order
        if x.shape[1] != len(layers) or x.shape[-1] != in_features:
            raise ValueError("batched KAN input/layer shape mismatch")
        # The original model constructs homogeneous layer shapes. Keep an
        # explicit guard so future architecture changes fail loudly.
        for layer in layers:
            if (layer.in_features, layer.out_features, layer.spline_order) != (
                    in_features, out_features, order):
                raise ValueError("all per-variable KAN layers must have identical shapes")
        base_weight = torch.stack([layer.base_weight for layer in layers], dim=0)  # [N,O,I]
        spline_weight = torch.stack([layer.spline_weight for layer in layers], dim=0)  # [N,O,I,C]
        grids = torch.stack([layer.grid for layer in layers], dim=0)  # [N,G]
        base = torch.einsum("bnti,noi->bnto", torch.nn.functional.silu(x), base_weight)
        basis = cls._batched_b_splines(x, grids, order)
        spline = torch.einsum("bntic,noic->bnto", basis, spline_weight)
        # scale_* are fixed floats in the shipped KANLinear implementation.
        return base * first.scale_base + spline * first.scale_spline

    def forward_vectorized(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate values ``[B,N,T]`` and return ``[B,N,T]``."""
        if values.ndim != 3 or values.shape[1] != self.num_variables:
            raise ValueError("values must have shape [B,N,T]")
        first_layers = [branch[0] for branch in self.branches]
        second_layers = [branch[1] for branch in self.branches]
        hidden = self._batched_layer(values.unsqueeze(-1), first_layers)
        output = self._batched_layer(hidden, second_layers).squeeze(-1)
        return output * self.active_mask.view(1, -1, 1).to(output.dtype)

    def forward_legacy(self, x_lag: torch.Tensor) -> torch.Tensor:
        B, N, L = x_lag.shape
        responses = []
        for j in range(N):
            if not self.active_mask[j]:
                responses.append(x_lag.new_zeros(B, L))
                continue
            xj = x_lag[:, j, :].reshape(B * L, 1)
            out = self.branches[j](xj).reshape(B, L)
            responses.append(out)
        return torch.stack(responses, dim=1)

    def forward(self, x_lag: torch.Tensor) -> torch.Tensor:
        if self.execution_mode == "legacy":
            return self.forward_legacy(x_lag)
        if self.execution_mode == "auto" and not x_lag.is_cuda:
            return self.forward_legacy(x_lag)
        return self.forward_vectorized(x_lag)

    def forward_sequence(self, raw_sequence: torch.Tensor) -> torch.Tensor:
        """Evaluate each unique chronological point once.

        Accepts ``[N,T]`` or ``[B,N,T]`` and preserves the same shape.
        """
        squeeze = raw_sequence.ndim == 2
        values = raw_sequence.unsqueeze(0) if squeeze else raw_sequence
        if values.ndim != 3:
            raise ValueError("raw_sequence must have shape [N,T] or [B,N,T]")
        out = self.forward_vectorized(values)
        return out.squeeze(0) if squeeze else out

    def get_branch_params(self, j):
        params = [p.data.view(-1) for p in self.branches[j].parameters()]
        return torch.cat(params) if params else torch.tensor([], device=self.active_mask.device)

    def set_branch_params(self, j, theta):
        offset = 0
        for p in self.branches[j].parameters():
            numel = p.data.numel()
            p.data.copy_(theta[offset:offset+numel].view_as(p.data))
            offset += numel

    def compute_branch_norms(self):
        # Avoid repeated flatten/cat allocations. This stays differentiability-free
        # because branch norms are a pruning/diagnostic quantity.
        norms = []
        for branch in self.branches:
            square = None
            for parameter in branch.parameters():
                term = parameter.detach().square().sum()
                square = term if square is None else square + term
            norms.append(square.sqrt() if square is not None else self.active_mask.new_tensor(0.0, dtype=torch.float32))
        return torch.stack(norms)

    def extra_repr(self):
        return (f"num_vars={self.num_variables}, hidden={self.hidden_kan}, "
                f"execution_mode={self.execution_mode}")


class TruthResponseOracle(nn.Module):
    """Response oracle: uses truth_functions.py to compute exact responses.
    
    For audit only. Verifies O1 forward pipeline correctness.
    Each active variable j gets ground-truth response function f_j.
    """
    def __init__(self, num_variables: int, active_vars):
        super().__init__()
        self.num_variables = num_variables
        self.active_vars = list(active_vars)
        from stage1.truth_functions import get_true_function
        self._true_fs = {j: get_true_function(j) for j in active_vars}
        # Dummy parameter for training framework compatibility
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_lag):
        """Compute exact true response. x_lag: [B, N, L], returns [B, N, L]."""
        import numpy as np
        B, N, L = x_lag.shape
        device = x_lag.device; dtype = x_lag.dtype
        response = torch.zeros(B, N, L, device=device, dtype=dtype)
        for j in self.active_vars:
            xj_np = x_lag[:, j, :].detach().cpu().numpy()
            fj_np = self._true_fs[j](xj_np.astype(np.float64))
            response[:, j, :] = torch.tensor(fj_np, device=device, dtype=dtype)
        return response

    def compute_branch_norms(self):
        norms = torch.zeros(self.num_variables)
        norms[self.active_vars] = 10.0  # mark active
        return norms


class MLPResponseOracle(nn.Module):
    """Per-variable small MLP: 1 -> h -> h -> 1 (ReLU hidden).
    
    For audit only. Compares MLP vs KAN capacity.
    """
    def __init__(self, num_variables: int, hidden_mlp: int = 16):
        super().__init__()
        self.num_variables = num_variables
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_mlp), nn.ReLU(),
                nn.Linear(hidden_mlp, hidden_mlp), nn.ReLU(),
                nn.Linear(hidden_mlp, 1),
            ) for _ in range(num_variables)
        ])

    def forward(self, x_lag):
        B, N, L = x_lag.shape; responses = []
        for j in range(N):
            xj = x_lag[:, j, :].reshape(B * L, 1)
            out = self.branches[j](xj).reshape(B, L)
            responses.append(out)
        return torch.stack(responses, dim=1)

    def compute_branch_norms(self):
        norms = []
        for j in range(self.num_variables):
            flat = torch.cat([p.data.view(-1) for p in self.branches[j].parameters()])
            norms.append(torch.norm(flat))
        return torch.stack(norms)
