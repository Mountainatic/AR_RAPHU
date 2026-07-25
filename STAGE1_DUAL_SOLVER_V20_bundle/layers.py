import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    """Paper Eq (25): h_i = x_i * w_i + b_i, with time-dim support.
    Input:  [B, N, L]
    Output: [B, N, L, D]
    """
    def __init__(self, num_nodes: int, embed_dim: int):
        super(InputEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        w = self.weight.unsqueeze(0).unsqueeze(2)
        b = self.bias.unsqueeze(0).unsqueeze(2)
        return x * w + b


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_range=[-3, 3]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid_min, grid_max = grid_range
        h = (grid_max - grid_min) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_min).float()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(noise.permute(2, 1, 0))

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[..., :-1]
            ) + (
                (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[..., 1:]
            )
        return bases

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum('bic,oic->bo', spline_basis, self.spline_weight)
        return base_output * self.scale_base + spline_output * self.scale_spline


class KANNetwork(nn.Module):
    def __init__(self, layer_sizes, grid_size=5):
        super(KANNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(KANLinear(layer_sizes[i], layer_sizes[i+1], grid_size=grid_size))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
