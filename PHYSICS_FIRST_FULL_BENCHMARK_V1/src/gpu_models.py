"""Compact causal model zoo for the frozen GPU benchmark.

Paper-inspired models that are not exact upstream reproductions are explicitly
named ``*_adapted`` in the configuration and result tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelOutput:
    prediction: Tensor
    auxiliary_loss: Tensor | None = None


def unwrap_output(output: Tensor | ModelOutput | tuple[Tensor, Tensor]) -> ModelOutput:
    if isinstance(output, ModelOutput):
        return output
    if isinstance(output, tuple):
        prediction, auxiliary = output
        return ModelOutput(prediction=prediction, auxiliary_loss=auxiliary)
    return ModelOutput(prediction=output)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class MLPRegressor(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden // 2), 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
        cell: str = "gru",
    ):
        super().__init__()
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = recurrent(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).squeeze(-1)


class LSTMSelfAttentionRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        encoded, _ = self.lstm(x)
        length = encoded.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        attended, _ = self.attention(encoded, encoded, encoded, attn_mask=causal_mask)
        state = self.norm(encoded[:, -1] + attended[:, -1])
        return self.head(state).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, x: Tensor) -> Tensor:
        return x if self.amount == 0 else x[..., :-self.amount]


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.block(x) + self.skip(x))


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 48,
        levels: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        current = input_dim
        for level in range(levels):
            blocks.append(CausalConvBlock(current, channels, kernel, 2**level, dropout))
            current = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        state = self.network(x.transpose(1, 2))[..., -1]
        return self.head(state).squeeze(-1)


class LinearTimeRegressor(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, normalized: bool = False):
        super().__init__()
        self.normalized = normalized
        self.temporal = nn.Linear(sequence_length, 1)
        self.channel = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        anchor = x[:, -1:, :] if self.normalized else torch.zeros_like(x[:, -1:, :])
        centered = x - anchor
        per_channel = self.temporal(centered.transpose(1, 2)).squeeze(-1)
        prediction = self.channel(per_channel).squeeze(-1)
        if self.normalized:
            prediction = prediction + anchor.mean(dim=-1).squeeze(-1) * 0.0
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, width: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[: x.shape[1]].to(dtype=x.dtype, device=x.device)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.project = nn.Linear(input_dim, width)
        self.position = PositionalEncoding(width, sequence_length)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        state = self.position(self.project(x))
        length = state.shape[1]
        causal_mask = torch.triu(
            torch.full((length, length), float("-inf"), device=x.device), diagonal=1
        )
        state = self.encoder(state, mask=causal_mask)
        return self.head(self.norm(state[:, -1])).squeeze(-1)


class PatchTSTLite(nn.Module):
    """Causal PatchTST-style encoder; explicitly an adapted compact baseline."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        patch_length: int = 12,
        stride: int = 12,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_length > sequence_length:
            raise ValueError("PATCH_LONGER_THAN_SEQUENCE")
        self.patch_length = patch_length
        self.stride = stride
        patch_count = 1 + (sequence_length - patch_length) // stride
        self.project = nn.Linear(patch_length * input_dim, width)
        self.position = PositionalEncoding(width, patch_count)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        # unfold gives B, P, C, patch; flatten preserves each causal patch only.
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        state = self.position(self.project(patches))
        length = state.shape[1]
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=x.device), diagonal=1
        )
        state = self.encoder(state, mask=mask)
        return self.head(self.norm(state[:, -1])).squeeze(-1)


class TimesNetLite(nn.Module):
    """Multi-kernel temporal block inspired by TimesNet, not an exact reproduction."""

    def __init__(self, input_dim: int, width: int = 48, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(width, width, kernel, padding=kernel - 1),
                    Chomp1d(kernel - 1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in (3, 5, 9)
            ]
        )
        self.mix = nn.Conv1d(width * len(self.branches), width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        state = self.input(x.transpose(1, 2))
        mixed = self.mix(torch.cat([branch(state) for branch in self.branches], dim=1))
        state = self.norm(state + mixed)
        return self.head(state[..., -1]).squeeze(-1)


class GraphTemporalEncoder(nn.Module):
    def __init__(self, sequence_length: int, node_count: int, width: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(sequence_length, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.node_count = node_count

    def forward(self, x: Tensor) -> Tensor:
        # B,T,C -> B,C,T -> B,C,width
        return self.temporal(x.transpose(1, 2))


class StaticGCNRegressor(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, width: int = 32, layers: int = 2):
        super().__init__()
        self.temporal = GraphTemporalEncoder(sequence_length, input_dim, width)
        adjacency = torch.eye(input_dim) + torch.ones(input_dim, input_dim) / input_dim
        degree = adjacency.sum(dim=1)
        normalized = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
        self.register_buffer("adjacency", normalized, persistent=True)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: Tensor) -> Tensor:
        nodes = self.temporal(x)
        for layer in self.layers:
            nodes = F.gelu(layer(torch.einsum("ij,bjd->bid", self.adjacency, nodes)))
        pooled = nodes.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class AdaptiveGraphRegressor(nn.Module):
    """Adaptive temporal graph model used for T-AKGNN-style adapted baselines."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        width: int = 32,
        heads: int = 4,
        kan_head: bool = False,
    ):
        super().__init__()
        self.temporal = GraphTemporalEncoder(sequence_length, input_dim, width)
        self.node_embeddings = nn.Parameter(torch.randn(input_dim, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.kan_head = RBFKANHead(width) if kan_head else nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        nodes = self.temporal(x) + self.node_embeddings.unsqueeze(0)
        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        pooled = self.norm(nodes + attended).mean(dim=1)
        return self.kan_head(pooled).squeeze(-1)


class RBFKANHead(nn.Module):
    """Small differentiable RBF expansion used as an adapted KAN head."""

    def __init__(self, input_dim: int, grid_size: int = 8):
        super().__init__()
        centers = torch.linspace(-2.5, 2.5, grid_size)
        self.register_buffer("centers", centers, persistent=True)
        self.log_width = nn.Parameter(torch.tensor(math.log(0.75)))
        self.weight = nn.Parameter(torch.empty(input_dim, grid_size))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        width = torch.exp(self.log_width).clamp_min(1e-3)
        basis = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / width) ** 2)
        return torch.einsum("bdg,dg->b", basis, self.weight) + self.bias


class NoGraphKANRegressor(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, width: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * input_dim, width),
            nn.LayerNorm(width),
            nn.Tanh(),
        )
        self.head = RBFKANHead(width)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


class GRUVAERegressor(nn.Module):
    """Supervised GRU-VAE baseline with a reconstruction/KL auxiliary loss."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        hidden: int = 48,
        latent: int = 16,
        beta: float = 1e-3,
        reconstruction_weight: float = 1e-2,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.beta = beta
        self.reconstruction_weight = reconstruction_weight
        self.encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.mean = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.regressor = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, sequence_length * input_dim),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        _, hidden = self.encoder(x)
        state = hidden[-1]
        mean = self.mean(state)
        logvar = self.logvar(state).clamp(-10.0, 10.0)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            latent = mean
        prediction = self.regressor(latent).squeeze(-1)
        reconstruction = self.decoder(latent).view_as(x)
        recon_loss = F.mse_loss(reconstruction, x)
        kl = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
        auxiliary = self.reconstruction_weight * recon_loss + self.beta * kl
        return ModelOutput(prediction=prediction, auxiliary_loss=auxiliary)


class BidirectionalLSTMControl(nn.Module):
    """Explicitly non-causal control; excluded from the formal online tables."""

    def __init__(self, input_dim: int, hidden: int = 48, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        _, (hidden, _) = self.encoder(x)
        state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.head(state).squeeze(-1)


class S4DLiteRegressor(nn.Module):
    """Compact causal structured-state-space adaptation using long depthwise filters."""

    def __init__(self, input_dim: int, width: int = 48, levels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        blocks: list[nn.Module] = []
        for level in range(levels):
            dilation = 2**level
            padding = 8 * dilation
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        width,
                        width,
                        kernel_size=9,
                        dilation=dilation,
                        padding=padding,
                        groups=width,
                    ),
                    Chomp1d(padding),
                    nn.Conv1d(width, width, 1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.GroupNorm(1, width) for _ in blocks])
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        state = self.input(x.transpose(1, 2))
        for block, norm in zip(self.blocks, self.norms):
            state = norm(state + block(state))
        return self.head(state[..., -1]).squeeze(-1)


class StaticGATRegressor(nn.Module):
    """Window-summary graph attention baseline."""

    def __init__(self, sequence_length: int, input_dim: int, width: int = 32, heads: int = 4):
        super().__init__()
        self.temporal = GraphTemporalEncoder(sequence_length, input_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        nodes = self.temporal(x)
        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        return self.head(self.norm(nodes + attended).mean(dim=1)).squeeze(-1)


class TemporalGraphRegressor(nn.Module):
    """Causal per-node recurrent encoder followed by graph message passing."""

    def __init__(
        self,
        input_dim: int,
        width: int = 32,
        heads: int = 4,
        attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.GRU(1, width, batch_first=True)
        self.node_embeddings = nn.Parameter(torch.randn(input_dim, width) * 0.02)
        self.attention = (
            nn.MultiheadAttention(width, heads, batch_first=True)
            if attention
            else None
        )
        adjacency = torch.eye(input_dim) + torch.ones(input_dim, input_dim) / input_dim
        degree = adjacency.sum(dim=1)
        self.register_buffer(
            "adjacency",
            adjacency / torch.sqrt(degree[:, None] * degree[None, :]),
            persistent=True,
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, channels = x.shape
        series = x.transpose(1, 2).reshape(batch * channels, steps, 1)
        _, hidden = self.encoder(series)
        nodes = hidden[-1].reshape(batch, channels, -1)
        nodes = nodes + self.node_embeddings.unsqueeze(0)
        if self.attention is None:
            message = torch.einsum("ij,bjd->bid", self.adjacency, nodes)
        else:
            message, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        return self.head(self.norm(nodes + message).mean(dim=1)).squeeze(-1)


class FixedGraphKANRegressor(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, width: int = 32):
        super().__init__()
        self.temporal = GraphTemporalEncoder(sequence_length, input_dim, width)
        adjacency = torch.eye(input_dim) + torch.ones(input_dim, input_dim) / input_dim
        degree = adjacency.sum(dim=1)
        self.register_buffer(
            "adjacency",
            adjacency / torch.sqrt(degree[:, None] * degree[None, :]),
            persistent=True,
        )
        self.message = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = RBFKANHead(width)

    def forward(self, x: Tensor) -> Tensor:
        nodes = self.temporal(x)
        nodes = F.gelu(
            self.message(torch.einsum("ij,bjd->bid", self.adjacency, nodes))
        )
        return self.head(self.norm(nodes.mean(dim=1)))


class TemporalAutoencoderRegressor(nn.Module):
    """Supervised causal temporal autoencoder baseline."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        hidden: int = 48,
        latent: int = 16,
        reconstruction_weight: float = 1e-2,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.reconstruction_weight = reconstruction_weight
        self.encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.latent = nn.Linear(hidden, latent)
        self.regressor = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, sequence_length * input_dim),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        _, hidden = self.encoder(x)
        latent = self.latent(hidden[-1])
        prediction = self.regressor(latent).squeeze(-1)
        reconstruction = self.decoder(latent).view_as(x)
        auxiliary = self.reconstruction_weight * F.mse_loss(reconstruction, x)
        return ModelOutput(prediction=prediction, auxiliary_loss=auxiliary)


class PyramidVAERegressor(nn.Module):
    """Causal multiresolution VAE adaptation for the supervised benchmark."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        hidden: int = 48,
        latent: int = 16,
        beta: float = 1e-3,
        reconstruction_weight: float = 1e-2,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.beta = beta
        self.reconstruction_weight = reconstruction_weight
        summary_dim = input_dim * 3
        self.summary = nn.Sequential(nn.Linear(summary_dim, hidden), nn.GELU())
        self.mean = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.regressor = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, sequence_length * input_dim),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        summaries = torch.cat(
            (
                x[:, -1],
                x[:, -min(12, x.shape[1]) :].mean(dim=1),
                x.mean(dim=1),
            ),
            dim=-1,
        )
        encoded = self.summary(summaries)
        mean = self.mean(encoded)
        logvar = self.logvar(encoded).clamp(-10.0, 10.0)
        latent = (
            mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            if self.training
            else mean
        )
        prediction = self.regressor(latent).squeeze(-1)
        reconstruction = self.decoder(latent).view_as(x)
        recon = F.mse_loss(reconstruction, x)
        kl = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
        return ModelOutput(
            prediction=prediction,
            auxiliary_loss=self.reconstruction_weight * recon + self.beta * kl,
        )


MODEL_ALIASES = {
    "mlp": "mlp",
    "lstm": "lstm",
    "gru": "gru",
    "tcn": "tcn",
    "dlinear": "dlinear",
    "nlinear": "nlinear",
    "lstm_sa": "lstm_sa",
    "transformer": "transformer",
    "patchtst_adapted": "patchtst_adapted",
    "timesnet_adapted": "timesnet_adapted",
    "static_gcn": "static_gcn",
    "t_akgnn_adapted": "t_akgnnn_adapted",
    "adaptive_graph_kan": "adaptive_graph_kan",
    "no_graph_kan_adapted": "no_graph_kan_adapted",
    "gru_vae_adapted": "gru_vae_adapted",
    "bilstm_control": "bilstm_control",
    "informer_lite_adapted": "patchtst_adapted",
    "autoformer_lite_adapted": "timesnet_adapted",
    "s4d_adapted": "s4d_adapted",
    "static_gat": "static_gat",
    "temporal_gcn": "temporal_gcn",
    "temporal_gat": "temporal_gat",
    "dgdl_adapted": "adaptive_graph_mlp",
    "akgnn_window_summary_adapted": "adaptive_graph_mlp",
    "adaptive_graph_mlp": "adaptive_graph_mlp",
    "fixed_graph_kan": "fixed_graph_kan",
    "dmvaer_adapted": "gru_vae_adapted",
    "pyramid_vae_adapted": "pyramid_vae_adapted",
    "temporal_autoencoder": "temporal_autoencoder",
}


def build_model(
    name: str,
    *,
    sequence_length: int,
    input_dim: int,
    parameters: dict[str, Any] | None = None,
) -> nn.Module:
    parameters = dict(parameters or {})
    normalized = MODEL_ALIASES.get(name, name)
    if normalized == "mlp":
        return MLPRegressor(sequence_length, input_dim, **parameters)
    if normalized == "lstm":
        return RecurrentRegressor(input_dim, cell="lstm", **parameters)
    if normalized == "gru":
        return RecurrentRegressor(input_dim, cell="gru", **parameters)
    if normalized == "tcn":
        return TCNRegressor(input_dim, **parameters)
    if normalized == "dlinear":
        return LinearTimeRegressor(sequence_length, input_dim, normalized=False)
    if normalized == "nlinear":
        return LinearTimeRegressor(sequence_length, input_dim, normalized=True)
    if normalized == "lstm_sa":
        return LSTMSelfAttentionRegressor(input_dim, **parameters)
    if normalized == "transformer":
        return TransformerRegressor(sequence_length, input_dim, **parameters)
    if normalized == "patchtst_adapted":
        return PatchTSTLite(sequence_length, input_dim, **parameters)
    if normalized == "timesnet_adapted":
        return TimesNetLite(input_dim, **parameters)
    if normalized == "static_gcn":
        return StaticGCNRegressor(sequence_length, input_dim, **parameters)
    if normalized == "t_akgnn_adapted":
        return AdaptiveGraphRegressor(sequence_length, input_dim, kan_head=False, **parameters)
    if normalized == "adaptive_graph_kan":
        return AdaptiveGraphRegressor(sequence_length, input_dim, kan_head=True, **parameters)
    if normalized == "no_graph_kan_adapted":
        return NoGraphKANRegressor(sequence_length, input_dim, **parameters)
    if normalized == "gru_vae_adapted":
        return GRUVAERegressor(sequence_length, input_dim, **parameters)
    if normalized == "bilstm_control":
        return BidirectionalLSTMControl(input_dim, **parameters)
    if normalized == "s4d_adapted":
        return S4DLiteRegressor(input_dim, **parameters)
    if normalized == "static_gat":
        return StaticGATRegressor(sequence_length, input_dim, **parameters)
    if normalized == "temporal_gcn":
        return TemporalGraphRegressor(input_dim, attention=False, **parameters)
    if normalized == "temporal_gat":
        return TemporalGraphRegressor(input_dim, attention=True, **parameters)
    if normalized == "adaptive_graph_mlp":
        return AdaptiveGraphRegressor(
            sequence_length, input_dim, kan_head=False, **parameters
        )
    if normalized == "fixed_graph_kan":
        return FixedGraphKANRegressor(sequence_length, input_dim, **parameters)
    if normalized == "pyramid_vae_adapted":
        return PyramidVAERegressor(sequence_length, input_dim, **parameters)
    if normalized == "temporal_autoencoder":
        return TemporalAutoencoderRegressor(sequence_length, input_dim, **parameters)
    raise KeyError(f"UNKNOWN_MODEL:{name}")
