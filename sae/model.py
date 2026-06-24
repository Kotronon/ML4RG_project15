"""Sparse Autoencoder (SAE) implementations.

Three variants:

1. VanillaSAE   – L1-penalised ReLU hidden layer (classic formulation)
2. TopKSAE      – Per-sample top-k sparsity (Anthropic / Conmy et al. 2024)
3. BatchTopKSAE – Batch-level top-k sparsity (Bussmann et al. 2024,
                  github.com/bartbussmann/BatchTopK)

All variants share the same encoder/decoder weight structure and support
decoder-norm clamping (unit-norm dictionary atoms) for interpretability.

Reference:
    Andrew Ng, "Sparse Autoencoder", CS294A lecture notes (2011).
    https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

SAEVariant = Literal["vanilla", "topk", "batchtopk"]
SAE_VARIANTS: tuple[str, ...] = ("vanilla", "topk", "batchtopk")


@dataclass
class SAEConfig:
    input_dim: int = 4096           # embedding dimensionality (e.g. 4096 for Evo2 7B)
    expansion_factor: int = 8       # hidden_dim = expansion_factor * input_dim
    variant: SAEVariant = "topk"
    # --- TopK / BatchTopK ---
    k: int = 32                     # number of active features per sample (TopK)
    k_fraction: float | None = None # if set, k = round(k_fraction * hidden_dim)
    # --- VanillaSAE ---
    l1_coeff: float = 1e-3
    # --- Shared ---
    normalize_decoder: bool = True  # clamp decoder columns to unit norm
    tied_weights: bool = False       # W_dec = W_enc.T (optional)
    init_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.k_fraction is not None:
            self.k = max(1, round(self.k_fraction * self.hidden_dim))

    @property
    def hidden_dim(self) -> int:
        return self.expansion_factor * self.input_dim


class SparseAutoencoder(nn.Module):
    """Base class — shared encoder/decoder structure."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h = cfg.input_dim, cfg.hidden_dim

        self.W_enc = nn.Parameter(torch.empty(d, h))
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.b_pre = nn.Parameter(torch.zeros(d))  # pre-encoder bias (decoder bias)

        if cfg.tied_weights:
            self.W_dec: nn.Parameter | None = None
        else:
            self.W_dec = nn.Parameter(torch.empty(h, d))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        if self.W_dec is not None:
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            # Normalise decoder columns to unit norm at init
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def decoder_weight(self) -> torch.Tensor:
        """Decoder weight matrix, shape [h, d]."""
        if self.W_dec is not None:
            return self.W_dec
        return self.W_enc.T  # tied weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activations before sparsity, shape [B, h]."""
        return (x - self.b_pre) @ self.W_enc + self.b_enc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sparse activations, shape [B, d]."""
        return z @ self.decoder_weight + self.b_pre

    @torch.no_grad()
    def clamp_decoder_norm(self) -> None:
        """Project decoder columns onto the unit sphere (called after each optimizer step)."""
        if not self.cfg.normalize_decoder:
            return
        if self.W_dec is not None:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class VanillaSAE(SparseAutoencoder):
    """SAE with ReLU activations + L1 sparsity penalty."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_act = self.encode(x)
        z = F.relu(pre_act)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.cfg.l1_coeff * z.abs().sum(dim=-1).mean()
        loss = recon_loss + sparsity_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "activations": z,       # [B, h]
            "x_hat": x_hat,
            "l0": (z > 0).float().sum(dim=-1).mean(),
        }


class TopKSAE(SparseAutoencoder):
    """SAE with per-sample top-k sparsity (Conmy et al. 2024 / Anthropic)."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_act = self.encode(x)
        z = self._topk(pre_act)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        # Auxiliary loss: encourage dead features to reconstruct the residual
        aux_loss = self._aux_loss(x - x_hat.detach(), pre_act, z)
        loss = recon_loss + aux_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "activations": z,
            "x_hat": x_hat,
            "l0": (z > 0).float().sum(dim=-1).mean(),
        }

    def _topk(self, pre_act: torch.Tensor) -> torch.Tensor:
        k = self.cfg.k
        top_vals, top_idx = pre_act.topk(k, dim=-1)
        top_vals = F.relu(top_vals)
        z = torch.zeros_like(pre_act)
        z.scatter_(-1, top_idx, top_vals)
        return z

    def _aux_loss(
        self,
        residual: torch.Tensor,
        pre_act: torch.Tensor,
        z: torch.Tensor,
        aux_k_factor: int = 2,
        aux_alpha: float = 1 / 32,
    ) -> torch.Tensor:
        """Auxiliary reconstruction loss for dead-feature revival."""
        dead_mask = (z == 0).all(dim=0)  # [h]
        if not dead_mask.any():
            return residual.new_zeros(())
        dead_pre = pre_act[:, dead_mask]
        k_aux = min(aux_k_factor * self.cfg.k, dead_mask.sum().item())
        top_vals, top_idx = dead_pre.topk(int(k_aux), dim=-1)
        top_vals = F.relu(top_vals)
        z_aux = torch.zeros_like(dead_pre)
        z_aux.scatter_(-1, top_idx, top_vals)
        # Partial decode using only dead features
        W_dead = self.decoder_weight[dead_mask]  # [n_dead, d]
        x_hat_aux = z_aux @ W_dead + self.b_pre
        return aux_alpha * F.mse_loss(x_hat_aux, residual)


class BatchTopKSAE(SparseAutoencoder):
    """SAE with batch-level top-k sparsity (Bussmann et al. 2024).

    Instead of keeping exactly k features active per sample, we keep the
    top B*k activations across the entire batch, where B is the batch size.
    This lets the sparsity budget shift to whichever samples need more features.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_act = self.encode(x)           # [B, h]
        z = self._batch_topk(pre_act)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss  # sparsity enforced structurally

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "activations": z,
            "x_hat": x_hat,
            "l0": (z > 0).float().sum(dim=-1).mean(),
        }

    def _batch_topk(self, pre_act: torch.Tensor) -> torch.Tensor:
        B, h = pre_act.shape
        k_total = B * self.cfg.k
        flat = pre_act.reshape(-1)               # [B*h]
        threshold, _ = flat.kthvalue(flat.numel() - k_total)
        z = F.relu(pre_act - threshold.detach())
        return z


SAE_CLASSES = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
}


def build_sae(cfg: SAEConfig) -> SparseAutoencoder:
    cls = SAE_CLASSES.get(cfg.variant)
    if cls is None:
        raise ValueError(f"Unknown SAE variant {cfg.variant!r}. Choose from {SAE_VARIANTS}.")
    return cls(cfg)
