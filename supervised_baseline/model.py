from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class smallCNN(nn.Module):
    def __init__(self, n_tfs: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        self.classifier = nn.Linear(in_features=128, out_features=n_tfs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class ResidualDilatedBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ResidualDilatedBlock requires an odd kernel_size")
        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResDilatedCNN(nn.Module):
    def __init__(
        self,
        n_tfs: int,
        *,
        dropout: float = 0.1,
        hidden_channels: int = 128,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_channels * 2, out_features=256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=n_tfs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        max_pool = x.amax(dim=-1)
        avg_pool = x.mean(dim=-1)
        x = torch.cat([max_pool, avg_pool], dim=1)
        return self.classifier(x)


class DenseSmallCNN(nn.Module):
    """Promoter-level CNN that emits one logit per TF and position."""

    def __init__(
        self,
        n_tfs: int,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, n_tfs, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseResDilatedCNN(nn.Module):
    """Promoter-level residual dilated CNN for dense TFBS scoring."""

    def __init__(
        self,
        n_tfs: int,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, n_tfs, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class DenseProteinResDilatedCNN(nn.Module):
    """Dense TFBS scorer conditioned on fixed TF protein embeddings.

    The DNA trunk is intentionally close to ``DenseResDilatedCNN``. Instead of
    learning one independent output channel per TF, the model projects DNA
    positions and TF protein embeddings into a shared space and scores them with
    a shared compatibility function. This makes held-out TF evaluation
    meaningful: unseen TFs can still be scored through their protein embedding.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        embedding_dim = int(tf_embeddings.shape[1])
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.dna_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.tf_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dna_bias = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.tf_bias = nn.Linear(hidden_channels, 1)
        self.score_scale = hidden_channels ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dna = self.stem(x)
        for block in self.blocks:
            dna = block(dna)

        dna_features = self.dna_projection(dna)
        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        tf_features = self.tf_projection(tf_embeddings)

        logits = torch.einsum("bcl,tc->btl", dna_features, tf_features)
        logits = logits * self.score_scale
        logits = logits + self.dna_bias(dna)
        logits = logits + self.tf_bias(tf_features).view(1, -1, 1)
        return logits


class LocalSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with an optional local sequence window."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        num_heads: int = 8,
        window_bp: int = 50,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if window_bp < 0:
            raise ValueError("window_bp must be non-negative")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")

        self.window_bp = int(window_bp)
        self.attn_norm = nn.LayerNorm(hidden_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        ffn_channels = max(hidden_channels, int(round(hidden_channels * ffn_multiplier)))
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, ffn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, hidden_channels),
            nn.Dropout(dropout),
        )

    def _attention_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        if self.window_bp == 0:
            return None
        positions = torch.arange(length, device=device)
        return (positions[:, None] - positions[None, :]).abs() > self.window_bp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attended, _ = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            attn_mask=self._attention_mask(x.shape[1], x.device),
            need_weights=False,
        )
        x = x + self.dropout(attended)
        return x + self.ffn(x)


class MultiHeadBilinearScorer(nn.Module):
    """Factorized TF-position compatibility scorer."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.1,
        bias_mode: str = "tf",
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if bias_mode not in {"none", "tf", "dna", "both"}:
            raise ValueError("bias_mode must be one of none, tf, dna, both")

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.bias_mode = bias_mode
        self.dropout = nn.Dropout(dropout)
        self.dna_projection = nn.Linear(hidden_channels, num_heads * head_dim)
        self.tf_projection = nn.Linear(hidden_channels, num_heads * head_dim)
        self.head_weights = nn.Parameter(torch.zeros(num_heads))
        self.dna_bias = nn.Linear(hidden_channels, 1) if bias_mode in {"dna", "both"} else None
        self.tf_bias = nn.Linear(hidden_channels, 1) if bias_mode in {"tf", "both"} else None
        self.score_scale = head_dim ** -0.5

    def forward(self, dna: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = dna.shape
        n_tfs = tf.shape[0]

        dna_heads = self.dna_projection(self.dropout(dna)).view(
            batch_size,
            length,
            self.num_heads,
            self.head_dim,
        )
        tf_heads = self.tf_projection(self.dropout(tf)).view(
            n_tfs,
            self.num_heads,
            self.head_dim,
        )
        head_weights = torch.softmax(self.head_weights, dim=0)
        logits = torch.einsum("blhd,thd,h->btl", dna_heads, tf_heads, head_weights)
        logits = logits * self.score_scale

        if self.dna_bias is not None:
            logits = logits + self.dna_bias(dna).squeeze(-1).unsqueeze(1)
        if self.tf_bias is not None:
            logits = logits + self.tf_bias(tf).view(1, -1, 1)
        return logits

    def forward_pairwise(self, dna: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = dna.shape
        if tf.ndim != 2 or tf.shape[0] != batch_size:
            raise ValueError(
                "Pairwise TF features must have shape [batch, hidden_channels]"
            )

        dna_heads = self.dna_projection(self.dropout(dna)).view(
            batch_size,
            length,
            self.num_heads,
            self.head_dim,
        )
        tf_heads = self.tf_projection(self.dropout(tf)).view(
            batch_size,
            self.num_heads,
            self.head_dim,
        )
        head_weights = torch.softmax(self.head_weights, dim=0)
        logits = torch.einsum("blhd,bhd,h->bl", dna_heads, tf_heads, head_weights)
        logits = logits * self.score_scale

        if self.dna_bias is not None:
            logits = logits + self.dna_bias(dna).squeeze(-1)
        if self.tf_bias is not None:
            logits = logits + self.tf_bias(tf)
        return logits.unsqueeze(1)


class MLPInteractionScorer(nn.Module):
    """Small interaction MLP over DNA/TF pair features."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        pair_dim: int = 64,
        scorer_hidden_dim: int = 128,
        dropout: float = 0.1,
        bias_mode: str = "tf",
    ) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be positive")
        if scorer_hidden_dim <= 0:
            raise ValueError("scorer_hidden_dim must be positive")
        if bias_mode not in {"none", "tf", "dna", "both"}:
            raise ValueError("bias_mode must be one of none, tf, dna, both")

        self.bias_mode = bias_mode
        self.dna_projection = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, pair_dim),
            nn.GELU(),
        )
        self.tf_projection = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, pair_dim),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim * 3, scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1),
        )
        self.dna_bias = nn.Linear(hidden_channels, 1) if bias_mode in {"dna", "both"} else None
        self.tf_bias = nn.Linear(hidden_channels, 1) if bias_mode in {"tf", "both"} else None

    def forward(self, dna: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        dna_pair = self.dna_projection(dna)
        tf_pair = self.tf_projection(tf)
        batch_size, length, _ = dna_pair.shape
        n_tfs = tf_pair.shape[0]

        dna_expanded = dna_pair.unsqueeze(1).expand(-1, n_tfs, -1, -1)
        tf_expanded = tf_pair.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, length, -1)
        pair_features = torch.cat(
            [
                dna_expanded,
                tf_expanded,
                dna_expanded * tf_expanded,
            ],
            dim=-1,
        )
        logits = self.mlp(pair_features).squeeze(-1)

        if self.dna_bias is not None:
            logits = logits + self.dna_bias(dna).squeeze(-1).unsqueeze(1)
        if self.tf_bias is not None:
            logits = logits + self.tf_bias(tf).view(1, -1, 1)
        return logits

    def forward_pairwise(self, dna: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        if tf.ndim != 2 or tf.shape[0] != dna.shape[0]:
            raise ValueError(
                "Pairwise TF features must have shape [batch, hidden_channels]"
            )

        dna_pair = self.dna_projection(dna)
        tf_pair = self.tf_projection(tf)
        tf_pair = tf_pair.unsqueeze(1).expand(-1, dna_pair.shape[1], -1)
        pair_features = torch.cat(
            [
                dna_pair,
                tf_pair,
                dna_pair * tf_pair,
            ],
            dim=-1,
        )
        logits = self.mlp(pair_features).squeeze(-1)

        if self.dna_bias is not None:
            logits = logits + self.dna_bias(dna).squeeze(-1)
        if self.tf_bias is not None:
            logits = logits + self.tf_bias(tf)
        return logits.unsqueeze(1)


class DenseProteinLocalAttention(nn.Module):
    """Dense protein-conditioned scorer with local DNA self-attention.

    DNA positions first exchange information within a configurable local window.
    The resulting task-adapted DNA vectors are scored against adapted
    mean-pooled TF protein embeddings using either a multi-head bilinear scorer
    or a small interaction MLP.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        tf_embedding_dropout: float = 0.0,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer: str = "multihead_bilinear",
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_hidden_dim: int = 128,
        scorer_bias_mode: str = "tf",
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if not 0.0 <= tf_embedding_dropout < 1.0:
            raise ValueError("tf_embedding_dropout must be in [0, 1)")
        if dna_attention_layers <= 0:
            raise ValueError("dna_attention_layers must be positive")
        if protein_noise_std < 0.0:
            raise ValueError("protein_noise_std must be non-negative")
        if scorer not in {"multihead_bilinear", "mlp"}:
            raise ValueError("scorer must be 'multihead_bilinear' or 'mlp'")

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        self.tf_embedding_dropout = float(tf_embedding_dropout)
        self.protein_noise_std = float(protein_noise_std)
        self.protein_l2_normalize = bool(protein_l2_normalize)

        embedding_dim = int(tf_embeddings.shape[1])
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dna_attention = nn.ModuleList(
            [
                LocalSelfAttentionBlock(
                    hidden_channels,
                    num_heads=dna_attention_heads,
                    window_bp=dna_attention_window_bp,
                    ffn_multiplier=dna_attention_ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(dna_attention_layers)
            ]
        )
        self.dna_adapter = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        self.protein_adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        if scorer == "multihead_bilinear":
            self.scorer = MultiHeadBilinearScorer(
                hidden_channels,
                num_heads=scorer_heads,
                head_dim=scorer_pair_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )
        else:
            self.scorer = MLPInteractionScorer(
                hidden_channels,
                pair_dim=scorer_pair_dim,
                scorer_hidden_dim=scorer_hidden_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dna = self.input_projection(x).transpose(1, 2)
        for block in self.dna_attention:
            dna = block(dna)
        dna = self.dna_adapter(dna)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        if self.training and self.tf_embedding_dropout > 0.0:
            tf_embeddings = F.dropout(
                tf_embeddings,
                p=self.tf_embedding_dropout,
                training=True,
            )
        if self.training and self.protein_noise_std > 0.0:
            tf_embeddings = (
                tf_embeddings
                + torch.randn_like(tf_embeddings) * self.protein_noise_std
            )
        tf_features = self.protein_adapter(tf_embeddings)
        if self.protein_l2_normalize:
            tf_features = F.normalize(tf_features, p=2, dim=-1)

        return self.scorer(dna, tf_features)


def _split_channels(total_channels: int, n_parts: int) -> list[int]:
    if total_channels <= 0:
        raise ValueError("total_channels must be positive")
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if total_channels < n_parts:
        raise ValueError("total_channels must be at least the number of parts")
    base = total_channels // n_parts
    remainder = total_channels % n_parts
    return [base + (1 if idx < remainder else 0) for idx in range(n_parts)]


class DenseProteinMotifCNN(nn.Module):
    """Protein-conditioned dense scorer with an explicit motif-CNN DNA stem.

    The DNA encoder starts with several odd-width convolutions over the raw
    sequence to expose reusable motif-like channels, then adds residual dilated
    context. The scoring path is shared with ``DenseProteinLocalAttention`` so
    the main ablation is DNA encoding rather than the protein/interaction head.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (7, 11, 15),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        tf_embedding_dropout: float = 0.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer: str = "multihead_bilinear",
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_hidden_dim: int = 128,
        scorer_bias_mode: str = "tf",
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if not motif_kernel_sizes:
            raise ValueError("motif_kernel_sizes must not be empty")
        if any(size <= 0 or size % 2 == 0 for size in motif_kernel_sizes):
            raise ValueError("motif_kernel_sizes must contain positive odd integers")
        if not 0.0 <= tf_embedding_dropout < 1.0:
            raise ValueError("tf_embedding_dropout must be in [0, 1)")
        if protein_noise_std < 0.0:
            raise ValueError("protein_noise_std must be non-negative")
        if scorer not in {"multihead_bilinear", "mlp"}:
            raise ValueError("scorer must be 'multihead_bilinear' or 'mlp'")

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        self.tf_embedding_dropout = float(tf_embedding_dropout)
        self.protein_noise_std = float(protein_noise_std)
        self.protein_l2_normalize = bool(protein_l2_normalize)

        stem_channels = _split_channels(hidden_channels, len(motif_kernel_sizes))
        self.motif_stem = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    out_channels,
                    kernel_size=motif_kernel,
                    padding=motif_kernel // 2,
                )
                for out_channels, motif_kernel in zip(stem_channels, motif_kernel_sizes)
            ]
        )
        self.stem_norm = nn.BatchNorm1d(hidden_channels)
        self.stem_activation = nn.GELU()
        self.stem_dropout = nn.Dropout(dropout)
        self.context_blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.dna_adapter = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )

        embedding_dim = int(tf_embeddings.shape[1])
        self.protein_adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        if scorer == "multihead_bilinear":
            self.scorer = MultiHeadBilinearScorer(
                hidden_channels,
                num_heads=scorer_heads,
                head_dim=scorer_pair_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )
        else:
            self.scorer = MLPInteractionScorer(
                hidden_channels,
                pair_dim=scorer_pair_dim,
                scorer_hidden_dim=scorer_hidden_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dna = torch.cat([conv(x) for conv in self.motif_stem], dim=1)
        dna = self.stem_dropout(self.stem_activation(self.stem_norm(dna)))
        for block in self.context_blocks:
            dna = block(dna)
        dna = self.dna_adapter(dna).transpose(1, 2)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        if self.training and self.tf_embedding_dropout > 0.0:
            tf_embeddings = F.dropout(
                tf_embeddings,
                p=self.tf_embedding_dropout,
                training=True,
            )
        if self.training and self.protein_noise_std > 0.0:
            tf_embeddings = tf_embeddings + torch.randn_like(tf_embeddings) * self.protein_noise_std
        tf_features = self.protein_adapter(tf_embeddings)
        if self.protein_l2_normalize:
            tf_features = F.normalize(tf_features, p=2, dim=-1)

        return self.scorer(dna, tf_features)


class DenseProteinResDilatedCrossAttention(nn.Module):
    """Protein-conditioned dense scorer with a local CNN path plus attention.

    The local bilinear scorer is the main path and keeps the same peak-localizing
    inductive bias as ``DenseProteinResDilatedCNN``. A TransBind-style
    protein-query attention branch contributes a small residual score, initialized
    near zero so it can add contextual signal without dominating early training.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        tf_embedding_dropout: float = 0.0,
        cross_attention_gate_logit_init: float = -3.0,
        cross_attention_context_pool_sizes: tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if not 0.0 <= tf_embedding_dropout < 1.0:
            raise ValueError("tf_embedding_dropout must be in [0, 1)")
        if not cross_attention_context_pool_sizes:
            raise ValueError("cross_attention_context_pool_sizes must not be empty")
        if any(size <= 0 for size in cross_attention_context_pool_sizes):
            raise ValueError("cross_attention_context_pool_sizes must be positive")

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        self.tf_embedding_dropout = float(tf_embedding_dropout)
        embedding_dim = int(tf_embeddings.shape[1])
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.dna_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.tf_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dna_bias = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.tf_bias = nn.Linear(hidden_channels, 1)

        self.context_pools = nn.ModuleList(
            [
                nn.MaxPool1d(kernel_size=size, stride=size)
                for size in cross_attention_context_pool_sizes
            ]
        )
        num_heads = min(8, hidden_channels)
        while hidden_channels % num_heads != 0:
            num_heads -= 1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.attention_tf_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.attention_position_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.attention_log_scale = nn.Parameter(
            torch.tensor(float(cross_attention_gate_logit_init))
        )
        self.score_scale = hidden_channels ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dna = self.stem(x)
        for block in self.blocks:
            dna = block(dna)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        if self.training and self.tf_embedding_dropout > 0.0:
            tf_embeddings = F.dropout(
                tf_embeddings,
                p=self.tf_embedding_dropout,
                training=True,
            )
        tf_features = self.tf_projection(tf_embeddings)

        dna_features = self.dna_projection(dna)
        local_logits = torch.einsum("bcl,tc->btl", dna_features, tf_features)
        local_logits = local_logits * self.score_scale
        local_logits = local_logits + self.dna_bias(dna)
        local_logits = local_logits + self.tf_bias(tf_features).view(1, -1, 1)

        context = torch.cat(
            [pool(dna).transpose(1, 2) for pool in self.context_pools],
            dim=1,
        )
        tf_query = tf_features.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            query=tf_query,
            key=context,
            value=context,
            need_weights=False,
        )
        attended = self.attention_norm(attended + tf_query)
        global_dna = context.mean(dim=1).unsqueeze(1).expand(-1, attended.shape[1], -1)
        attention_tf = self.attention_tf_projection(
            torch.cat([attended, global_dna], dim=-1)
        )
        attention_pos = self.attention_position_projection(dna)
        attention_logits = torch.einsum("bcl,btc->btl", attention_pos, attention_tf)
        attention_logits = attention_logits * self.score_scale

        attention_scale = torch.sigmoid(self.attention_log_scale)
        return local_logits + attention_scale * attention_logits


class DenseTransBindCnnLstmAttention(nn.Module):
    """Dense TransBind-style scorer with CNN, BiLSTM, self-attention, and TF queries.

    The pooled CNN/BiLSTM/Transformer branch follows the TransBind paper more
    closely and provides contextual DNA keys/values for protein-query
    cross-attention. A full-resolution CNN branch is kept for dense promoter
    scoring, so the output remains ``[batch, TF, position]``.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 320,
        kernel_size: int = 26,
        pool_size: int = 13,
        lstm_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even for the bidirectional LSTM")
        if hidden_channels % transformer_heads != 0:
            raise ValueError("hidden_channels must be divisible by transformer_heads")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        embedding_dim = int(tf_embeddings.shape[1])

        self.local_dna = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels // 2,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.tf_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=transformer_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_channels)
        self.tf_context_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.position_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.position_bias = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.tf_bias = nn.Linear(hidden_channels, 1)
        self.score_scale = hidden_channels ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local = self.local_dna(x)

        context = self.context_pool(local).transpose(1, 2)
        context, _ = self.bilstm(context)
        context = self.self_attention(context)
        global_dna = context.mean(dim=1)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        tf_query = self.tf_projection(tf_embeddings)
        tf_query = tf_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            query=tf_query,
            key=context,
            value=context,
            need_weights=False,
        )
        attended = self.cross_norm(attended + tf_query)

        global_dna = global_dna.unsqueeze(1).expand(-1, attended.shape[1], -1)
        tf_context = self.tf_context_projection(torch.cat([attended, global_dna], dim=-1))

        position_features = self.position_projection(local)
        logits = torch.einsum("bcl,btc->btl", position_features, tf_context)
        logits = logits * self.score_scale
        logits = logits + self.position_bias(local)
        logits = logits + self.tf_bias(tf_context)
        return logits

class DenseMotifDilatedAttentionCNN(nn.Module):
    """DNA-only dense base scorer with motif, dilated, and attention context.

    The first stage uses motif-scale convolutions over the input sequence. The
    residual dilated stack then broadens the local receptive field, and the
    self-attention blocks let each position use configurable local or full
    promoter context before the final per-position head.
    """

    def __init__(
        self,
        n_tfs: int,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
    ) -> None:
        super().__init__()
        if n_tfs <= 0:
            raise ValueError("n_tfs must be positive")
        if not motif_kernel_sizes:
            raise ValueError("motif_kernel_sizes must not be empty")
        if any(size <= 0 or size % 2 == 0 for size in motif_kernel_sizes):
            raise ValueError("motif_kernel_sizes must contain positive odd integers")
        if dna_attention_heads <= 0:
            raise ValueError("dna_attention_heads must be positive")
        if hidden_channels % dna_attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by dna_attention_heads")
        if dna_attention_layers < 0:
            raise ValueError("dna_attention_layers must be non-negative")

        stem_channels = _split_channels(hidden_channels, len(motif_kernel_sizes))
        self.motif_stem = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    out_channels,
                    kernel_size=motif_kernel,
                    padding=motif_kernel // 2,
                )
                for out_channels, motif_kernel in zip(stem_channels, motif_kernel_sizes)
            ]
        )
        self.stem_norm = nn.BatchNorm1d(hidden_channels)
        self.stem_activation = nn.GELU()
        self.stem_dropout = nn.Dropout(dropout)

        self.context_blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.attention_blocks = nn.ModuleList(
            [
                LocalSelfAttentionBlock(
                    hidden_channels,
                    num_heads=dna_attention_heads,
                    window_bp=dna_attention_window_bp,
                    ffn_multiplier=dna_attention_ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(dna_attention_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_tfs),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dna = torch.cat([conv(x) for conv in self.motif_stem], dim=1)
        dna = self.stem_dropout(self.stem_activation(self.stem_norm(dna)))
        for block in self.context_blocks:
            dna = block(dna)

        dna = dna.transpose(1, 2)
        for block in self.attention_blocks:
            dna = block(dna)

        return dna

    def score_features(self, dna: torch.Tensor) -> torch.Tensor:
        return self.head(dna).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_features(self.encode(x))


class DenseMotifDilatedAttentionMultiTaskCNN(DenseMotifDilatedAttentionCNN):
    """DNA-only dense scorer with separate TF-specific and merged-site heads."""

    def __init__(
        self,
        n_tfs: int,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
    ) -> None:
        if n_tfs < 2:
            raise ValueError(
                "DenseMotifDilatedAttentionMultiTaskCNN needs at least one "
                "TF-specific output plus one merged output"
            )
        super().__init__(
            n_tfs=1,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
        )
        del self.head
        self.tf_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_tfs - 1),
        )
        self.merged_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def score_features(self, dna: torch.Tensor) -> torch.Tensor:
        tf_logits = self.tf_head(dna)
        merged_logits = self.merged_head(dna)
        return torch.cat([tf_logits, merged_logits], dim=-1).transpose(1, 2)


class DenseProteinResidualBilinearCNN(nn.Module):
    """Protein-conditioned dense scorer built around a DNA-only prior.

    The DNA branch mirrors ``DenseMotifDilatedAttentionCNN`` and produces both a
    generic binding prior and rich per-position features. A protein adapter maps
    fixed TF embeddings into the same hidden space, and a multi-head bilinear
    scorer adds a TF-specific residual to the DNA prior.
    """

    supports_tf_indices = True
    supports_pairwise_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
        tf_embedding_dropout: float = 0.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_bias_mode: str = "tf",
        protein_delta_gate_logit_init: float = -3.0,
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if not 0.0 <= tf_embedding_dropout < 1.0:
            raise ValueError("tf_embedding_dropout must be in [0, 1)")
        if protein_noise_std < 0.0:
            raise ValueError("protein_noise_std must be non-negative")

        self.dna_model = DenseMotifDilatedAttentionCNN(
            n_tfs=1,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
        )
        self.register_buffer("tf_embeddings", tf_embeddings.float())
        self.tf_embedding_dropout = float(tf_embedding_dropout)
        self.protein_noise_std = float(protein_noise_std)
        self.protein_l2_normalize = bool(protein_l2_normalize)
        self.dna_frozen = False

        embedding_dim = int(tf_embeddings.shape[1])
        self.protein_adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        self.residual_scorer = MultiHeadBilinearScorer(
            hidden_channels,
            num_heads=scorer_heads,
            head_dim=scorer_pair_dim,
            dropout=dropout,
            bias_mode=scorer_bias_mode,
        )
        self.protein_delta_gate_logit = nn.Parameter(
            torch.tensor(float(protein_delta_gate_logit_init))
        )

    def freeze_dna_branch(self) -> None:
        for parameter in self.dna_model.parameters():
            parameter.requires_grad = False
        self.dna_frozen = True
        self.dna_model.eval()

    def train(self, mode: bool = True) -> "DenseProteinResidualBilinearCNN":
        super().train(mode)
        if self.dna_frozen:
            self.dna_model.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
        pairwise_tf_indices: bool = False,
    ) -> torch.Tensor:
        dna_features = self.dna_model.encode(x)
        dna_prior = self.dna_model.score_features(dna_features)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            if pairwise_tf_indices and (
                tf_indices.ndim != 1 or tf_indices.shape[0] != x.shape[0]
            ):
                raise ValueError(
                    "pairwise_tf_indices=True requires tf_indices with shape [batch]"
                )
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        if self.training and self.tf_embedding_dropout > 0.0:
            tf_embeddings = F.dropout(
                tf_embeddings,
                p=self.tf_embedding_dropout,
                training=True,
            )
        if self.training and self.protein_noise_std > 0.0:
            tf_embeddings = tf_embeddings + torch.randn_like(tf_embeddings) * self.protein_noise_std
        protein_features = self.protein_adapter(tf_embeddings)
        if self.protein_l2_normalize:
            protein_features = F.normalize(protein_features, p=2, dim=-1)

        if pairwise_tf_indices:
            residual = self.residual_scorer.forward_pairwise(
                dna_features,
                protein_features,
            )
        else:
            residual = self.residual_scorer(dna_features, protein_features)
        residual_scale = torch.sigmoid(self.protein_delta_gate_logit)
        return dna_prior + residual_scale * residual


class DenseProteinDirectScorerCNN(nn.Module):
    """Protein-conditioned dense scorer without an additive DNA-only prior.

    This model reuses ``DenseMotifDilatedAttentionCNN`` as a contextual DNA
    encoder, but discards its final DNA-only output head during scoring. The
    final logit is produced directly from the DNA position feature and the
    adapted TF protein embedding. This is useful when a generic "any TF binds"
    prior is too broad for TF-specific held-out transfer.
    """

    supports_tf_indices = True
    supports_pairwise_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
        tf_embedding_dropout: float = 0.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer: str = "multihead_bilinear",
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_hidden_dim: int = 128,
        scorer_bias_mode: str = "tf",
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if not 0.0 <= tf_embedding_dropout < 1.0:
            raise ValueError("tf_embedding_dropout must be in [0, 1)")
        if protein_noise_std < 0.0:
            raise ValueError("protein_noise_std must be non-negative")
        if scorer not in {"multihead_bilinear", "mlp"}:
            raise ValueError("scorer must be 'multihead_bilinear' or 'mlp'")

        self.dna_model = DenseMotifDilatedAttentionCNN(
            n_tfs=1,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
        )
        self.register_buffer("tf_embeddings", tf_embeddings.float())
        self.tf_embedding_dropout = float(tf_embedding_dropout)
        self.protein_noise_std = float(protein_noise_std)
        self.protein_l2_normalize = bool(protein_l2_normalize)
        self.dna_frozen = False

        embedding_dim = int(tf_embeddings.shape[1])
        self.dna_adapter = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        self.protein_adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        if scorer == "multihead_bilinear":
            self.scorer = MultiHeadBilinearScorer(
                hidden_channels,
                num_heads=scorer_heads,
                head_dim=scorer_pair_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )
        else:
            self.scorer = MLPInteractionScorer(
                hidden_channels,
                pair_dim=scorer_pair_dim,
                scorer_hidden_dim=scorer_hidden_dim,
                dropout=dropout,
                bias_mode=scorer_bias_mode,
            )

    def freeze_dna_branch(self) -> None:
        for parameter in self.dna_model.parameters():
            parameter.requires_grad = False
        self.dna_frozen = True
        self.dna_model.eval()

    def train(self, mode: bool = True) -> "DenseProteinDirectScorerCNN":
        super().train(mode)
        if self.dna_frozen:
            self.dna_model.eval()
        return self

    def _protein_features(self, tf_indices: torch.Tensor | None) -> torch.Tensor:
        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        if self.training and self.tf_embedding_dropout > 0.0:
            tf_embeddings = F.dropout(
                tf_embeddings,
                p=self.tf_embedding_dropout,
                training=True,
            )
        if self.training and self.protein_noise_std > 0.0:
            tf_embeddings = (
                tf_embeddings
                + torch.randn_like(tf_embeddings) * self.protein_noise_std
            )

        protein_features = self.protein_adapter(tf_embeddings)
        if self.protein_l2_normalize:
            protein_features = F.normalize(protein_features, p=2, dim=-1)
        return protein_features

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
        pairwise_tf_indices: bool = False,
    ) -> torch.Tensor:
        dna_features = self.dna_adapter(self.dna_model.encode(x))

        if pairwise_tf_indices and (
            tf_indices is None or tf_indices.ndim != 1 or tf_indices.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "pairwise_tf_indices=True requires tf_indices with shape [batch]"
            )

        protein_features = self._protein_features(tf_indices)
        if pairwise_tf_indices:
            return self.scorer.forward_pairwise(dna_features, protein_features)
        return self.scorer(dna_features, protein_features)


class DenseProteinFiLMMotifCNN(DenseProteinDirectScorerCNN):
    """Dense TFBS scorer that conditions the DNA encoder with FiLM.

    A TF embedding generates feature-wise affine parameters after the motif
    stem, each residual dilated block, and each local self-attention block.
    Unlike late fusion, the TF therefore changes the DNA features that later
    context blocks receive. The model still returns one score per DNA base.
    """

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
        tf_embedding_dropout: float = 0.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer: str = "multihead_bilinear",
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_hidden_dim: int = 128,
        scorer_bias_mode: str = "tf",
        film_eval_tf_chunk_size: int = 1,
    ) -> None:
        super().__init__(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
        )
        if film_eval_tf_chunk_size <= 0:
            raise ValueError("film_eval_tf_chunk_size must be positive")

        self.hidden_channels = int(hidden_channels)
        self.film_eval_tf_chunk_size = int(film_eval_tf_chunk_size)
        self.n_film_sites = (
            1
            + len(self.dna_model.context_blocks)
            + len(self.dna_model.attention_blocks)
        )
        self.film_generator = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.n_film_sites * 2 * hidden_channels),
        )

        # FiLM starts as the identity so a pretrained DNA encoder is not
        # disrupted before protein-conditioned gradients have shaped the adapter.
        final_layer = self.film_generator[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("FiLM generator must end in a linear layer")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def _film_parameters(
        self,
        protein_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = self.film_generator(protein_features).view(
            protein_features.shape[0],
            self.n_film_sites,
            2,
            self.hidden_channels,
        )
        # Bound scales make early fine-tuning of the frozen/pretrained encoder
        # substantially more stable than unconstrained affine modulation.
        gamma = parameters[:, :, 0, :].tanh()
        beta = parameters[:, :, 1, :]
        return gamma, beta

    @staticmethod
    def _apply_film_conv(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

    @staticmethod
    def _apply_film_sequence(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def _encode_pairwise_conditioned(
        self,
        x: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        if protein_features.ndim != 2 or protein_features.shape[0] != x.shape[0]:
            raise ValueError(
                "Pairwise protein features must have shape [batch, hidden_channels]"
            )
        gamma, beta = self._film_parameters(protein_features)

        dna = torch.cat([conv(x) for conv in self.dna_model.motif_stem], dim=1)
        dna = self.dna_model.stem_dropout(
            self.dna_model.stem_activation(self.dna_model.stem_norm(dna))
        )
        film_index = 0
        dna = self._apply_film_conv(dna, gamma[:, film_index], beta[:, film_index])
        film_index += 1

        for block in self.dna_model.context_blocks:
            dna = block(dna)
            dna = self._apply_film_conv(dna, gamma[:, film_index], beta[:, film_index])
            film_index += 1

        dna = dna.transpose(1, 2)
        for block in self.dna_model.attention_blocks:
            dna = block(dna)
            dna = self._apply_film_sequence(
                dna,
                gamma[:, film_index],
                beta[:, film_index],
            )
            film_index += 1

        if film_index != self.n_film_sites:
            raise RuntimeError("FiLM site count does not match DNA encoder blocks")
        return dna

    def _score_pairwise(
        self,
        x: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        dna_features = self.dna_adapter(
            self._encode_pairwise_conditioned(x, protein_features)
        )
        return self.scorer.forward_pairwise(dna_features, protein_features)

    def _score_all_tfs(
        self,
        x: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score all requested TFs in memory-safe chunks during dense evaluation."""
        batch_size = x.shape[0]
        outputs = []
        for protein_chunk in protein_features.split(self.film_eval_tf_chunk_size, dim=0):
            n_chunk_tfs = protein_chunk.shape[0]
            expanded_x = (
                x.unsqueeze(1)
                .expand(-1, n_chunk_tfs, -1, -1)
                .reshape(batch_size * n_chunk_tfs, x.shape[1], x.shape[2])
            )
            expanded_proteins = (
                protein_chunk.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .reshape(batch_size * n_chunk_tfs, self.hidden_channels)
            )
            logits = self._score_pairwise(expanded_x, expanded_proteins)
            outputs.append(logits.view(batch_size, n_chunk_tfs, x.shape[-1]))
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
        pairwise_tf_indices: bool = False,
    ) -> torch.Tensor:
        if pairwise_tf_indices:
            if (
                tf_indices is None
                or tf_indices.ndim != 1
                or tf_indices.shape[0] != x.shape[0]
            ):
                raise ValueError(
                    "pairwise_tf_indices=True requires tf_indices with shape [batch]"
                )
            return self._score_pairwise(x, self._protein_features(tf_indices))

        return self._score_all_tfs(x, self._protein_features(tf_indices))


class DenseProteinWindowLocalizationCNN(DenseProteinDirectScorerCNN):
    """Protein-conditioned localization model with coupled window evidence.

    The localization head emits one TF-specific logit per base. A second head
    is not learned independently; instead it pools the base logits into a
    differentiable window-level logit. This makes the window-level task teach
    the position-level scorer rather than letting a separate classifier solve a
    coarser task on its own.
    """

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        input_channels: int = 4,
        hidden_channels: int = 128,
        motif_kernel_sizes: tuple[int, ...] = (21,),
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        dna_attention_window_bp: int = 50,
        dna_attention_layers: int = 2,
        dna_attention_heads: int = 8,
        dna_attention_ffn_multiplier: float = 4.0,
        tf_embedding_dropout: float = 0.0,
        protein_noise_std: float = 0.0,
        protein_l2_normalize: bool = False,
        scorer: str = "multihead_bilinear",
        scorer_heads: int = 8,
        scorer_pair_dim: int = 32,
        scorer_hidden_dim: int = 128,
        scorer_bias_mode: str = "tf",
        window_pooling: str = "topk_logmeanexp",
        window_pooling_top_k: int = 10,
    ) -> None:
        super().__init__(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
        )
        if window_pooling not in {"max", "logsumexp", "topk_logmeanexp"}:
            raise ValueError(
                "window_pooling must be one of max, logsumexp, topk_logmeanexp"
            )
        if window_pooling_top_k <= 0:
            raise ValueError("window_pooling_top_k must be positive")
        self.window_pooling = window_pooling
        self.window_pooling_top_k = int(window_pooling_top_k)

    def train(self, mode: bool = True) -> "DenseProteinWindowLocalizationCNN":
        super().train(mode)
        if self.dna_frozen:
            self.dna_model.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
        pairwise_tf_indices: bool = False,
    ) -> dict[str, torch.Tensor]:
        dna_features = self.dna_adapter(self.dna_model.encode(x))

        if pairwise_tf_indices and (
            tf_indices is None or tf_indices.ndim != 1 or tf_indices.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "pairwise_tf_indices=True requires tf_indices with shape [batch]"
            )

        protein_features = self._protein_features(tf_indices)
        if pairwise_tf_indices:
            localization_logits = self.scorer.forward_pairwise(
                dna_features,
                protein_features,
            )
        else:
            localization_logits = self.scorer(dna_features, protein_features)

        return {
            "logits": localization_logits,
            "window_logits": self._pool_window_logits(localization_logits),
        }

    def _pool_window_logits(self, localization_logits: torch.Tensor) -> torch.Tensor:
        if self.window_pooling == "max":
            return localization_logits.amax(dim=-1)
        if self.window_pooling == "logsumexp":
            return torch.logsumexp(localization_logits, dim=-1)

        k = min(self.window_pooling_top_k, localization_logits.shape[-1])
        top_logits = localization_logits.topk(k, dim=-1).values
        return torch.logsumexp(top_logits, dim=-1) - math.log(k)


class TransBindLite(nn.Module):
    """Protein-conditioned TF binding model.

    The DNA trunk produces positional sequence features. Each fixed TF protein
    embedding acts as a query over those positions, yielding one logit per TF.
    """

    supports_tf_indices = True

    def __init__(
        self,
        n_tfs: int,
        tf_embeddings: torch.Tensor,
        *,
        hidden_channels: int = 320,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        num_heads: int = 8,
        use_max_pool: bool = True,
        learned_tf_bias: bool = False,
    ) -> None:
        super().__init__()
        if tf_embeddings.ndim != 2:
            raise ValueError("tf_embeddings must have shape [n_tfs, embedding_dim]")
        if tf_embeddings.shape[0] != n_tfs:
            raise ValueError(
                f"Expected {n_tfs} TF embeddings, got {tf_embeddings.shape[0]}"
            )
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")

        self.register_buffer("tf_embeddings", tf_embeddings.float())
        embedding_dim = int(tf_embeddings.shape[1])
        dna_layers: list[nn.Module] = [
            nn.Conv1d(in_channels=4, out_channels=hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        ]
        if use_max_pool:
            dna_layers.append(nn.MaxPool1d(kernel_size=2))
        dna_layers.extend(
            ResidualDilatedBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.dna_encoder = nn.Sequential(*dna_layers)
        self.tf_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.tf_bias = nn.Parameter(torch.zeros(n_tfs)) if learned_tf_bias else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dna = self.dna_encoder(x).transpose(1, 2)
        global_dna = dna.mean(dim=1)

        tf_embeddings = self.tf_embeddings
        if tf_indices is not None:
            tf_embeddings = tf_embeddings.index_select(0, tf_indices)
        tf_query = self.tf_projection(tf_embeddings)
        tf_query = tf_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(query=tf_query, key=dna, value=dna, need_weights=False)
        attended = self.norm(attended + tf_query)

        global_dna = global_dna.unsqueeze(1).expand(-1, attended.shape[1], -1)
        logits = self.classifier(torch.cat([attended, global_dna], dim=-1)).squeeze(-1)
        if self.tf_bias is not None:
            tf_bias = self.tf_bias
            if tf_indices is not None:
                tf_bias = tf_bias.index_select(0, tf_indices)
            logits = logits + tf_bias
        return logits


MODEL_NAMES = ("small_cnn", "res_dilated_cnn", "transbind_lite")
DENSE_MODEL_NAMES = (
    "dense_small_cnn",
    "dense_res_dilated_cnn",
    "dense_motif_dilated_attention_cnn",
    "dense_motif_dilated_attention_multitask_cnn",
    "dense_protein_residual_bilinear_cnn",
    "dense_protein_direct_scorer_cnn",
    "dense_protein_film_motif_cnn",
    "dense_protein_window_localization_cnn",
    "dense_protein_res_dilated_cnn",
    "dense_protein_local_attention",
    "dense_protein_motif_cnn",
    "dense_protein_res_dilated_crossattention",
    "dense_transbind_cnn_lstm_attention",
)


def parse_dilations(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        dilations = value
    elif isinstance(value, list):
        dilations = tuple(value)
    else:
        dilations = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dilations:
        raise ValueError("At least one dilation must be provided")
    if any(dilation <= 0 for dilation in dilations):
        raise ValueError("Dilations must be positive integers")
    return dilations


def parse_int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    values = parse_dilations(value)
    return values


def build_model(
    model_name: str,
    *,
    n_tfs: int,
    tf_embeddings: torch.Tensor | None = None,
    hidden_channels: int = 128,
    kernel_size: int = 7,
    dropout: float = 0.1,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    num_heads: int = 8,
    transbind_dilations: tuple[int, ...] = (1, 2, 4),
    transbind_use_max_pool: bool = True,
    transbind_tf_bias: bool = False,
) -> nn.Module:
    if model_name == "small_cnn":
        return smallCNN(n_tfs=n_tfs)
    if model_name == "res_dilated_cnn":
        return ResDilatedCNN(
            n_tfs=n_tfs,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
        )
    if model_name == "transbind_lite":
        if tf_embeddings is None:
            raise ValueError("transbind_lite requires tf_embeddings")
        return TransBindLite(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilations=transbind_dilations,
            dropout=dropout,
            num_heads=num_heads,
            use_max_pool=transbind_use_max_pool,
            learned_tf_bias=transbind_tf_bias,
        )
    raise ValueError(f"Unknown model: {model_name!r}. Choose one of {MODEL_NAMES}.")


def build_dense_model(
    model_name: str,
    *,
    n_tfs: int,
    input_channels: int,
    tf_embeddings: torch.Tensor | None = None,
    hidden_channels: int = 128,
    kernel_size: int = 7,
    dropout: float = 0.1,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    tf_embedding_dropout: float = 0.0,
    cross_attention_gate_logit_init: float = -3.0,
    cross_attention_context_pool_sizes: tuple[int, ...] = (4,),
    dna_attention_window_bp: int = 50,
    dna_attention_layers: int = 2,
    dna_attention_heads: int = 8,
    dna_attention_ffn_multiplier: float = 4.0,
    motif_kernel_sizes: tuple[int, ...] = (7, 11, 15),
    protein_noise_std: float = 0.0,
    protein_l2_normalize: bool = False,
    protein_delta_gate_logit_init: float = -3.0,
    scorer: str = "multihead_bilinear",
    scorer_heads: int = 8,
    scorer_pair_dim: int = 32,
    scorer_hidden_dim: int = 128,
    scorer_bias_mode: str = "tf",
    film_eval_tf_chunk_size: int = 1,
    window_pooling: str = "topk_logmeanexp",
    window_pooling_top_k: int = 10,
) -> nn.Module:
    if model_name == "dense_small_cnn":
        return DenseSmallCNN(
            n_tfs=n_tfs,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    if model_name == "dense_res_dilated_cnn":
        return DenseResDilatedCNN(
            n_tfs=n_tfs,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
        )
    if model_name == "dense_motif_dilated_attention_cnn":
        return DenseMotifDilatedAttentionCNN(
            n_tfs=n_tfs,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
        )
    if model_name == "dense_motif_dilated_attention_multitask_cnn":
        return DenseMotifDilatedAttentionMultiTaskCNN(
            n_tfs=n_tfs,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
        )
    if model_name == "dense_protein_residual_bilinear_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_residual_bilinear_cnn requires tf_embeddings")
        if scorer != "multihead_bilinear":
            raise ValueError(
                "dense_protein_residual_bilinear_cnn currently supports only "
                "scorer='multihead_bilinear'"
            )
        return DenseProteinResidualBilinearCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_bias_mode=scorer_bias_mode,
            protein_delta_gate_logit_init=protein_delta_gate_logit_init,
        )
    if model_name == "dense_protein_direct_scorer_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_direct_scorer_cnn requires tf_embeddings")
        return DenseProteinDirectScorerCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
        )
    if model_name == "dense_protein_film_motif_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_film_motif_cnn requires tf_embeddings")
        return DenseProteinFiLMMotifCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
            film_eval_tf_chunk_size=film_eval_tf_chunk_size,
        )
    if model_name == "dense_protein_window_localization_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_window_localization_cnn requires tf_embeddings")
        return DenseProteinWindowLocalizationCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
            window_pooling=window_pooling,
            window_pooling_top_k=window_pooling_top_k,
        )
    if model_name == "dense_protein_res_dilated_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_res_dilated_cnn requires tf_embeddings")
        return DenseProteinResDilatedCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
        )
    if model_name == "dense_protein_local_attention":
        if tf_embeddings is None:
            raise ValueError("dense_protein_local_attention requires tf_embeddings")
        return DenseProteinLocalAttention(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            tf_embedding_dropout=tf_embedding_dropout,
            dna_attention_window_bp=dna_attention_window_bp,
            dna_attention_layers=dna_attention_layers,
            dna_attention_heads=dna_attention_heads,
            dna_attention_ffn_multiplier=dna_attention_ffn_multiplier,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
        )
    if model_name == "dense_protein_motif_cnn":
        if tf_embeddings is None:
            raise ValueError("dense_protein_motif_cnn requires tf_embeddings")
        return DenseProteinMotifCNN(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            motif_kernel_sizes=motif_kernel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            tf_embedding_dropout=tf_embedding_dropout,
            protein_noise_std=protein_noise_std,
            protein_l2_normalize=protein_l2_normalize,
            scorer=scorer,
            scorer_heads=scorer_heads,
            scorer_pair_dim=scorer_pair_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            scorer_bias_mode=scorer_bias_mode,
        )
    if model_name == "dense_protein_res_dilated_crossattention":
        if tf_embeddings is None:
            raise ValueError(
                "dense_protein_res_dilated_crossattention requires tf_embeddings"
            )
        return DenseProteinResDilatedCrossAttention(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            tf_embedding_dropout=tf_embedding_dropout,
            cross_attention_gate_logit_init=cross_attention_gate_logit_init,
            cross_attention_context_pool_sizes=cross_attention_context_pool_sizes,
        )
    if model_name == "dense_transbind_cnn_lstm_attention":
        if tf_embeddings is None:
            raise ValueError("dense_transbind_cnn_lstm_attention requires tf_embeddings")
        return DenseTransBindCnnLstmAttention(
            n_tfs=n_tfs,
            tf_embeddings=tf_embeddings,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown dense model: {model_name!r}. Choose one of {DENSE_MODEL_NAMES}."
    )
