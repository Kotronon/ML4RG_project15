from __future__ import annotations

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


class TransBindLite(nn.Module):
    """Protein-conditioned TF binding model.

    The DNA trunk produces positional sequence features. Each fixed TF protein
    embedding acts as a query over those positions, yielding one logit per TF.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dna = self.dna_encoder(x).transpose(1, 2)
        global_dna = dna.mean(dim=1)

        tf_query = self.tf_projection(self.tf_embeddings)
        tf_query = tf_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(query=tf_query, key=dna, value=dna, need_weights=False)
        attended = self.norm(attended + tf_query)

        global_dna = global_dna.unsqueeze(1).expand(-1, attended.shape[1], -1)
        logits = self.classifier(torch.cat([attended, global_dna], dim=-1)).squeeze(-1)
        if self.tf_bias is not None:
            logits = logits + self.tf_bias
        return logits


MODEL_NAMES = ("small_cnn", "res_dilated_cnn", "transbind_lite")
DENSE_MODEL_NAMES = (
    "dense_small_cnn",
    "dense_res_dilated_cnn",
    "dense_protein_res_dilated_cnn",
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
