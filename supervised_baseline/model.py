from __future__ import annotations

import torch
import torch.nn as nn


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


MODEL_NAMES = ("small_cnn", "res_dilated_cnn")


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
    hidden_channels: int = 128,
    kernel_size: int = 7,
    dropout: float = 0.1,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
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
    raise ValueError(f"Unknown model: {model_name!r}. Choose one of {MODEL_NAMES}.")
