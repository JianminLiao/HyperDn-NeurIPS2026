"""
Reproduction of Afkham, Chung & Chung (2021) "Learning regularization parameters
of inverse problems via deep neural networks", Inverse Problems 37(10):105017.

Architecture follows their Experiment 3 (image deblurring with star-shaped inclusions):
  3 conv layers (5x5, BN, ReLU, MaxPool2x2) with 16→32→64 channels
  + FC head → output params

Differences from original:
  - Input: 3-channel RGB (they used 1-channel grayscale)
  - Output: configuration-dependent encoded hyperparameters. DiffPIR retains
    two training outputs (log10 effective lambda and historical zero padding);
    CBM3D predicts one output, log10 sigma_psd.
  - Single-stage training (they used two-stage for two separate heads)
  - AdamW optimizer (they used Adam with dynamic LR)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AfkhamCNN(nn.Module):
    """
    Afkham et al. (2021) Experiment 3 CNN.
    3 conv blocks (Conv 5x5 + BN + ReLU + MaxPool 2x2), channels 16→32→64.
    Followed by flatten + FC layers → output.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv 5x5, 16 channels
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: Conv 5x5, 32 channels
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: Conv 5x5, 64 channels
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # 3 hidden FC layers as described in Afkham et al. Experiment 3
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)


class AfkhamCNNFoundationWrapper(nn.Module):
    """Wrap AfkhamCNN with PGCCPiFoundationModel-compatible forward signature.

    Accepts (and ignores) regularizer_ids / data_term_ids / noise_attributes /
    observation_meta so that evaluate_pgccpi_foundation.py can use it unchanged.
    Output is zero-padded to max_param_dim.
    """

    def __init__(self, afkham: AfkhamCNN, max_param_dim: int):
        super().__init__()
        self.afkham = afkham
        self.max_param_dim = int(max_param_dim)

    def forward(
        self,
        images: torch.Tensor,
        *,
        regularizer_ids: torch.Tensor,
        data_term_ids=None,
        noise_attributes=None,
        observation_meta=None,
    ):
        raw = self.afkham(images)                        # (B, out_dim)
        B = images.shape[0]
        padded = images.new_zeros(B, self.max_param_dim)
        padded[:, : raw.shape[1]] = raw
        return {"param_pred": padded}
