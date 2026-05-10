"""
enhancer_unet.py — Stage 1: Discriminative 1-D U-Net Speech Denoiser
=====================================================================

Architecture overview:
    4-level encoder-decoder with skip connections.
    Each level uses ResBlock modules with GroupNorm + GELU.
    The model predicts the noise residual and subtracts it from the input.

    Input:  noisy waveform  [B, 1, T]
    Output: clean waveform  [B, 1, T]

    Total parameters: ~40.9 M (base_ch=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv1d → GroupNorm → GELU → Conv1d → GroupNorm + skip."""

    def __init__(self, ch: int):
        super().__init__()
        g = min(8, ch)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GroupNorm(g, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GroupNorm(g, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Down(nn.Module):
    """Downsample by 2× via strided convolution, then apply a ResBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsample by 2× via transposed convolution, concatenate skip, then ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        g = min(8, out_ch)
        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-length mismatches from strided ops
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return self.net(torch.cat([x, skip], dim=1))


class SpeechEnhancerUNet(nn.Module):
    """
    1-D U-Net that performs speech enhancement via noise residual learning.

    The model estimates the noise component in the input and subtracts it:
        output = noisy_input − predicted_noise

    This is more stable than direct clean prediction because:
    1. Noise is less structured than speech → easier target
    2. The identity shortcut means the model only needs to learn the "delta"
    3. Gradient flow is improved through the residual path

    Args:
        base_ch: Base channel count.  Encoder widths are [C, 2C, 4C, 8C, 16C].
                 Default 64 → 40.9M params.  Use 32 for a lighter model (~10M).
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        C = base_ch

        # Stem
        self.inp = nn.Conv1d(1, C, 3, padding=1)

        # Encoder path
        self.d1 = Down(C, C * 2)
        self.d2 = Down(C * 2, C * 4)
        self.d3 = Down(C * 4, C * 8)
        self.d4 = Down(C * 8, C * 16)

        # Bottleneck
        self.bot = nn.Sequential(
            ResBlock(C * 16),
            ResBlock(C * 16),
            ResBlock(C * 16),
        )

        # Decoder path with skip connections
        self.u4 = Up(C * 16, C * 8, C * 8)
        self.u3 = Up(C * 8, C * 4, C * 4)
        self.u2 = Up(C * 4, C * 2, C * 2)
        self.u1 = Up(C * 2, C, C)

        # Output head → noise estimate → subtract from input
        self.out = nn.Sequential(
            ResBlock(C),
            nn.Conv1d(C, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy: [B, 1, T] noisy waveform (values in [-1, 1])

        Returns:
            [B, 1, T] estimated clean waveform (clamped to [-1, 1])
        """
        x0 = self.inp(noisy)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        b = self.bot(x4)

        x = self.u4(b, x3)
        x = self.u3(x, x2)
        x = self.u2(x, x1)
        x = self.u1(x, x0)

        noise_est = self.out(x)
        return (noisy - noise_est).clamp(-1, 1)
