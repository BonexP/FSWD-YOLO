"""DPU-oriented modules for the FSWD-YOLO deployment candidate."""

from __future__ import annotations

import torch
from torch import nn

from .block import Bottleneck, C3k, GhostBottleneck, SimamModule
from .conv import Conv

__all__ = (
    "C2PSFCADPU",
    "C3k2DPU",
    "C3k2GhostSimAMinnerDPU",
    "DPUChannelAttentionBlock",
    "DPUSpatialAttentionBlock",
    "SplitFreeC2f",
)


class SplitFreeC2f(nn.Module):
    """C2f topology using independent projections instead of channel slicing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv_keep = Conv(c1, self.c, 1, 1)
        self.cv_process = Conv(c1, self.c, 1, 1)
        self.cv_fuse = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the split-free C2f path."""
        y = [self.cv_keep(x), self.cv_process(x)]
        y.extend(block(y[-1]) for block in self.m)
        return self.cv_fuse(torch.cat(y, dim=1))


class C3k2DPU(SplitFreeC2f):
    """Split-free replacement for C3k2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k2GhostSimAMinnerDPU(SplitFreeC2f):
    """Split-free Ghost-SimAM C2f variant used by the FSWD backbone."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(GhostBottleneck(self.c, self.c), SimamModule()) for _ in range(n)
        )


class DPUSpatialAttentionBlock(nn.Module):
    """Local spatial gate and residual pointwise feed-forward block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = Conv(channels, channels, 5, 1, g=channels, act=False)
        self.pointwise = Conv(channels, channels, 1, 1, act=False)
        self.gate = nn.Hardsigmoid()
        self.ffn = nn.Sequential(
            Conv(channels, 2 * channels, 1, 1, act=nn.Hardswish()),
            Conv(2 * channels, channels, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial gating followed by a residual feed-forward block."""
        gate = self.gate(self.pointwise(self.depthwise(x)))
        x = x + x * gate
        return x + self.ffn(x)


class DPUChannelAttentionBlock(nn.Module):
    """Squeeze-excitation channel gate built from DPU-friendly primitives."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.activate = nn.Hardswish()
        self.expand = nn.Conv2d(hidden_channels, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a pooled channel gate."""
        gate = self.gate(self.expand(self.activate(self.reduce(self.pool(x)))))
        return x * gate


class C2PSFCADPU(nn.Module):
    """Three-branch DPU replacement for C2PSFCA."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        channel_reduction: int = 4,
    ) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSFCADPU requires equal input and output channels")
        if channel_reduction <= 0:
            raise ValueError("channel_reduction must be greater than zero")

        self.c = int(c1 * e)
        self.keep_projection = Conv(c1, self.c, 1, 1)
        self.spatial_projection = Conv(c1, self.c, 1, 1)
        self.channel_projection = Conv(c1, self.c, 1, 1)
        self.spatial_blocks = nn.Sequential(*(DPUSpatialAttentionBlock(self.c) for _ in range(n)))
        self.channel_blocks = nn.Sequential(
            *(DPUChannelAttentionBlock(self.c, channel_reduction) for _ in range(n))
        )
        self.fuse = Conv(3 * self.c, c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the preserved keep, spatial, and channel branches."""
        keep = self.keep_projection(x)
        spatial = self.spatial_blocks(self.spatial_projection(x))
        channel = self.channel_blocks(self.channel_projection(x))
        return self.fuse(torch.cat((keep, spatial, channel), dim=1))
