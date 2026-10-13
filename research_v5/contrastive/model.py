"""Two-branch fully convolutional hatch embedding.

The native branch sees the drawing at full resolution. The context branch sees
a 2x average-pooled copy so its receptive field covers a wider neighborhood.
Both grids are stride 8. Each spatial vector is L2-normalized.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from constants import EMBED_DIM, INPUT_CHANNELS, STRIDE

NATIVE_SPEC = (
    (48, 1, 1),
    (64, 2, 1),
    (96, 1, 1),
    (128, 2, 1),
    (160, 1, 1),
    (192, 2, 1),
    (192, 1, 1),
    (128, 1, 2),
)
CONTEXT_SPEC = (
    (48, 1, 1),
    (64, 2, 1),
    (96, 1, 1),
    (128, 2, 1),
    (160, 1, 1),
    (160, 1, 2),
    (160, 1, 2),
    (128, 1, 4),
)

DESIGN_NOTES = (
    "Per-location channel LayerNorm replaces GroupNorm. GroupNorm statistics depend on the spatial extent of whatever tensor is passed in, so a tile would not match one forward of the whole drawing.",
    "The native stack is eight 3x3 convolutions, output stride 8, receptive field 77px. The context stack runs on a 2x average pool and ends at the same stride-8 grid; its receptive field is about 170px in native pixels.",
    "Images are white-padded on the bottom and right to a multiple of 8. The fit split contains sheets whose sides are not divisible by 8, and the pad keeps the cell grid aligned.",
    "The query crop used in training is expanded by a 160px white-paper halo. The anchor averages only cells whose centers lie inside the query box; the halo supplies their receptive field.",
)


class ChannelLayerNorm(nn.Module):
    """Normalize the channel vector at each pixel. The statistic does not pool over space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laid = x.permute(0, 2, 3, 1).contiguous()
        laid = self.norm(laid)
        return laid.permute(0, 3, 1, 2).contiguous()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False,
        )
        self.norm = ChannelLayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _stack(in_ch: int, spec: tuple[tuple[int, int, int], ...]) -> tuple[nn.Sequential, int]:
    blocks = []
    channels = in_ch
    for out_ch, stride, dilation in spec:
        blocks.append(ConvBlock(channels, out_ch, stride, dilation))
        channels = out_ch
    return nn.Sequential(*blocks), channels


class HatchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native, native_ch = _stack(INPUT_CHANNELS, NATIVE_SPEC)
        self.context, context_ch = _stack(INPUT_CHANNELS, CONTEXT_SPEC)
        self.project = nn.Conv2d(native_ch + context_ch, EMBED_DIM, kernel_size=1)
        self.embed_dim = EMBED_DIM
        self.stride = STRIDE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_stride_multiple(x)
        native = self.native(x)
        context = self.context(F.avg_pool2d(x, kernel_size=2, stride=2))
        if context.shape[-2:] != native.shape[-2:]:
            raise RuntimeError("native and context grids do not match")
        return F.normalize(self.project(torch.cat((native, context), dim=1)), p=2, dim=1)


def _require_stride_multiple(x: torch.Tensor) -> None:
    if x.shape[-2] % STRIDE or x.shape[-1] % STRIDE:
        raise RuntimeError("input sides must be multiples of 8")


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())
