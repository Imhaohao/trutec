"""Query-conditioned U-Net. Normalization is per location so a tile matches a full forward."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .constants import QUERY_SIZE, VEC_DIM


class ChannelLayerNorm(nn.Module):
    """Normalize the channel vector at each pixel. The statistic does not pool over space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laid = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(laid).permute(0, 3, 1, 2).contiguous()


class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False,
        )
        self.norm = ChannelLayerNorm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Stage(nn.Module):
    """One stride-2 convolution, then a stride-1 convolution at the coarser scale."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = Conv(in_ch, out_ch, stride=2)
        self.block = Conv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class QueryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Conv(4, 32, stride=2),
            Conv(32, 64, stride=2),
            Conv(64, 128, stride=2),
            Conv(128, 128),
        )
        self.vector = nn.Linear(128, VEC_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.layers(x)
        if int(feat.shape[-2]) != 12 or int(feat.shape[-1]) != 12:
            raise RuntimeError("query feature map must be 12x12")
        return self.vector(feat.mean(dim=(2, 3))), feat


class DrawingEncoder(nn.Module):
    """Full resolution through the 1/8 map, which is the first conditioned scale."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = Conv(4, 32)
        self.to_half = Stage(32, 64)
        self.to_quarter = Stage(64, 128)
        self.to_eighth = Stage(128, 192)

    def forward(self, x: torch.Tensor):
        half = self.to_half(self.stem(x))
        quarter = self.to_quarter(half)
        eighth = self.to_eighth(quarter)
        return half, quarter, eighth


def correlation_map(features: torch.Tensor, query_map: torch.Tensor) -> torch.Tensor:
    """Max cosine similarity between each drawing cell and the 12x12 query map."""
    drawing = F.normalize(features, dim=1)
    query = F.normalize(query_map, dim=1)
    sim = torch.matmul(drawing.flatten(2).transpose(1, 2), query.flatten(2))
    best = sim.max(dim=-1).values
    return best.reshape(features.shape[0], 1, features.shape[2], features.shape[3])


class Condition(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.film = nn.Linear(VEC_DIM, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.film.bias.data[:channels] = 1
        self.project = nn.Conv2d(128, channels, kernel_size=1, bias=False)
        self.mix = nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False)
        self.norm = ChannelLayerNorm(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor, query_map: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(vec).chunk(2, dim=1)
        filmed = features * gamma[:, :, None, None] + beta[:, :, None, None]
        corr = correlation_map(features, self.project(query_map))
        mixed = self.mix(torch.cat((filmed, corr), dim=1))
        return self.act(self.norm(mixed))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mix = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.norm = ChannelLayerNorm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.act(self.norm(self.mix(torch.cat((up, skip), dim=1))))


class HatchContextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_encoder = QueryEncoder()
        self.encoder = DrawingEncoder()
        self.cond8 = Condition(192)
        self.down16 = Stage(192, 256)
        self.cond16 = Condition(256)
        self.down32 = Stage(256, 256)
        self.context = nn.Sequential(Conv(256, 256, dilation=2), Conv(256, 256, dilation=2), Conv(256, 256, dilation=2))
        self.up16 = Up(256, 256, 192)
        self.up8 = Up(192, 192, 128)
        self.up4 = Up(128, 128, 64)
        self.up2 = Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, drawing: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _check_pair(drawing, query)
        vec, query_map = self.query_encoder(query)
        return self.decode(self.encoder(drawing), vec, query_map)

    def decode(self, front, vec: torch.Tensor, query_map: torch.Tensor) -> torch.Tensor:
        half, quarter, eighth = front
        at8 = self.cond8(eighth, query_map, vec)
        at16 = self.cond16(self.down16(at8), query_map, vec)
        deep = self.context(self.down32(at16))
        x = self.up16(deep, at16)
        x = self.up8(x, at8)
        x = self.up4(x, quarter)
        x = self.up2(x, half)
        return self.head(x)


def _check_pair(drawing: torch.Tensor, query: torch.Tensor) -> None:
    if int(drawing.shape[-2]) % 32 or int(drawing.shape[-1]) % 32:
        raise RuntimeError("drawing sides must be multiples of 32")
    if int(query.shape[-2]) != QUERY_SIZE or int(query.shape[-1]) != QUERY_SIZE:
        raise RuntimeError("query must be 96x96")


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())


def stride_count(module: nn.Module, stride: int) -> int:
    total = 0
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d) and int(layer.stride[0]) == stride:
            total += 1
    return total
