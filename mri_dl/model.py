from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 1)

    @staticmethod
    def _match_size(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        th, tw = ref.shape[-2:]
        dh, dw = th - t.shape[-2], tw - t.shape[-1]
        if dh > 0 or dw > 0:
            t = F.pad(t, [max(dw//2,0), max(dw-dw//2,0), max(dh//2,0), max(dh-dh//2,0)])
        if t.shape[-2] > th:
            s = (t.shape[-2]-th)//2
            t = t[..., s:s+th, :]
        if t.shape[-1] > tw:
            s = (t.shape[-1]-tw)//2
            t = t[..., :, s:s+tw]
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._match_size(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), e1), e1], dim=1))
        return x0 + self.output_conv(d1)
