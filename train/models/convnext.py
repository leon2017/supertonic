import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        kernel_size: int = 7,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        x = x + residual
        return x
