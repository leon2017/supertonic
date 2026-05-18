import torch
import torch.nn as nn
import math


def create_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).float()


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()

    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
