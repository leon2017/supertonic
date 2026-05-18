import torch
import torch.nn as nn

from .convnext import ConvNeXtBlock
from .attention import CrossAttention


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        mel_channels: int = 228,
        channels: int = 512,
        intermediate: int = 512,
        num_convnext: int = 6,
        num_cross_attn: int = 2,
        num_heads: int = 4,
        num_style_tokens: int = 50,
        kernel_size: int = 5
    ):
        super().__init__()

        self.input_proj = nn.Linear(mel_channels, channels)

        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size)
            for _ in range(num_convnext)
        ])

        # Learnable style queries
        self.style_queries = nn.Parameter(torch.randn(1, num_style_tokens, channels))

        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_q': nn.LayerNorm(channels),
                'norm_kv': nn.LayerNorm(channels),
                'attn': CrossAttention(channels, channels, num_heads),
            }) for _ in range(num_cross_attn)
        ])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, C_mel, T)
        x = mel.transpose(1, 2)  # (B, T, C_mel)
        x = self.input_proj(x)   # (B, T, C)
        x = x.transpose(1, 2)   # (B, C, T)

        for block in self.convnext_blocks:
            x = block(x)

        # Cross-attention: style queries attend to mel features
        x = x.transpose(1, 2)  # (B, T, C)
        B = x.shape[0]
        queries = self.style_queries.expand(B, -1, -1)  # (B, num_tokens, C)

        for ca_block in self.cross_attn_blocks:
            residual = queries
            q = ca_block['norm_q'](queries)
            kv = ca_block['norm_kv'](x)
            queries = ca_block['attn'](q, kv) + residual

        # Output: (B, C, num_tokens)
        return queries.transpose(1, 2)
