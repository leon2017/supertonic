import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .convnext import ConvNeXtBlock
from .attention import MultiHeadSelfAttention, CrossAttention


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        channels: int = 512,
        intermediate: int = 2048,
        num_convnext: int = 6,
        num_self_attn: int = 4,
        num_cross_attn: int = 2,
        num_heads: int = 4,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_proj = nn.Conv1d(embedding_dim, channels, 1)

        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate) for _ in range(num_convnext)
        ])

        self.self_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(channels),
                'attn': MultiHeadSelfAttention(channels, num_heads),
            }) for _ in range(num_self_attn)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_q': nn.LayerNorm(channels),
                'norm_kv': nn.LayerNorm(channels),
                'attn': CrossAttention(channels, channels, num_heads),
            }) for _ in range(num_cross_attn)
        ])

    def forward(self, text_ids: torch.Tensor, ref_emb: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(text_ids)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.input_proj(x)  # (B, C, L)

        for block in self.convnext_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Self-attention: (B, C, L) -> (B, L, C) -> attn -> (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        for sa_block in self.self_attn_blocks:
            residual = x
            x = sa_block['norm'](x)
            x = sa_block['attn'](x, mask=text_mask) + residual

        # Cross-attention with reference
        ref = ref_emb.transpose(1, 2)  # (B, num_tokens, C)
        for ca_block in self.cross_attn_blocks:
            residual = x
            q = ca_block['norm_q'](x)
            kv = ca_block['norm_kv'](ref)
            x = ca_block['attn'](q, kv) + residual

        x = x.transpose(1, 2)  # (B, C, L)
        return x
