import torch
import torch.nn as nn

from .convnext import ConvNeXtBlock
from .attention import MultiHeadSelfAttention, CrossAttention


class DPReferenceEncoder(nn.Module):
    def __init__(self, mel_channels: int = 228, channels: int = 256, intermediate: int = 256, num_convnext: int = 4, num_cross_attn: int = 2, num_heads: int = 2, kernel_size: int = 5):
        super().__init__()

        self.input_proj = nn.Linear(mel_channels, channels)

        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size)
            for _ in range(num_convnext)
        ])

        self.style_query = nn.Parameter(torch.randn(1, 1, channels))

        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_q': nn.LayerNorm(channels),
                'norm_kv': nn.LayerNorm(channels),
                'attn': CrossAttention(channels, channels, num_heads),
            }) for _ in range(num_cross_attn)
        ])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        for block in self.convnext_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        B = x.shape[0]
        query = self.style_query.expand(B, -1, -1)

        for ca_block in self.cross_attn_blocks:
            residual = query
            q = ca_block['norm_q'](query)
            kv = ca_block['norm_kv'](x)
            query = ca_block['attn'](q, kv) + residual

        return query.squeeze(1)  # (B, channels)


class DPTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 64, channels: int = 256, intermediate: int = 1024, num_convnext: int = 6, num_self_attn: int = 2, num_heads: int = 2):
        super().__init__()

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

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(text_ids)
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        for block in self.convnext_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        for sa_block in self.self_attn_blocks:
            residual = x
            x = sa_block['norm'](x)
            x = sa_block['attn'](x, mask=text_mask) + residual

        x = x.transpose(1, 2)

        if text_mask is not None:
            x = x * text_mask

        x = self.pool(x).squeeze(-1)
        return x  # (B, channels)


class DurationPredictor(nn.Module):
    def __init__(self, vocab_size: int = 10000, mel_channels: int = 228, text_channels: int = 256, style_channels: int = 256, hidden_dim: int = 256):
        super().__init__()

        self.text_encoder = DPTextEncoder(vocab_size=vocab_size, channels=text_channels)
        self.ref_encoder = DPReferenceEncoder(mel_channels=mel_channels, channels=style_channels)

        input_dim = text_channels + style_channels
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids: torch.Tensor, mel: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        text_emb = self.text_encoder(text_ids, text_mask)  # (B, C_text)
        style_emb = self.ref_encoder(mel)  # (B, C_style)

        combined = torch.cat([text_emb, style_emb], dim=-1)  # (B, C_text + C_style)
        duration = self.estimator(combined).squeeze(-1)  # (B,)

        return torch.relu(duration)  # duration must be positive
