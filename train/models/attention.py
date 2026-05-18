import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert query_dim % num_heads == 0

        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(B, T_kv, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(B, T_kv, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T_q, self.query_dim)
        out = self.out_proj(out)

        return out
