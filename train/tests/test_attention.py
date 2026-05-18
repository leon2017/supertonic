import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attention import MultiHeadSelfAttention, CrossAttention


def test_self_attention_forward():
    attn = MultiHeadSelfAttention(channels=256, num_heads=4)
    x = torch.randn(4, 50, 256)
    y = attn(x)
    assert y.shape == (4, 50, 256)
    assert not torch.isnan(y).any()


def test_self_attention_with_mask():
    attn = MultiHeadSelfAttention(channels=128, num_heads=4)
    x = torch.randn(2, 50, 128)
    mask = torch.ones(2, 1, 50)
    mask[:, :, 25:] = 0
    y = attn(x, mask)
    assert y.shape == x.shape


def test_cross_attention_forward():
    attn = CrossAttention(query_dim=512, kv_dim=256, num_heads=4)
    q = torch.randn(4, 100, 512)
    kv = torch.randn(4, 50, 256)
    y = attn(q, kv)
    assert y.shape == (4, 100, 512)
    assert not torch.isnan(y).any()


def test_cross_attention_gradient():
    attn = CrossAttention(query_dim=64, kv_dim=32, num_heads=2)
    q = torch.randn(2, 20, 64, requires_grad=True)
    kv = torch.randn(2, 10, 32, requires_grad=True)
    y = attn(q, kv)
    loss = y.sum()
    loss.backward()
    assert q.grad is not None
    assert kv.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
