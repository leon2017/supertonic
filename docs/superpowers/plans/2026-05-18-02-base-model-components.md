# 基础模型组件实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 ConvNeXt、Self/Cross-Attention 等共享模型组件，为后续训练提供基础模块

**Architecture:** ConvNeXt Block（带 dilation 支持）+ Multi-Head Self-Attention + Cross-Attention + 辅助工具（LayerNorm, activation 等）

**Tech Stack:** PyTorch 2.2+, torch.nn

---

## File Structure

```
train/
├── models/
│   ├── __init__.py
│   ├── convnext.py          # ConvNeXt block
│   ├── attention.py         # Self/Cross-Attention
│   └── utils.py             # 辅助工具
└── tests/
    ├── __init__.py
    ├── test_convnext.py
    └── test_attention.py
```

---

### Task 1: 实现 ConvNeXt Block

**Files:**
- Create: `train/models/__init__.py`
- Create: `train/models/convnext.py`
- Create: `train/tests/__init__.py`
- Create: `train/tests/test_convnext.py`

- [ ] **Step 1: 创建 models/__init__.py**

```python
"""Model components for Chinese TTS training."""
```

- [ ] **Step 2: 编写 ConvNeXt Block 实现**

```python
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with optional dilation.
    
    Architecture:
        DepthwiseConv → LayerNorm → Linear (expand) → GELU → Linear (project) → Residual
    
    Args:
        channels: Number of input/output channels
        intermediate_channels: Intermediate expansion channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate for depthwise conv
    """
    
    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        kernel_size: int = 7,
        dilation: int = 1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(channels)
        
        # Pointwise expansion
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (B, C, T)
            
        Returns:
            Output tensor, shape (B, C, T)
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)  # (B, C, T)
        
        # Transpose for LayerNorm: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # LayerNorm + Pointwise
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Transpose back: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # Residual connection
        x = x + residual
        
        return x


if __name__ == '__main__':
    # Quick test
    block = ConvNeXtBlock(channels=512, intermediate_channels=2048, kernel_size=7, dilation=1)
    x = torch.randn(2, 512, 100)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch"
    print("ConvNeXt block test passed!")
```

- [ ] **Step 3: 编写单元测试**

```python
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.convnext import ConvNeXtBlock


def test_convnext_forward():
    """Test ConvNeXt forward pass."""
    batch_size = 4
    channels = 256
    seq_len = 50
    
    block = ConvNeXtBlock(
        channels=channels,
        intermediate_channels=1024,
        kernel_size=7,
        dilation=1
    )
    
    x = torch.randn(batch_size, channels, seq_len)
    y = block(x)
    
    assert y.shape == (batch_size, channels, seq_len)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_convnext_dilation():
    """Test ConvNeXt with different dilations."""
    block_d1 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=1)
    block_d2 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=2)
    block_d4 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=4)
    
    x = torch.randn(2, 128, 100)
    
    y1 = block_d1(x)
    y2 = block_d2(x)
    y4 = block_d4(x)
    
    assert y1.shape == y2.shape == y4.shape == x.shape


def test_convnext_gradient():
    """Test ConvNeXt gradient flow."""
    block = ConvNeXtBlock(channels=64, intermediate_channels=256)
    x = torch.randn(2, 64, 50, requires_grad=True)
    
    y = block(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 4: 运行测试**

运行：`pytest train/tests/test_convnext.py -v`

预期输出：
```
test_convnext_forward PASSED
test_convnext_dilation PASSED
test_convnext_gradient PASSED
```

- [ ] **Step 5: 提交 ConvNeXt 实现**

```bash
git add train/models/__init__.py train/models/convnext.py train/tests/__init__.py train/tests/test_convnext.py
git commit -m "feat(models): implement ConvNeXt block with dilation support"
```

---

### Task 2: 实现 Attention 模块

**Files:**
- Create: `train/models/attention.py`
- Create: `train/tests/test_attention.py`

- [ ] **Step 1: 编写 Self-Attention 和 Cross-Attention**

```python
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (B, T, C)
            mask: Optional attention mask, shape (B, 1, T) or (B, T, T)
            
        Returns:
            Output tensor, shape (B, T, C)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 3:  # (B, 1, T)
                mask = mask.unsqueeze(1)  # (B, 1, 1, T)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        
        # Output projection
        out = self.proj(out)
        
        return out


class CrossAttention(nn.Module):
    """Cross-attention between query and key-value pairs.
    
    Args:
        query_dim: Query dimension
        kv_dim: Key-value dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
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
        """Forward pass.
        
        Args:
            query: Query tensor, shape (B, T_q, C_q)
            key_value: Key-value tensor, shape (B, T_kv, C_kv)
            mask: Optional mask, shape (B, 1, T_kv)
            
        Returns:
            Output tensor, shape (B, T_q, C_q)
        """
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(B, T_kv, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(B, T_kv, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_kv)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, 1, T_kv)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ v  # (B, H, T_q, D)
        out = out.transpose(1, 2).reshape(B, T_q, self.query_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


if __name__ == '__main__':
    # Quick test
    self_attn = MultiHeadSelfAttention(channels=512, num_heads=4)
    x = torch.randn(2, 100, 512)
    y = self_attn(x)
    print(f"Self-Attention: {x.shape} -> {y.shape}")
    
    cross_attn = CrossAttention(query_dim=512, kv_dim=256, num_heads=4)
    q = torch.randn(2, 100, 512)
    kv = torch.randn(2, 50, 256)
    y = cross_attn(q, kv)
    print(f"Cross-Attention: query {q.shape}, kv {kv.shape} -> {y.shape}")
    
    print("Attention modules test passed!")
```

- [ ] **Step 2: 编写 Attention 单元测试**

```python
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attention import MultiHeadSelfAttention, CrossAttention


def test_self_attention_forward():
    """Test self-attention forward pass."""
    batch_size = 4
    seq_len = 50
    channels = 256
    
    attn = MultiHeadSelfAttention(channels=channels, num_heads=4)
    x = torch.randn(batch_size, seq_len, channels)
    y = attn(x)
    
    assert y.shape == (batch_size, seq_len, channels)
    assert not torch.isnan(y).any()


def test_self_attention_with_mask():
    """Test self-attention with mask."""
    attn = MultiHeadSelfAttention(channels=128, num_heads=4)
    x = torch.randn(2, 50, 128)
    mask = torch.ones(2, 1, 50)
    mask[:, :, 25:] = 0  # Mask out second half
    
    y = attn(x, mask)
    assert y.shape == x.shape


def test_cross_attention_forward():
    """Test cross-attention forward pass."""
    batch_size = 4
    query_len = 100
    kv_len = 50
    query_dim = 512
    kv_dim = 256
    
    attn = CrossAttention(query_dim=query_dim, kv_dim=kv_dim, num_heads=4)
    q = torch.randn(batch_size, query_len, query_dim)
    kv = torch.randn(batch_size, kv_len, kv_dim)
    
    y = attn(q, kv)
    
    assert y.shape == (batch_size, query_len, query_dim)
    assert not torch.isnan(y).any()


def test_cross_attention_gradient():
    """Test cross-attention gradient flow."""
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
```

- [ ] **Step 3: 运行测试**

运行：`pytest train/tests/test_attention.py -v`

预期输出：
```
test_self_attention_forward PASSED
test_self_attention_with_mask PASSED
test_cross_attention_forward PASSED
test_cross_attention_gradient PASSED
```

- [ ] **Step 4: 提交 Attention 实现**

```bash
git add train/models/attention.py train/tests/test_attention.py
git commit -m "feat(models): implement multi-head self/cross attention"
```

---

### Task 3: 实现辅助工具模块

**Files:**
- Create: `train/models/utils.py`

- [ ] **Step 1: 编写辅助工具函数**

```python
import torch
import torch.nn as nn


def create_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Create mask from sequence lengths.
    
    Args:
        lengths: Sequence lengths, shape (B,)
        max_len: Maximum length (default: max of lengths)
        
    Returns:
        Mask tensor, shape (B, 1, max_len), 1 for valid positions
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).float()  # (B, 1, T)


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Generate sequence mask.
    
    Args:
        length: Lengths tensor, shape (B,)
        max_length: Maximum length
        
    Returns:
        Boolean mask, shape (B, max_length)
    """
    if max_length is None:
        max_length = length.max()
    
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.
    
    Args:
        channels: Number of channels
        max_len: Maximum sequence length
    """
    
    def __init__(self, channels: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: Input tensor, shape (B, T, C)
            
        Returns:
            Output with positional encoding, shape (B, T, C)
        """
        return x + self.pe[:, :x.size(1)]


import math


if __name__ == '__main__':
    # Test mask creation
    lengths = torch.tensor([10, 15, 8])
    mask = create_mask_from_lengths(lengths, max_len=20)
    print(f"Mask shape: {mask.shape}")
    assert mask.shape == (3, 1, 20)
    
    # Test positional encoding
    pe = PositionalEncoding(channels=128, max_len=1000)
    x = torch.randn(2, 50, 128)
    y = pe(x)
    print(f"Positional encoding: {x.shape} -> {y.shape}")
    assert y.shape == x.shape
    
    print("Utils test passed!")
```

- [ ] **Step 2: 提交辅助工具**

```bash
git add train/models/utils.py
git commit -m "feat(models): add utility functions for masking and positional encoding"
```

---

### Task 4: 更新 models/__init__.py 导出

**Files:**
- Modify: `train/models/__init__.py`

- [ ] **Step 1: 更新 __init__.py**

```python
"""Model components for Chinese TTS training."""

from .convnext import ConvNeXtBlock
from .attention import MultiHeadSelfAttention, CrossAttention
from .utils import create_mask_from_lengths, sequence_mask, PositionalEncoding

__all__ = [
    'ConvNeXtBlock',
    'MultiHeadSelfAttention',
    'CrossAttention',
    'create_mask_from_lengths',
    'sequence_mask',
    'PositionalEncoding',
]
```

- [ ] **Step 2: 提交更新**

```bash
git add train/models/__init__.py
git commit -m "feat(models): export all base components"
```

---

### Task 5: 运行完整测试套件

**Files:**
- None (testing only)

- [ ] **Step 1: 运行所有测试**

运行：`pytest train/tests/ -v`

预期输出：
```
train/tests/test_convnext.py::test_convnext_forward PASSED
train/tests/test_convnext.py::test_convnext_dilation PASSED
train/tests/test_convnext.py::test_convnext_gradient PASSED
train/tests/test_attention.py::test_self_attention_forward PASSED
train/tests/test_attention.py::test_self_attention_with_mask PASSED
train/tests/test_attention.py::test_cross_attention_forward PASSED
train/tests/test_attention.py::test_cross_attention_gradient PASSED

======== 7 passed in X.XXs ========
```

- [ ] **Step 2: 验证导入**

运行：`python -c "from train.models import *; print('All imports successful!')"`

预期输出：`All imports successful!`

---

## 验证清单

完成所有任务后，验证：

- [ ] ConvNeXt block 支持不同 dilation
- [ ] Self-attention 支持 mask
- [ ] Cross-attention 正确处理不同维度的 query 和 key-value
- [ ] 所有模块梯度流正常
- [ ] 辅助工具函数正确
- [ ] 所有测试通过
- [ ] 模块可以正确导入

---

## 注意事项

1. **数值稳定性**：Attention 使用 scaled dot-product，避免梯度爆炸
2. **内存效率**：ConvNeXt 使用 depthwise convolution 减少参数量
3. **可扩展性**：所有模块支持可变序列长度和 batch size
4. **测试覆盖**：每个模块都有单元测试验证正确性
