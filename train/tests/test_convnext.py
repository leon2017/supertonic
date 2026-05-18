import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.convnext import ConvNeXtBlock


def test_convnext_forward():
    block = ConvNeXtBlock(channels=256, intermediate_channels=1024, kernel_size=7, dilation=1)
    x = torch.randn(4, 256, 50)
    y = block(x)
    assert y.shape == (4, 256, 50)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_convnext_dilation():
    block_d1 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=1)
    block_d2 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=2)
    block_d4 = ConvNeXtBlock(channels=128, intermediate_channels=512, dilation=4)

    x = torch.randn(2, 128, 100)
    y1 = block_d1(x)
    y2 = block_d2(x)
    y4 = block_d4(x)
    assert y1.shape == y2.shape == y4.shape == x.shape


def test_convnext_gradient():
    block = ConvNeXtBlock(channels=64, intermediate_channels=256)
    x = torch.randn(2, 64, 50, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
