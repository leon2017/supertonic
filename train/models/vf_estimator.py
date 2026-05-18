import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .convnext import ConvNeXtBlock


class TimeEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.unsqueeze(-1))


class ConditioningBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.scale = nn.Linear(channels, channels)
        self.shift = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, C)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        scale = self.scale(cond).unsqueeze(1)  # (B, 1, C)
        shift = self.shift(cond).unsqueeze(1)  # (B, 1, C)
        x = x * (1 + scale) + shift
        return x.transpose(1, 2)  # (B, C, T)


class VFEstimatorBlock(nn.Module):
    def __init__(self, channels: int, intermediate: int, kernel_size: int = 7):
        super().__init__()

        self.dilated_convnext = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size, dilation=d)
            for d in [1, 2, 4, 8]
        ])

        self.standard_convnext = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size)
            for _ in range(2)
        ])

        self.time_cond = ConditioningBlock(channels)
        self.text_cond = ConditioningBlock(channels)
        self.ref_cond = ConditioningBlock(channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor, ref_emb: torch.Tensor) -> torch.Tensor:
        for block in self.dilated_convnext:
            x = block(x)

        for block in self.standard_convnext:
            x = block(x)

        # Apply conditioning
        x = self.time_cond(x, time_emb)
        x = self.text_cond(x, text_emb.mean(dim=2))  # pool text over time
        x = self.ref_cond(x, ref_emb.mean(dim=2))    # pool ref over time

        return x


class VFEstimator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 144,
        channels: int = 256,
        intermediate: int = 1024,
        num_blocks: int = 4,
        kernel_size: int = 7,
        text_channels: int = 512,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.input_proj = nn.Conv1d(latent_dim, channels, 1)

        self.time_embed = TimeEmbedding(channels)

        self.text_proj = nn.Conv1d(text_channels, channels, 1)
        self.ref_proj = nn.Conv1d(text_channels, channels, 1)

        self.blocks = nn.ModuleList([
            VFEstimatorBlock(channels, intermediate, kernel_size)
            for _ in range(num_blocks)
        ])

        self.final_blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size)
            for _ in range(4)
        ])

        self.output_proj = nn.Conv1d(channels, latent_dim, 1)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        ref_emb: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_proj(noisy_latent)

        time_emb = self.time_embed(t)
        text_proj = self.text_proj(text_emb)
        ref_proj = self.ref_proj(ref_emb)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, time_emb, text_proj, ref_proj, use_reentrant=False)
            else:
                x = block(x, time_emb, text_proj, ref_proj)

        for block in self.final_blocks:
            x = block(x)

        output = self.output_proj(x)

        if latent_mask is not None:
            output = output * latent_mask

        return output
