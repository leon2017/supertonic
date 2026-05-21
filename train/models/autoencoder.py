import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .convnext import ConvNeXtBlock


class LatentEncoder(nn.Module):
    def __init__(
        self,
        mel_channels: int = 228,
        latent_dim: int = 24,
        channels: int = 512,
        intermediate: int = 2048,
        num_layers: int = 10,
        kernel_size: int = 7,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.input_proj = nn.Conv1d(mel_channels, channels, 1)

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Conv1d(channels, latent_dim, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(mel)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        latent = self.output_proj(x)
        return latent


class LatentDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 24,
        channels: int = 512,
        intermediate: int = 2048,
        num_layers: int = 10,
        kernel_size: int = 7,
        hop_length: int = 512,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.hop_length = hop_length

        dilations = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

        self.input_proj = nn.Conv1d(latent_dim, channels, 1)

        # Progressive upsampling: 512 = 8 × 8 × 8
        upsample_rates = [8, 8, 8]
        self.upsamples = nn.ModuleList()
        for rate in upsample_rates:
            self.upsamples.append(nn.ConvTranspose1d(
                channels, channels,
                kernel_size=rate * 2, stride=rate, padding=rate // 2
            ))
        self.upsample_act = nn.GELU()

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(channels, intermediate, kernel_size, dilation=dilations[i])
            for i in range(num_layers)
        ])

        self.output_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(latent)

        for upsample in self.upsamples:
            x = self.upsample_act(upsample(x))

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        wav = self.output_proj(x).squeeze(1)
        return torch.tanh(wav)


class SpeechAutoencoder(nn.Module):
    def __init__(
        self,
        mel_channels: int = 228,
        latent_dim: int = 24,
        channels: int = 512,
        intermediate: int = 2048,
        num_layers: int = 10,
        kernel_size: int = 7,
        hop_length: int = 512,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.encoder = LatentEncoder(
            mel_channels, latent_dim, channels, intermediate,
            num_layers, kernel_size, use_checkpoint
        )
        self.decoder = LatentDecoder(
            latent_dim, channels, intermediate, num_layers,
            kernel_size, hop_length, use_checkpoint
        )

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(mel)
        wav = self.decoder(latent)
        return wav, latent

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
