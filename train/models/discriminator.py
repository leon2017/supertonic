import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels, channels * 2, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels * 2, channels * 4, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels * 4, channels * 8, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels * 8, channels * 8, (5, 1), 1, padding=(2, 0)),
        ])

        self.output = nn.Conv2d(channels * 8, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []

        B, T = x.shape
        pad_len = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad_len))
        x = x.view(B, 1, -1, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.output(x)
        features.append(x)

        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []

        for disc in self.discriminators:
            out, features = disc(x)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, (3, 9), padding=(1, 4)),
            nn.Conv2d(channels, channels * 2, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(channels * 2, channels * 4, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(channels * 4, channels * 4, (3, 3), padding=(1, 1)),
        ])

        self.output = nn.Conv2d(channels * 4, 1, (3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []

        # Compute spectrogram
        spec = torch.stft(
            x, self.n_fft, self.hop_length,
            return_complex=True, window=torch.hann_window(self.n_fft, device=x.device)
        )
        spec = spec.abs().unsqueeze(1)

        for conv in self.convs:
            spec = conv(spec)
            spec = F.leaky_relu(spec, 0.1)
            features.append(spec)

        out = self.output(spec)
        features.append(out)

        return out.flatten(1, -1), features


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions: list[tuple[int, int]] = [(1024, 256), (2048, 512), (4096, 1024)]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ResolutionDiscriminator(n_fft, hop) for n_fft, hop in resolutions
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []

        for disc in self.discriminators:
            out, features = disc(x)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features
