import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionMelLoss(nn.Module):
    def __init__(self, fft_sizes: list[int] = [1024, 2048, 4096], hop_length: int = 512, n_mels: int = 228, sample_rate: int = 44100):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

    def _mel_spectrogram(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(x, n_fft, self.hop_length, window=window, return_complex=True)
        return spec.abs()

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred_wav.device)

        for n_fft in self.fft_sizes:
            pred_spec = self._mel_spectrogram(pred_wav, n_fft)
            target_spec = self._mel_spectrogram(target_wav, n_fft)
            loss = loss + F.l1_loss(pred_spec, target_spec)

        return loss / len(self.fft_sizes)
