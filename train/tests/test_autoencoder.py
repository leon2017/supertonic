import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import LatentEncoder, LatentDecoder, SpeechAutoencoder
from models.discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator


def test_encoder_forward():
    encoder = LatentEncoder(mel_channels=228, latent_dim=24, channels=256, intermediate=1024, num_layers=3)
    mel = torch.randn(2, 228, 50)
    latent = encoder(mel)
    assert latent.shape == (2, 24, 50)
    assert not torch.isnan(latent).any()


def test_decoder_forward():
    decoder = LatentDecoder(latent_dim=24, channels=256, intermediate=1024, num_layers=3, hop_length=512)
    latent = torch.randn(2, 24, 50)
    wav = decoder(latent)
    assert wav.shape[0] == 2
    assert wav.shape[1] > 50  # upsampled
    assert wav.abs().max() <= 1.0  # tanh output


def test_autoencoder_forward():
    model = SpeechAutoencoder(
        mel_channels=228, latent_dim=24, channels=256,
        intermediate=1024, num_layers=3, hop_length=512
    )
    mel = torch.randn(2, 228, 50)
    wav, latent = model(mel)
    assert latent.shape == (2, 24, 50)
    assert wav.shape[0] == 2


def test_mpd_forward():
    mpd = MultiPeriodDiscriminator()
    x = torch.randn(2, 16000)
    outputs, features = mpd(x)
    assert len(outputs) == 5
    assert len(features) == 5


def test_mrd_forward():
    mrd = MultiResolutionDiscriminator()
    x = torch.randn(2, 16000)
    outputs, features = mrd(x)
    assert len(outputs) == 3
    assert len(features) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
