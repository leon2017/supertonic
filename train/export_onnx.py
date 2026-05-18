import argparse
import torch
import numpy as np
import yaml
from pathlib import Path

from models.autoencoder import SpeechAutoencoder
from models.text_encoder import TextEncoder
from models.reference_encoder import ReferenceEncoder
from models.vf_estimator import VFEstimator
from models.duration_predictor import DurationPredictor


def export_vocoder(model: SpeechAutoencoder, output_path: str):
    model.eval()
    decoder = model.decoder

    latent = torch.randn(1, 24, 100)

    torch.onnx.export(
        decoder,
        latent,
        output_path,
        input_names=['latent'],
        output_names=['wav'],
        dynamic_axes={
            'latent': {0: 'batch', 2: 'latent_len'},
            'wav': {0: 'batch', 1: 'audio_len'}
        },
        opset_version=17
    )
    print(f"Vocoder 已导出: {output_path}")


def export_text_encoder(text_enc: TextEncoder, ref_enc: ReferenceEncoder, output_path: str):
    """Export text encoder + reference encoder as a combined module."""

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_enc, ref_enc):
            super().__init__()
            self.text_enc = text_enc
            self.ref_enc = ref_enc

        def forward(self, text_ids, style_ttl, text_mask):
            # style_ttl is pre-computed reference embedding
            text_emb = self.text_enc(text_ids, style_ttl, text_mask)
            return text_emb

    wrapper = TextEncoderWrapper(text_enc, ref_enc)
    wrapper.eval()

    text_ids = torch.randint(0, 1000, (1, 50), dtype=torch.long)
    style_ttl = torch.randn(1, 512, 50)
    text_mask = torch.ones(1, 1, 50)

    torch.onnx.export(
        wrapper,
        (text_ids, style_ttl, text_mask),
        output_path,
        input_names=['text_ids', 'style_ttl', 'text_mask'],
        output_names=['text_emb'],
        dynamic_axes={
            'text_ids': {0: 'batch', 1: 'seq_len'},
            'style_ttl': {0: 'batch', 2: 'style_len'},
            'text_mask': {0: 'batch', 2: 'seq_len'},
            'text_emb': {0: 'batch', 2: 'seq_len'}
        },
        opset_version=17
    )
    print(f"Text Encoder 已导出: {output_path}")


def export_vf_estimator(vf_est: VFEstimator, output_path: str, latent_dim: int = 144):
    vf_est.eval()

    noisy_latent = torch.randn(1, latent_dim, 50)
    text_emb = torch.randn(1, 512, 100)
    ref_emb = torch.randn(1, 512, 50)
    t = torch.tensor([0.5])
    latent_mask = torch.ones(1, 1, 50)

    torch.onnx.export(
        vf_est,
        (noisy_latent, text_emb, ref_emb, t, latent_mask),
        output_path,
        input_names=['noisy_latent', 'text_emb', 'style_ttl', 'timestep', 'latent_mask'],
        output_names=['xt'],
        dynamic_axes={
            'noisy_latent': {0: 'batch', 2: 'latent_len'},
            'text_emb': {0: 'batch', 2: 'text_len'},
            'style_ttl': {0: 'batch', 2: 'style_len'},
            'latent_mask': {0: 'batch', 2: 'latent_len'},
            'xt': {0: 'batch', 2: 'latent_len'}
        },
        opset_version=17
    )
    print(f"VF Estimator 已导出: {output_path}")


def export_duration_predictor(dp: DurationPredictor, output_path: str):
    dp.eval()

    text_ids = torch.randint(0, 1000, (1, 50), dtype=torch.long)
    mel = torch.randn(1, 228, 100)
    text_mask = torch.ones(1, 1, 50)

    torch.onnx.export(
        dp,
        (text_ids, mel, text_mask),
        output_path,
        input_names=['text_ids', 'style_dp', 'text_mask'],
        output_names=['duration'],
        dynamic_axes={
            'text_ids': {0: 'batch', 1: 'seq_len'},
            'style_dp': {0: 'batch', 2: 'mel_len'},
            'text_mask': {0: 'batch', 2: 'seq_len'},
            'duration': {0: 'batch'}
        },
        opset_version=17
    )
    print(f"Duration Predictor 已导出: {output_path}")


def export_style_vectors(ref_enc: ReferenceEncoder, mel_path: str, output_path: str):
    """Extract and save style vectors from reference audio."""
    ref_enc.eval()

    mel = torch.FloatTensor(np.load(mel_path)).unsqueeze(0)

    with torch.no_grad():
        style_ttl = ref_enc(mel)  # (1, C, num_tokens)

    import json
    style_data = {
        'style_ttl': {
            'dims': list(style_ttl.shape),
            'data': style_ttl.flatten().tolist()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(style_data, f)

    print(f"Style vectors 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('--autoencoder_checkpoint', type=str, help='Autoencoder checkpoint')
    parser.add_argument('--ttl_checkpoint', type=str, help='Text-to-Latent checkpoint')
    parser.add_argument('--duration_checkpoint', type=str, help='Duration Predictor checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--config', type=str, default='train/configs/text_to_latent.yaml', help='模型配置')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')  # Export on CPU for compatibility

    # Export Vocoder (from autoencoder)
    if args.autoencoder_checkpoint:
        print("\n--- 导出 Vocoder ---")
        autoencoder = SpeechAutoencoder(
            mel_channels=config['data']['n_mels'],
            latent_dim=config['model']['latent_dim']
        )
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device, weights_only=False)
        autoencoder.load_state_dict(ckpt['model_state_dict'])
        export_vocoder(autoencoder, str(output_dir / 'vocoder.onnx'))

    # Export Text Encoder + VF Estimator
    if args.ttl_checkpoint:
        print("\n--- 导出 Text Encoder ---")
        ckpt = torch.load(args.ttl_checkpoint, map_location=device, weights_only=False)

        text_enc = TextEncoder(
            vocab_size=config['model']['vocab_size'],
            embedding_dim=config['model']['char_embedding_dim'],
            channels=config['model']['text_channels'],
            num_heads=config['model']['text_num_heads'],
        )
        text_enc.load_state_dict(ckpt['text_encoder_state_dict'])

        ref_enc = ReferenceEncoder(
            mel_channels=config['data']['n_mels'],
            channels=config['model']['text_channels']
        )
        ref_enc.load_state_dict(ckpt['ref_encoder_state_dict'])

        export_text_encoder(text_enc, ref_enc, str(output_dir / 'text_encoder.onnx'))

        print("\n--- 导出 VF Estimator ---")
        Kc = config['model']['temporal_compression']
        compressed_dim = config['model']['latent_dim'] * Kc

        vf_est = VFEstimator(
            latent_dim=compressed_dim,
            channels=config['model']['vf_channels'],
            intermediate=config['model']['vf_intermediate'],
            num_blocks=config['model']['vf_num_blocks'],
            text_channels=config['model']['text_channels'],
        )
        vf_est.load_state_dict(ckpt['vf_estimator_state_dict'])

        export_vf_estimator(vf_est, str(output_dir / 'vector_estimator.onnx'), compressed_dim)

    # Export Duration Predictor
    if args.duration_checkpoint:
        print("\n--- 导出 Duration Predictor ---")
        dp = DurationPredictor(
            vocab_size=config['model']['vocab_size'],
            mel_channels=config['data']['n_mels']
        )
        ckpt = torch.load(args.duration_checkpoint, map_location=device, weights_only=False)
        dp.load_state_dict(ckpt['model_state_dict'])
        export_duration_predictor(dp, str(output_dir / 'duration_predictor.onnx'))

    print("\n导出完成！")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
