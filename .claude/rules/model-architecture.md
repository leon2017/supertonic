# Model Architecture Rules (TTS)

## Architecture Overview

The TTS model consists of 4 components trained separately or jointly:

```
Text → DurationPredictor → duration
    → TextEncoder → text_emb
    → VectorEstimator (flow matching) → latent
    → Vocoder → waveform
```

## DurationPredictor

- Input: text embeddings + style_dp
- Output: predicted duration (seconds)
- Architecture: Transformer encoder or simple MLP
- Loss: MSE or MAE against ground truth duration

## TextEncoder

- Input: text token IDs + style_ttl
- Output: text embeddings (B, D, L)
- Architecture: Transformer encoder with positional encoding
- Supports 31+ languages via shared embedding space

## VectorEstimator (Flow Matching)

- Input: noisy latent + text_emb + style_ttl + timestep
- Output: denoised latent
- Architecture: U-Net or Transformer with cross-attention
- Training: flow matching objective (ODE-based, not diffusion)
- Inference: iterative denoising (8-16 steps)

## Vocoder

- Input: latent (B, D, T)
- Output: waveform (B, T_audio) at 44.1kHz
- Architecture: HiFi-GAN, BigVGAN, or similar
- Often pre-trained and frozen during TTS training

## Style Vectors

- `style_ttl`: controls timbre/prosody (learned per speaker)
- `style_dp`: controls rhythm/pace (learned per speaker)
- Extracted from reference audio or learned embeddings

## Model Variants

| Variant | Params | Use Case |
|---------|--------|----------|
| Base | 99M | Production (current) |
| Large | 200M+ | High quality research |
| Tiny | 30M | Mobile/edge |

## Constraints

- All models must support variable-length input (use masks)
- Batch norm only in encoder; layer norm in decoder
- Activation: GELU or SiLU (not ReLU)
- Weight init: Xavier for linear, orthogonal for conv

## Adding New Components

When adding modules (e.g., prosody predictor, emotion control):
1. Create separate file in `models/`
2. Make it optional via config flag
3. Ensure ONNX exportable
4. Add unit test verifying output shape
