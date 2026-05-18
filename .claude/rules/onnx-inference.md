# ONNX Inference Rules

## Model Files

4 ONNX models in `assets/onnx/`:
- `duration_predictor.onnx` — predicts speech duration
- `text_encoder.onnx` — encodes text to embedding
- `vector_estimator.onnx` — flow matching denoiser (called N times)
- `vocoder.onnx` — latent to waveform

Config: `assets/onnx/tts.json`
Unicode map: `assets/onnx/unicode_indexer.json`

## Session Configuration

```python
opts = ort.SessionOptions()
# For CPU inference (default):
providers = ["CPUExecutionProvider"]

# For GPU inference (experimental):
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

- Default to CPU — GPU mode is not fully tested
- Use single SessionOptions instance shared across all models
- Load all 4 models at initialization, not per-inference

## Input/Output Contracts

### DurationPredictor
- Input: `text_ids` (B, L) int64, `style_dp` (B, D1, D2) float32, `text_mask` (B, 1, L) float32
- Output: `duration` (B,) float32 — seconds

### TextEncoder
- Input: `text_ids` (B, L) int64, `style_ttl` (B, D1, D2) float32, `text_mask` (B, 1, L) float32
- Output: `text_emb` (B, D, L) float32

### VectorEstimator
- Input: `noisy_latent` (B, D, T) float32, `text_emb`, `style_ttl`, `text_mask`, `latent_mask` (B, 1, T), `current_step` (B,), `total_step` (B,)
- Output: `xt` (B, D, T) float32

### Vocoder
- Input: `latent` (B, D, T) float32
- Output: `wav` (B, T_audio) float32

## Performance Guidelines

- VectorEstimator is called `total_step` times (default 8) — it dominates inference time
- For batch inference, pad all inputs to max length and use masks
- Latent dimensions derived from config: `ldim * chunk_compress_factor`
- Never load models inside inference loop

## Numerical Safety

- Random latent sampling uses `np.random.randn` — seed for reproducibility if needed
- Apply `latent_mask` to zero out padding positions in noisy latent
- Duration division by speed must not produce zero or negative values
- WAV output clipping: values should be in [-1, 1] range
