---
name: onnx-debug
description: Debug ONNX Runtime inference issues — model loading failures, precision mismatches, performance bottlenecks
version: 1.0.0
source: supertonic-tts
domain: inference
---

# ONNX Debug

Debug ONNX Runtime inference issues.

## When to Use

- Model fails to load (file not found, incompatible opset)
- Output sounds wrong (distorted, silent, noise)
- Inference is unexpectedly slow
- Numerical precision issues between platforms
- Batch inference produces different results than single

## Common Issues & Solutions

### Model Loading Failure

```python
# Check model files exist
import os
required = ["duration_predictor.onnx", "text_encoder.onnx", 
            "vector_estimator.onnx", "vocoder.onnx", "tts.json", "unicode_indexer.json"]
missing = [f for f in required if not os.path.exists(os.path.join(onnx_dir, f))]
print(f"Missing: {missing}")
```

Fix: Download models from HuggingFace:
```bash
pip install supertonic  # auto-downloads on first use
# Or manual: huggingface-cli download Supertone/supertonic-3
```

### Silent or Noise Output

1. Check duration predictor output is positive and reasonable (0.5-30s)
2. Check latent mask is not all-zeros
3. Verify style vectors are loaded correctly (not all zeros)
4. Check speed parameter (very high speed → very short duration → silence)

```python
wav, dur = text_to_speech("Hello", "en", style, total_step=8, speed=1.05)
print(f"Duration: {dur}")
print(f"Wav range: [{wav.min():.4f}, {wav.max():.4f}]")
print(f"Wav energy: {np.mean(wav**2):.6f}")
```

### Slow Inference

1. VectorEstimator is called `total_step` times — reduce steps (4 minimum)
2. Check provider: CPU is default, GPU needs explicit setup
3. Long text → many chunks → linear slowdown (expected)
4. First inference is slow (model warmup) — subsequent calls are faster

```python
# Profile each model
import time
for name, session in [("DP", dp_ort), ("TextEnc", text_enc_ort), 
                       ("VecEst", vector_est_ort), ("Vocoder", vocoder_ort)]:
    start = time.time()
    # run inference...
    print(f"{name}: {time.time()-start:.3f}s")
```

### Cross-Platform Precision Differences

- Expected: max absolute difference < 1e-4 between platforms
- ONNX Runtime version differences can cause small variations
- Float32 vs Float16: some platforms may use different precision
- Random seed for latent sampling causes non-deterministic output

```python
# For reproducible output (debugging only):
np.random.seed(42)
wav, dur = text_to_speech(text, lang, style, total_step, speed)
```

### Unicode Indexer KeyError

Character not in `unicode_indexer.json`:
```python
text = "Hello 🎉"
for c in text:
    if str(ord(c)) not in indexer:
        print(f"Missing: '{c}' (U+{ord(c):04X})")
```

Fix: Remove unsupported characters in preprocessing (emoji removal should handle most cases).

## Diagnostic Script

```python
import numpy as np
import onnxruntime as ort

def diagnose_tts(onnx_dir, text="Hello world.", lang="en", style_path="assets/voice_styles/M1.json"):
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    tts = load_text_to_speech(onnx_dir)
    style = load_voice_style([style_path])
    
    wav, dur = tts(text, lang, style, total_step=8, speed=1.05)
    
    print(f"Duration: {dur[0]:.2f}s")
    print(f"Wav shape: {wav.shape}")
    print(f"Wav range: [{wav.min():.4f}, {wav.max():.4f}]")
    print(f"Sample rate: {tts.sample_rate}")
    
    if np.isnan(wav).any():
        print("ERROR: NaN in output!")
    if wav.max() == 0:
        print("ERROR: Silent output!")
    if dur[0] <= 0:
        print("ERROR: Zero/negative duration!")
```
