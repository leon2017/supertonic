---
name: train-setup
description: Set up or resume CUDA training for TTS model components
version: 1.0.0
source: supertonic-tts
domain: training
---

# Train Setup

Set up or resume a training experiment for the TTS model.

## When to Use

- User wants to start a new training run
- User wants to modify training config
- User wants to resume from checkpoint
- User asks about training setup or hyperparameters
- User is adding a new language and needs to fine-tune

## Steps

### 1. Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### 2. Verify Dataset Paths

```python
# Check training data structure
data_dir/
├── train/
│   ├── audio/  # WAV files
│   ├── text/   # Transcripts
│   └── metadata.csv  # audio_id, text, speaker_id, duration
├── val/
└── test/
```

### 3. Create Training Config

```yaml
# configs/train_tts.yaml
model:
  duration_predictor:
    hidden_dim: 256
    num_layers: 4
  text_encoder:
    hidden_dim: 512
    num_heads: 8
    num_layers: 6
  vector_estimator:
    hidden_dim: 512
    num_steps: 8  # inference steps
  vocoder:
    pretrained: "path/to/vocoder.pth"
    freeze: true
