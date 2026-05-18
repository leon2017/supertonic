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

training:
  epochs: 200
  batch_size: 16
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: cosine
  warmup_epochs: 5
  grad_clip: 1.0
  amp: true  # mixed precision

data:
  sample_rate: 44100
  num_workers: 4
  languages: ["en", "ko", "ja"]  # multi-lingual training

loss:
  flow_matching: 1.0
  duration: 0.1
  style_consistency: 0.01

checkpoint:
  save_every: 10
  keep_last: 5

wandb:
  project: supertonic-tts
  entity: null
```

### 4. Launch Training

```bash
# Single GPU
python train.py --config configs/train_tts.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py --config configs/train_tts.yaml

# Resume from checkpoint
python train.py --config configs/train_tts.yaml --resume checkpoints/last.pth
```

### 5. Monitor Training

- WandB dashboard for loss curves
- TensorBoard for audio samples
- Validation metrics every N epochs

## Fine-tuning for New Language

When adding Chinese support:

```yaml
# configs/finetune_chinese.yaml
model:
  pretrained: "checkpoints/base_multilingual.pth"
  freeze_encoder: false  # or true for faster convergence

data:
  languages: ["zh"]  # Chinese only
  train_dir: "data/chinese/train"
  val_dir: "data/chinese/val"

training:
  epochs: 50  # fewer epochs for fine-tuning
  lr: 5e-5  # lower learning rate
```

## Common Issues

- **OOM**: Reduce batch_size or enable gradient checkpointing
- **Slow data loading**: Increase num_workers, use SSD
- **NaN loss**: Check learning rate, enable gradient clipping
- **Poor quality**: Verify data preprocessing matches inference
