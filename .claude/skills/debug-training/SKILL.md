---
name: debug-training
description: Diagnose and fix CUDA training issues including NaN loss, OOM, and convergence failures for TTS
version: 1.0.0
source: supertonic-tts
domain: training
---

# Debug Training

Diagnose and fix TTS training issues.

## When to Use

- Training loss is NaN or exploding
- Model not converging
- GPU OOM errors
- Training speed is unexpectedly slow
- Validation metrics not improving
- Audio quality is poor despite low loss

## Common Issues & Solutions

### Loss is NaN

1. Check learning rate (try 10x smaller)
2. Check gradient clipping is enabled
3. Check for division by zero in custom losses
4. Check input data for NaN/Inf values
5. Disable AMP temporarily to isolate FP16 issues

```python
for name, loss_fn in losses.items():
    val = loss_fn(pred, target, mask)
    if torch.isnan(val):
        print(f"NaN in {name}")
        print(f"  pred range: [{pred.min()}, {pred.max()}]")
        print(f"  target range: [{target.min()}, {target.max()}]")
```

### GPU OOM

1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce model hidden dimensions
4. Use gradient accumulation instead of larger batch
5. Check for memory leaks (tensors not freed)

```python
torch.cuda.memory_summary(device=None, abbreviated=True)
```

### Slow Training

1. Check `num_workers` in DataLoader (should be >= 4)
2. Check `pin_memory=True`
3. Profile with `torch.profiler`
4. Check disk I/O (use SSD, cache preprocessed data)
5. Check if data augmentation is bottleneck

### Not Converging

1. Visualize predictions vs GT at different epochs
2. Check loss weight balance (one loss dominating?)
3. Try training with only flow matching loss first
4. Verify data pipeline produces correct pairs
5. Check if model capacity is sufficient
6. Verify text preprocessing matches inference exactly

### Poor Audio Quality Despite Low Loss

1. Check vocoder quality (try pre-trained vocoder)
2. Verify sample rate matches (44.1kHz)
3. Check for clipping in output
4. Verify style vectors are loaded correctly
5. Check duration prediction accuracy
6. Listen to training data quality

## Diagnostic Commands

```bash
# Monitor GPU
nvidia-smi
watch -n 1 nvidia-smi

# Profile training
python -m torch.profiler.profile train.py --config configs/debug.yaml --profile

# Validate data pipeline
python scripts/validate_data.py --data_dir data/train/ --num_samples 10

# Test single batch
python scripts/test_single_batch.py --config configs/train_tts.yaml
```

## Debug Script

```python
import torch
from train import load_model, load_data

def debug_training():
    model = load_model(config).cuda()
    train_loader = load_data(config)
    
    # Test single batch
    batch = next(iter(train_loader))
    text_ids, audio, duration, style = batch
    
    print(f"Text shape: {text_ids.shape}")
    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {duration}")
    
    # Forward pass
    with torch.cuda.amp.autocast():
        output = model(text_ids, style)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check gradients
    loss = F.mse_loss(output, audio)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 100:
                print(f"Large gradient in {name}: {grad_norm}")
```
