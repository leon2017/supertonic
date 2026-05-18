# Training Pipeline Rules (TTS)

## Overview

This project uses PyTorch + CUDA for training the TTS model. The trained model is then exported to ONNX for inference.

## Training Components

- **Duration Predictor**: Predicts speech duration from text
- **Text Encoder**: Encodes text to embedding space
- **Vector Estimator**: Flow matching denoiser (main model)
- **Vocoder**: Converts latent to waveform (often pre-trained and frozen)

## Training Script Structure

```python
# train.py structure:
# 1. Config loading (Hydra or argparse)
# 2. Dataset & DataLoader setup
# 3. Model, optimizer, scheduler init
# 4. Resume from checkpoint if specified
# 5. Training loop with validation
# 6. Logging (WandB / TensorBoard)
# 7. Checkpoint saving
# 8. ONNX export after training
```

## Distributed Training

- Use PyTorch DDP (DistributedDataParallel)
- Launch via `torchrun --nproc_per_node=N train.py`
- All logging/saving only on rank 0
- Sync batch stats across GPUs

## Mixed Precision

- Use `torch.amp.autocast('cuda')` + `GradScaler`
- Keep loss computation in float32
- Duration/latent predictions in float32 (precision-sensitive)

## Checkpointing

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_metric": best_metric,
    "config": config,
}
```

- Save every N epochs + best model
- Naming: `{experiment_name}_epoch{N}.pth`, `{experiment_name}_best.pth`

## Hyperparameter Discipline

- All hyperparameters in YAML config files under `configs/`
- Log full config to WandB at training start
- Reproducibility: log random seed, git hash, environment info

## Validation

- Validate every N epochs
- Metrics: MOS (if available), duration accuracy, spectrogram quality
- Early stopping on primary metric
- Save audio samples for qualitative evaluation

## Data Loading

- Prefetch with `num_workers >= 4`
- Pin memory for GPU training
- Support multi-language training data
- Text preprocessing must match inference preprocessing exactly

## ONNX Export After Training

After training completes:
1. Load best checkpoint
2. Export each component to ONNX separately
3. Validate ONNX output matches PyTorch output
4. Save to `assets/onnx/` for inference
