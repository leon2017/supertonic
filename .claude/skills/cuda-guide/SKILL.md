---
name: cuda-guide
description: CUDA/GPU programming patterns and best practices for TTS training with PyTorch
version: 1.0.0
source: supertonic-tts
domain: training
---

# CUDA Guide for TTS Training

CUDA/GPU best practices for training TTS models with PyTorch.

## When to Use

- User is setting up GPU training environment
- User encounters CUDA errors or OOM issues
- User asks about GPU optimization for TTS training
- User wants to use multi-GPU training

## PyTorch CUDA Basics

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Move data to GPU
text_ids = text_ids.to(device)
audio = audio.to(device)
```

## Memory Management

### Avoid OOM

```python
# 1. Clear cache periodically
torch.cuda.empty_cache()

# 2. Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.layer, x)

# 3. Use gradient accumulation for large batch sizes
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Monitor Memory Usage

```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Multi-GPU Training (DDP)

```python
# train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)
    
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Training loop...
    
    dist.destroy_process_group()

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

## Performance Optimization

### DataLoader Settings

```python
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # CPU workers for data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
)
```

### Efficient Operations

```python
# Use in-place operations when safe
x.add_(y)  # instead of x = x + y

# Avoid CPU-GPU transfers in training loop
# BAD: loss.item() every iteration
# GOOD: accumulate losses, log every N iterations

# Use torch.no_grad() for validation
with torch.no_grad():
    val_output = model(val_batch)
```

## Common CUDA Errors

### RuntimeError: CUDA out of memory

Solutions:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Clear cache: `torch.cuda.empty_cache()`

### RuntimeError: CUDA error: device-side assert triggered

Usually caused by:
- Invalid index in embedding layer
- NaN in loss computation
- Mismatched tensor dimensions

Debug with:
```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

### RuntimeError: all tensors must be on the same device

Ensure all model inputs are on the same device:
```python
text_ids = text_ids.to(device)
audio = audio.to(device)
style = style.to(device)
```

## Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices for TTS Training

1. **Use mixed precision** — 2x speedup with minimal quality loss
2. **Pin memory** — faster data transfer to GPU
3. **Gradient checkpointing** — trade compute for memory
4. **Profile first** — identify bottlenecks before optimizing
5. **Monitor GPU utilization** — aim for >80% with `nvidia-smi`
