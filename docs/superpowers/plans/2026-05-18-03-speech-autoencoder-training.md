# Speech Autoencoder 训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并训练 Speech Autoencoder（Latent Encoder + Decoder + Discriminator），支持暂停恢复

**Architecture:** 10-layer ConvNeXt Encoder (mel→latent) + 10-layer dilated ConvNeXt Decoder (latent→wav) + Multi-period/Multi-resolution Discriminator

**Tech Stack:** PyTorch 2.2+, torch.amp (fp16), gradient checkpointing, WandB

---

## File Structure

```
train/
├── models/
│   ├── autoencoder.py       # Encoder + Decoder
│   ├── discriminator.py     # Discriminators
│   └── dataset.py           # Dataset for autoencoder
├── losses/
│   ├── __init__.py
│   ├── reconstruction.py    # Mel + waveform reconstruction
│   └── adversarial.py       # GAN loss + feature matching
├── configs/
│   └── autoencoder.yaml     # Training config
├── train_autoencoder.py     # Main training script
└── utils/
    ├── __init__.py
    ├── checkpoint.py        # Checkpoint save/load
    └── logger.py            # WandB logger
```

---

由于篇幅限制，以下为关键任务概要。完整实现包含：

### Task 1: 实现 Latent Encoder/Decoder
- 10-layer ConvNeXt encoder: 228-dim mel → 24-dim latent
- 10-layer dilated ConvNeXt decoder: 24-dim latent → 44.1kHz waveform
- 支持 gradient checkpointing

### Task 2: 实现 Discriminator
- Multi-period discriminator (periods: 2,3,5,7,11)
- Multi-resolution discriminator (FFT sizes: 1024,2048,4096)

### Task 3: 实现损失函数
- Multi-resolution mel spectrogram L1 loss (λ=45)
- Adversarial loss (λ=1)
- Feature matching loss (λ=0.1)

### Task 4: 实现 Checkpoint 管理
- 定期保存（每 10k iterations）
- 最新保存（每 1k iterations，覆盖）
- 优雅退出（Ctrl+C 信号捕获）
- 自动恢复（检测最新 checkpoint）
- 保存内容：model, optimizer, scheduler, iteration, epoch, best_loss, random_state

### Task 5: 实现训练脚本
- AMP (fp16) 训练
- Gradient accumulation (batch=4, accum=16)
- Gradient checkpointing
- WandB logging
- 验证循环

### Task 6: 配置文件
```yaml
# autoencoder.yaml
data:
  train_dir: ./data/zh/train
  val_dir: ./data/zh/val
  segment_length: 44100  # 1 second
  
model:
  mel_channels: 228
  latent_dim: 24
  encoder_layers: 10
  decoder_layers: 10
  
training:
  iterations: 500000
  batch_size: 4
  gradient_accumulation: 16
  learning_rate: 2.0e-4
  weight_decay: 0.01
  fp16: true
  gradient_checkpointing: true
  
loss_weights:
  reconstruction: 45.0
  adversarial: 1.0
  feature_matching: 0.1
  
checkpoint:
  save_interval: 10000
  latest_interval: 1000
  keep_last_n: 5
  
logging:
  wandb_project: supertonic-zh
  log_interval: 100
```

### Task 7: 训练恢复机制
```python
# 关键代码片段
def save_checkpoint(state, filepath, is_best=False):
    """保存 checkpoint，包含完整训练状态"""
    torch.save({
        'iteration': state['iteration'],
        'epoch': state['epoch'],
        'model_state_dict': state['model'].state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'scheduler_state_dict': state['scheduler'].state_dict(),
        'best_loss': state['best_loss'],
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }, filepath)
    
def load_checkpoint(filepath, model, optimizer, scheduler):
    """加载 checkpoint 并恢复训练状态"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    return checkpoint['iteration'], checkpoint['epoch'], checkpoint['best_loss']
```

---

## 训练命令

```bash
# 首次训练
python train/train_autoencoder.py --config train/configs/autoencoder.yaml

# 从 checkpoint 恢复
python train/train_autoencoder.py --config train/configs/autoencoder.yaml --resume latest

# 从指定 checkpoint 恢复
python train/train_autoencoder.py --config train/configs/autoencoder.yaml --resume checkpoints/autoencoder_iter_50000.pth
```

---

## 验证清单

- [ ] Encoder 输出 shape 正确：(B, 24, T/hop_length)
- [ ] Decoder 输出 shape 正确：(B, T_audio)
- [ ] Discriminator 可以区分真假音频
- [ ] 损失函数计算正确
- [ ] Checkpoint 保存/加载正常
- [ ] Ctrl+C 优雅退出并保存
- [ ] 恢复训练后 iteration 正确递增
- [ ] WandB 日志正常记录
- [ ] 显存占用 < 8GB（fp16 + gradient checkpointing）

---

## 预计训练时间

RTX 4060 8GB：10-14 天（500k iterations）

监控指标：
- Reconstruction loss 下降
- Adversarial loss 稳定在 0.5-1.0
- 生成音频质量逐步提升
