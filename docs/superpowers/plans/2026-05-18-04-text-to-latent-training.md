# Text-to-Latent 训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现并训练 Text-to-Latent 模块（TextEncoder + ReferenceEncoder + VF Estimator），支持暂停恢复

**Architecture:** Character Embedding + ConvNeXt + Self/Cross-Attention (TextEncoder) + Reference Encoder (style extraction) + Flow Matching VF Estimator

**Tech Stack:** PyTorch 2.2+, Flow Matching, Classifier-Free Guidance, fp16, gradient checkpointing

---

## File Structure

```
train/
├── models/
│   ├── text_encoder.py          # Text Encoder
│   ├── reference_encoder.py     # Reference Encoder (style)
│   ├── vf_estimator.py          # Vector Field Estimator
│   └── dataset_ttl.py           # Dataset for text-to-latent
├── losses/
│   └── flow_matching.py         # Flow matching loss
├── configs/
│   └── text_to_latent.yaml      # Training config
└── train_text_to_latent.py      # Main training script
```

---

## 关键任务概要

### Task 1: 实现 Text Encoder
- Character embedding (128-dim) + Unicode indexer 支持
- 6 ConvNeXt blocks
- 4 Self-Attention blocks (512 channels, 4 heads)
- 2 Cross-Attention layers (与 Reference Encoder 交互)

### Task 2: 实现 Reference Encoder
- Linear + 6 ConvNeXt blocks (kernel=5, intermediate=512)
- 2 Cross-Attention layers
- 输出 50 个固定大小的 style vectors

### Task 3: 实现 VF Estimator
- Linear projection → 256-dim
- 4 重复块（每块：4 dilated ConvNeXt + 2 standard ConvNeXt + conditioning）
- Time/Text/Reference conditioning
- 支持 Classifier-Free Guidance (5% dropout)

### Task 4: 实现 Flow Matching Loss
```python
def flow_matching_loss(model_output, target_velocity, mask):
    """
    ℒ_TTL = 𝔼_{t,(z₁,c),p(z₀)} ‖m · (v(z_t, z_ref, c, t) − (z₁ − (1−σ_min)z₀))‖₁
    
    Args:
        model_output: Predicted velocity, shape (B, C, T)
        target_velocity: Ground truth velocity, shape (B, C, T)
        mask: Latent mask, shape (B, 1, T)
    """
    loss = torch.abs(model_output - target_velocity)
    loss = loss * mask
    return loss.sum() / mask.sum()
```

### Task 5: 实现训练脚本
- 时间压缩：latent (24, T) → (144, T/6)
- 随机 t ~ U[0,1] 采样
- CFG dropout (5%)
- Euler's method 推理（32 步）
- Checkpoint 管理（同 autoencoder）

### Task 6: 配置文件
```yaml
# text_to_latent.yaml
data:
  train_dir: ./data/zh/train
  val_dir: ./data/zh/val
  unicode_indexer: ./assets/onnx/unicode_indexer_zh.json
  
model:
  char_embedding_dim: 128
  text_channels: 512
  text_num_heads: 4
  latent_dim: 24
  temporal_compression: 6  # Kc
  vf_channels: 256
  vf_num_blocks: 4
  
training:
  iterations: 300000
  batch_size: 4
  gradient_accumulation: 16
  learning_rate: 5.0e-4
  cfg_dropout: 0.05
  cfg_scale: 3.0  # inference only
  euler_steps: 32  # inference only
  
checkpoint:
  save_interval: 10000
  latest_interval: 1000
```

---

## 训练命令

```bash
# 首次训练（需要先训练好 autoencoder）
python train/train_text_to_latent.py \
    --config train/configs/text_to_latent.yaml \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pth

# 恢复训练
python train/train_text_to_latent.py \
    --config train/configs/text_to_latent.yaml \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pth \
    --resume latest
```

---

## 验证清单

- [ ] Text Encoder 正确编码 Unicode 字符
- [ ] Reference Encoder 输出固定 50 个 style vectors
- [ ] VF Estimator 支持时间压缩（Kc=6）
- [ ] Flow matching loss 计算正确
- [ ] CFG dropout 正常工作
- [ ] Euler's method 推理生成合理 latent
- [ ] Checkpoint 保存/恢复正常
- [ ] 显存占用 < 8GB

---

## 预计训练时间

RTX 4060 8GB：7-14 天（300k iterations）

监控指标：
- Flow matching loss 下降
- 生成 latent 质量（通过 vocoder 解码验证）
- Text-audio 对齐质量
