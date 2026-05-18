# Duration Predictor 训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现并训练 Duration Predictor（预测整句时长），支持暂停恢复

**Architecture:** DP Reference Encoder + DP Text Encoder + Duration Estimator (2-layer MLP)

**Tech Stack:** PyTorch 2.2+, L1 loss, fp16

---

## File Structure

```
train/
├── models/
│   ├── duration_predictor.py    # Complete duration predictor
│   └── dataset_duration.py      # Dataset for duration
├── configs/
│   └── duration.yaml            # Training config
└── train_duration.py            # Main training script
```

---

## 关键任务概要

### Task 1: 实现 DP Reference Encoder
- Linear + 4 ConvNeXt blocks (kernel=5, intermediate=256)
- 2 Cross-Attention layers
- 输出 64-dim style vector

### Task 2: 实现 DP Text Encoder
- Character embedding (64-dim)
- 6 ConvNeXt blocks
- 2 Self-Attention blocks (256 channels, 2 heads)

### Task 3: 实现 Duration Estimator
```python
class DurationEstimator(nn.Module):
    def __init__(self, input_dim=164, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, text_emb, style_dp):
        # text_emb: (B, C_text), style_dp: (B, C_style)
        x = torch.cat([text_emb, style_dp], dim=-1)  # (B, 164)
        x = self.fc1(x)
        x = self.act(x)
        duration = self.fc2(x).squeeze(-1)  # (B,)
        return duration
```

### Task 4: 实现 L1 Loss
```python
def duration_loss(pred_duration, target_duration):
    return torch.abs(pred_duration - target_duration).mean()
```

### Task 5: 配置文件
```yaml
# duration.yaml
data:
  train_dir: ./data/zh/train
  val_dir: ./data/zh/val
  unicode_indexer: ./assets/onnx/unicode_indexer_zh.json
  
model:
  char_embedding_dim: 64
  text_channels: 256
  text_num_heads: 2
  style_dim: 64
  hidden_dim: 256
  
training:
  iterations: 100000
  batch_size: 8
  gradient_accumulation: 8
  learning_rate: 1.0e-4
  
checkpoint:
  save_interval: 10000
  latest_interval: 1000
```

---

## 训练命令

```bash
# 首次训练
python train/train_duration.py --config train/configs/duration.yaml

# 恢复训练
python train/train_duration.py --config train/configs/duration.yaml --resume latest
```

---

## 验证清单

- [ ] Duration Predictor 输出 shape 正确：(B,)
- [ ] L1 loss 计算正确
- [ ] 预测时长与真实时长误差 < 10%
- [ ] Checkpoint 保存/恢复正常

---

## 预计训练时间

RTX 4060 8GB：1-2 天（100k iterations）

监控指标：
- L1 loss 下降
- 预测时长 vs 真实时长相关性
