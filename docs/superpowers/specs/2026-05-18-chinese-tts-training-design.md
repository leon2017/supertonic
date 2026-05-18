# 中文 TTS 训练设计文档

## 概述

基于 SupertonicTTS 论文复现模型架构，使用 AISHELL-3 中文语料从头训练一个支持中文的 TTS 模型，最终导出 ONNX 接入现有推理 SDK。

## 背景与动机

Supertonic 支持 31 种语言但不包含中文。官方未提供 PyTorch 权重，无法直接微调。因此选择根据论文重写模型结构，用中文语料独立训练，导出 ONNX 后集成到现有 SDK。

## 参考论文

- [SupertonicTTS: Towards Highly Scalable and Efficient Text-to-Speech System](https://arxiv.org/abs/2503.23108v2)

## 模型架构

总参数量：~44M

### Speech Autoencoder（语音自编码器）

- **Latent Encoder**: 10 层 ConvNeXt (kernel=7, intermediate=2048)，228-dim mel → 24-dim latent
- **Latent Decoder**: 10 层 dilated ConvNeXt (dilation=[1,2,4,1,2,4,1,1,1,1])，24-dim latent → 44.1kHz waveform
- **Discriminator**: Multi-period + Multi-resolution discriminator

### Text Encoder（文本编码器）

- Character embedding: 128-dim lookup table
- 6 ConvNeXt blocks
- 4 Self-Attention blocks (512 channels, 4 heads)
- 2 Cross-Attention layers (与 Reference Encoder 输出交互)

### Reference Encoder（参考编码器）

- Linear + 6 ConvNeXt blocks (kernel=5, intermediate=512)
- 2 Cross-Attention layers
- 输出 50 个固定大小的 style vectors

### VF Estimator（向量场估计器）

- Linear projection → 256-dim
- 4 重复块，每块含：
  - 4 dilated ConvNeXt (dilation=1,2,4,8)
  - 2 standard ConvNeXt
  - Time/Text/Reference conditioning blocks
- 4 final ConvNeXt blocks

### Duration Predictor（时长预测器）

- DP Reference Encoder: Linear + 4 ConvNeXt (kernel=5, intermediate=256) + 2 cross-attention → 64-dim
- DP Text Encoder: Character embedding (64-dim) + 6 ConvNeXt + 2 self-attention (256 channels, 2 heads)
- Duration Estimator: 两层 Linear + PReLU (164-dim → scalar)

## 关键设计决策

### 时间压缩

Latent 从 (24, T) reshape 为 (144, T/6)，压缩因子 Kc=6，减少 VF Estimator 处理的序列长度。

### Classifier-Free Guidance

- 训练时 5% 概率 drop condition
- 推理时 CFG coefficient = 3

### 字符级输入

直接使用 Unicode 字符嵌入，无需 G2P (Grapheme-to-Phoneme) 转换。中文汉字直接作为 token 输入。

### Utterance-Level Duration

预测整句时长而非音素级时长，简化管线。

## 训练数据

### AISHELL-3

- 规模：85+ 小时多说话人普通话语音
- 说话人：~218 位
- 采样率：16kHz（需上采样至 44.1kHz）
- 转录：汉字 + 拼音（只使用汉字）

### 数据预处理流程

1. 音频重采样 16kHz → 44.1kHz (librosa.resample)
2. 文本清洗：去除标注噪声、规范标点
3. 构建 Unicode Indexer：扩展加入 CJK Unified Ideographs (U+4E00-U+9FFF) + 中文标点
4. Mel 频谱提取：228-dim, 44.1kHz, hop=512
5. 数据集划分：train/val/test = 90/5/5
6. Metadata 生成：CSV (audio_path, text, speaker_id, duration)

## 硬件约束与适配

### 硬件

- GPU: RTX 4060 8GB
- CUDA: 已配置

### 显存优化策略

| 技术 | 效果 |
|------|------|
| AMP (fp16) | 显存减半 |
| Gradient Accumulation | 小 batch 模拟大 batch |
| Gradient Checkpointing | 用计算换显存，省 ~40% |
| 音频裁剪 | 固定长度片段避免长序列 |

### 训练配置

| 阶段 | 组件 | 实际 Batch | Accumulation | 等效 Batch | 预计时间 |
|------|------|-----------|-------------|-----------|---------|
| 1 | Speech Autoencoder | 4 | 16 | 64 | 10-14 天 |
| 2 | Text-to-Latent | 2-4 | 16-32 | 64 | 7-14 天 |
| 3 | Duration Predictor | 8 | 8 | 64 | 1-2 天 |

**总训练时间预估：3-4 周**

### 风险与缓解

- **VF Estimator 显存不足**：如 4 重复块放不下，减少到 2-3 块（牺牲部分质量）
- **训练中断**：每 10k iterations 保存 checkpoint
- **16kHz 上采样质量上限**：可接受，对 flow matching 训练足够

## 训练损失函数

### 阶段 1：Speech Autoencoder

- **重建损失**：Multi-resolution mel spectrogram L1 (FFT: 1024/2048/4096)，权重 λ_recon=45
- **对抗损失**：Multi-period + Multi-resolution discriminator，权重 λ_adv=1
- **Feature Matching 损失**：权重 λ_fm=0.1

### 阶段 2：Text-to-Latent (Flow Matching)

```
ℒ_TTL = 𝔼_{t,(z₁,c),p(z₀)} ‖m · (v(z_t, z_ref, c, t) − (z₁ − (1−σ_min)z₀))‖₁
```

- t ~ U[0,1], p(z₀) = N(0,1), σ_min = 10⁻⁸
- Euler's method, 训练时随机 t，推理时 32 步（可减至 8 步）

### 阶段 3：Duration Predictor

- L1 loss: |predicted_duration - ground_truth_duration|

## 训练超参数

```yaml
# 通用
optimizer: AdamW
weight_decay: 0.01
lr_scheduler: step (halve every 300k)
seed: 42
fp16: true
gradient_checkpointing: true

# Autoencoder
autoencoder:
  learning_rate: 2e-4
  iterations: 500000
  batch_size: 4
  gradient_accumulation: 16
  segment_length: 44100  # 1 秒音频片段
  mel_channels: 228
  latent_dim: 24
  hop_length: 512

# Text-to-Latent
text_to_latent:
  learning_rate: 5e-4
  iterations: 300000
  batch_size: 4
  gradient_accumulation: 16
  temporal_compression: 6
  cfg_dropout: 0.05
  cfg_scale: 3.0  # 推理时
  euler_steps: 32  # 推理时

# Duration Predictor
duration:
  learning_rate: 1e-4
  iterations: 100000
  batch_size: 8
  gradient_accumulation: 8
```

## 代码结构

```
supertonic/
├── train/
│   ├── configs/
│   │   ├── autoencoder.yaml
│   │   ├── text_to_latent.yaml
│   │   └── duration.yaml
│   ├── data/
│   │   ├── dataset.py              # Dataset + DataLoader
│   │   ├── preprocess.py           # AISHELL-3 预处理
│   │   └── unicode_builder.py      # 构建/扩展 unicode_indexer.json
│   ├── models/
│   │   ├── convnext.py             # ConvNeXt block
│   │   ├── attention.py            # Self/Cross-Attention
│   │   ├── autoencoder.py          # Latent Encoder + Decoder
│   │   ├── text_encoder.py         # Text Encoder
│   │   ├── vf_estimator.py         # Vector Field Estimator
│   │   ├── duration_predictor.py   # Duration Predictor
│   │   ├── reference_encoder.py    # Reference Encoder
│   │   └── discriminator.py        # Discriminators
│   ├── losses/
│   │   ├── flow_matching.py
│   │   ├── reconstruction.py
│   │   └── adversarial.py
│   ├── train_autoencoder.py
│   ├── train_text_to_latent.py
│   ├── train_duration.py
│   ├── export_onnx.py
│   └── requirements.txt
```

## ONNX 导出

训练完成后导出 4 个 ONNX 模型：

1. `duration_predictor.onnx` — 输入: text_ids, style_dp, text_mask → 输出: duration
2. `text_encoder.onnx` — 输入: text_ids, style_ttl, text_mask → 输出: text_emb
3. `vector_estimator.onnx` — 输入: noisy_latent, text_emb, style_ttl, text_mask, latent_mask, current_step, total_step → 输出: xt
4. `vocoder.onnx` — 输入: latent → 输出: wav

导出后验证：PyTorch 与 ONNX 输出误差 < 1e-5。

## SDK 集成

### 1. Unicode Indexer 更新

扩展 `unicode_indexer.json`：
- 添加 CJK Unified Ideographs (U+4E00-U+9FFF)
- 添加中文标点（。，！？；：""''、）
- 添加 `<zh>` / `</zh>` 语言标签

### 2. AVAILABLE_LANGS 更新

在以下 10 个文件中添加 `"zh"`：
- `py/helper.py`
- `nodejs/helper.js`
- `web/helper.js`
- `go/helper.go`
- `rust/src/*.rs`
- `cpp/helper.cpp`
- `java/Helper.java`
- `csharp/Helper.cs`
- `swift/Sources/**/*.swift`
- `flutter/lib/**/*.dart`

### 3. 中文文本预处理

- `max_len = 120`（已有 CJK 逻辑）
- 分句：按 `。！？；` 切分
- 现有 `_preprocess_text` 已兼容中文终止标点

### 4. Voice Style

训练时 Reference Encoder 从音频中提取 style vectors（style_ttl 和 style_dp）。训练完成后，对每个目标说话人的参考音频运行 Reference Encoder，将输出保存为 JSON 格式供推理使用。推理时直接加载 JSON，不再需要参考音频。

## 验证标准

1. 合成中文语音无运行时错误
2. 音频时长与文本长度成正比
3. 输出 WAV：44.1kHz, 16-bit, mono
4. 主观听感：发音清晰、韵律自然
5. 不影响其他语言推理（独立模型文件）

## 交付物

- `train/` 目录：完整训练代码
- 4 个 ONNX 模型文件
- 更新的 `unicode_indexer.json`
- 中文 voice style JSON 文件
- 所有 SDK 的 `AVAILABLE_LANGS` 更新

