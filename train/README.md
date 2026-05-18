# Supertonic 中文 TTS 训练

基于 SupertonicTTS 论文复现，使用 AISHELL-3 训练中文 TTS 模型。

## 环境配置

```bash
cd train
pip install -r requirements.txt
```

## 数据预处理完整流程

### 1. 下载 AISHELL-3

```bash
python train/data/download_aishell3.py --output_dir ./data/aishell3
```

预计下载时间：30-60 分钟（取决于网速）
数据集大小：约 30GB

### 2. 预处理数据

```bash
python train/data/preprocess.py \
    --aishell3_dir ./data/aishell3 \
    --output_dir ./data/zh \
    --target_sr 44100 \
    --n_mels 228 \
    --hop_length 512
```

预计处理时间：2-4 小时
输出目录结构：
```
data/zh/
├── train/
│   ├── audio/          # 44.1kHz WAV 文件
│   ├── text/           # 清洗后的文本
│   ├── mel/            # Mel 频谱 (.npy)
│   └── metadata.csv    # 元数据
├── val/
└── test/
```

### 3. 构建 Unicode Indexer

```bash
python train/data/unicode_builder.py \
    --data_dir ./data/zh \
    --output ./assets/onnx/unicode_indexer_zh.json \
    --min_freq 5
```

输出：`unicode_indexer_zh.json`，包含所有中文字符的 Unicode → index 映射。

## 数据集统计

预期输出（AISHELL-3）：
- 训练集：~76 小时，~19000 样本
- 验证集：~4 小时，~1000 样本
- 测试集：~4 小时，~1000 样本
- 说话人数：~218 位

## 下一步

数据预处理完成后，可以开始训练：
1. Speech Autoencoder 训练
2. Text-to-Latent 训练
3. Duration Predictor 训练
