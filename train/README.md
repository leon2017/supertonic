# Supertonic 中文 TTS 训练

基于 SupertonicTTS 论文复现,使用 AISHELL-3 训练中文 TTS 模型。

## 数据预处理

1. 下载 AISHELL-3:`python train/data/download_aishell3.py --output_dir ./data/aishell3`
2. 预处理数据:`python train/data/preprocess.py --aishell3_dir ./data/aishell3 --output_dir ./data/zh`
3. 构建 Unicode Indexer:`python train/data/unicode_builder.py --data_dir ./data/zh --output unicode_indexer.json`

## 训练

详见各训练脚本的 README。
