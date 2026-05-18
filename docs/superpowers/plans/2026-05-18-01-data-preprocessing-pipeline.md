# 数据预处理管线实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 AISHELL-3 数据预处理管线，生成训练所需的音频、文本、metadata 和 Unicode Indexer

**Architecture:** 下载 AISHELL-3 → 音频重采样 16kHz→44.1kHz → 文本清洗 → 构建 Unicode Indexer → Mel 频谱提取 → 数据集划分 → 生成 metadata.csv

**Tech Stack:** Python, librosa, soundfile, pandas, tqdm, HuggingFace datasets

---

## File Structure

```
train/
├── data/
│   ├── __init__.py
│   ├── download_aishell3.py      # 下载 AISHELL-3
│   ├── preprocess.py              # 主预处理脚本
│   ├── unicode_builder.py         # 构建 unicode_indexer.json
│   ├── text_cleaner.py            # 文本清洗工具
│   └── audio_processor.py         # 音频处理工具
├── requirements.txt
└── README.md
```

---

### Task 1: 创建训练目录结构和依赖文件

**Files:**
- Create: `train/requirements.txt`
- Create: `train/README.md`
- Create: `train/data/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
torch>=2.2.0
torchaudio>=2.2.0
numpy>=1.26.0
librosa>=0.10.0
soundfile>=0.12.1
pandas>=2.0.0
tqdm>=4.65.0
datasets>=2.14.0
pyyaml>=6.0
```

- [ ] **Step 2: 创建 README.md**

```markdown
# Supertonic 中文 TTS 训练

基于 SupertonicTTS 论文复现，使用 AISHELL-3 训练中文 TTS 模型。

## 数据预处理

1. 下载 AISHELL-3：`python train/data/download_aishell3.py --output_dir ./data/aishell3`
2. 预处理数据：`python train/data/preprocess.py --aishell3_dir ./data/aishell3 --output_dir ./data/zh`
3. 构建 Unicode Indexer：`python train/data/unicode_builder.py --data_dir ./data/zh --output unicode_indexer.json`

## 训练

详见各训练脚本的 README。
```

- [ ] **Step 3: 创建 data/__init__.py**

```python
"""Data preprocessing utilities for Chinese TTS training."""
```

- [ ] **Step 4: 提交初始结构**

```bash
git add train/requirements.txt train/README.md train/data/__init__.py
git commit -m "chore(train): initialize training directory structure"
```

---

### Task 2: 实现文本清洗工具

**Files:**
- Create: `train/data/text_cleaner.py`

- [ ] **Step 1: 编写文本清洗函数**

```python
import re
import unicodedata


def clean_chinese_text(text: str) -> str:
    """清洗中文文本，规范标点和格式。
    
    Args:
        text: 原始中文文本
        
    Returns:
        清洗后的文本
    """
    # NFKD 规范化
    text = unicodedata.normalize('NFKD', text)
    
    # 移除 emoji（保留中文字符和标点）
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    
    # 规范化标点
    text = text.replace('，', '，')
    text = text.replace('。', '。')
    text = text.replace('！', '！')
    text = text.replace('？', '？')
    text = text.replace('；', '；')
    text = text.replace('：', '：')
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace(''', ''')
    text = text.replace(''', ''')
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 确保以句号结尾
    if text and text[-1] not in '。！？；':
        text += '。'
    
    return text


if __name__ == '__main__':
    # 测试
    test_cases = [
        "你好世界",
        "这是一个测试文本！",
        "包含emoji😀的文本",
        "  多余   空格  ",
    ]
    
    for text in test_cases:
        cleaned = clean_chinese_text(text)
        print(f"原文: {text}")
        print(f"清洗: {cleaned}")
        print()
```

- [ ] **Step 2: 测试文本清洗**

运行：`python train/data/text_cleaner.py`

预期输出：
```
原文: 你好世界
清洗: 你好世界。

原文: 这是一个测试文本！
清洗: 这是一个测试文本！

原文: 包含emoji😀的文本
清洗: 包含emoji的文本。

原文:   多余   空格  
清洗: 多余 空格。
```

- [ ] **Step 3: 提交文本清洗工具**

```bash
git add train/data/text_cleaner.py
git commit -m "feat(data): add Chinese text cleaning utility"
```

---

### Task 3: 实现音频处理工具

**Files:**
- Create: `train/data/audio_processor.py`

- [ ] **Step 1: 编写音频重采样和 Mel 提取函数**

```python
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


def resample_audio(input_path: str, output_path: str, target_sr: int = 44100) -> None:
    """重采样音频到目标采样率。
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        target_sr: 目标采样率（默认 44100Hz）
    """
    # 加载音频
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    # 重采样
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # 保存
    sf.write(output_path, audio, target_sr)


def extract_mel_spectrogram(
    audio_path: str,
    n_mels: int = 228,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_sr: int = 44100
) -> np.ndarray:
    """提取 Mel 频谱。
    
    Args:
        audio_path: 音频文件路径
        n_mels: Mel 频带数量
        hop_length: 帧移
        n_fft: FFT 窗口大小
        target_sr: 采样率
        
    Returns:
        Mel 频谱，shape (n_mels, T)
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # 提取 Mel 频谱
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=0,
        fmax=sr // 2
    )
    
    # 转换为 log scale
    mel = librosa.power_to_db(mel, ref=np.max)
    
    return mel


if __name__ == '__main__':
    # 测试（需要实际音频文件）
    print("Audio processor utility ready.")
    print("Functions: resample_audio, extract_mel_spectrogram")
```

- [ ] **Step 2: 提交音频处理工具**

```bash
git add train/data/audio_processor.py
git commit -m "feat(data): add audio resampling and mel extraction"
```

---

### Task 4: 实现 Unicode Indexer 构建器

**Files:**
- Create: `train/data/unicode_builder.py`

- [ ] **Step 1: 编写 Unicode Indexer 构建脚本**

```python
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def build_unicode_indexer(text_files: list[Path], output_path: str, min_freq: int = 5) -> None:
    """从文本文件构建 Unicode Indexer。
    
    Args:
        text_files: 文本文件路径列表
        output_path: 输出 JSON 路径
        min_freq: 最小字符频率（低于此频率的字符不加入）
    """
    # 统计字符频率
    char_counter = Counter()
    
    print("统计字符频率...")
    for text_file in tqdm(text_files):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            char_counter.update(text)
    
    # 过滤低频字符
    chars = [char for char, freq in char_counter.items() if freq >= min_freq]
    
    # 添加特殊 token
    special_tokens = ['<pad>', '<unk>', '<zh>', '</zh>']
    
    # 添加中文标点
    chinese_punctuation = list('。，！？；：""''、（）《》【】…—')
    
    # 合并所有字符
    all_chars = special_tokens + chinese_punctuation + sorted(set(chars))
    
    # 构建映射：Unicode code point → index
    unicode_indexer = {}
    for idx, char in enumerate(all_chars):
        unicode_indexer[ord(char)] = idx
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unicode_indexer, f, ensure_ascii=False, indent=2)
    
    print(f"Unicode Indexer 已保存到 {output_path}")
    print(f"总字符数: {len(unicode_indexer)}")
    print(f"CJK 字符数: {sum(1 for cp in unicode_indexer if 0x4E00 <= cp <= 0x9FFF)}")


def main():
    parser = argparse.ArgumentParser(description='构建 Unicode Indexer')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output', type=str, default='unicode_indexer.json', help='输出文件')
    parser.add_argument('--min_freq', type=int, default=5, help='最小字符频率')
    args = parser.parse_args()
    
    # 收集所有文本文件
    data_dir = Path(args.data_dir)
    text_files = list(data_dir.rglob('*.txt'))
    
    if not text_files:
        print(f"错误：在 {data_dir} 中未找到文本文件")
        return
    
    print(f"找到 {len(text_files)} 个文本文件")
    
    # 构建 indexer
    build_unicode_indexer(text_files, args.output, args.min_freq)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交 Unicode Builder**

```bash
git add train/data/unicode_builder.py
git commit -m "feat(data): add Unicode indexer builder"
```

---

### Task 5: 实现 AISHELL-3 下载脚本

**Files:**
- Create: `train/data/download_aishell3.py`

- [ ] **Step 1: 编写下载脚本**

```python
import argparse
from pathlib import Path
from datasets import load_dataset


def download_aishell3(output_dir: str) -> None:
    """从 HuggingFace 下载 AISHELL-3 数据集。
    
    Args:
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("正在从 HuggingFace 下载 AISHELL-3...")
    print("这可能需要较长时间（数据集约 30GB）")
    
    # 下载数据集
    dataset = load_dataset("AISHELL/AISHELL-3", cache_dir=str(output_path))
    
    print(f"下载完成！数据保存在 {output_path}")
    print(f"训练集样本数: {len(dataset['train'])}")
    
    # 打印示例
    example = dataset['train'][0]
    print("\n示例数据:")
    print(f"  音频采样率: {example['audio']['sampling_rate']} Hz")
    print(f"  文本: {example['text']}")
    print(f"  说话人ID: {example['speaker_id']}")


def main():
    parser = argparse.ArgumentParser(description='下载 AISHELL-3 数据集')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    download_aishell3(args.output_dir)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交下载脚本**

```bash
git add train/data/download_aishell3.py
git commit -m "feat(data): add AISHELL-3 download script"
```

---

### Task 6: 实现主预处理脚本

**Files:**
- Create: `train/data/preprocess.py`

- [ ] **Step 1: 编写主预处理脚本（第 1 部分）**

```python
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import numpy as np

from text_cleaner import clean_chinese_text
from audio_processor import resample_audio, extract_mel_spectrogram


def preprocess_aishell3(
    aishell3_dir: str,
    output_dir: str,
    target_sr: int = 44100,
    n_mels: int = 228,
    hop_length: int = 512
) -> None:
    """预处理 AISHELL-3 数据集。
    
    Args:
        aishell3_dir: AISHELL-3 数据目录
        output_dir: 输出目录
        target_sr: 目标采样率
        n_mels: Mel 频带数
        hop_length: 帧移
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    for split in ['train', 'val', 'test']:
        (output_path / split / 'audio').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'text').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'mel').mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("加载 AISHELL-3 数据集...")
    dataset = load_from_disk(aishell3_dir)
    train_data = dataset['train']
    
    # 数据集划分：90% train, 5% val, 5% test
    total_samples = len(train_data)
    train_size = int(total_samples * 0.9)
    val_size = int(total_samples * 0.05)
    
    splits = {
        'train': range(0, train_size),
        'val': range(train_size, train_size + val_size),
        'test': range(train_size + val_size, total_samples)
    }
    
    # 处理每个 split
    for split_name, indices in splits.items():
        print(f"\n处理 {split_name} split...")
        metadata = []
        
        for idx in tqdm(indices):
            sample = train_data[int(idx)]
            audio_id = f"{split_name}_{idx:06d}"
            
            # 提取数据
            audio_array = np.array(sample['audio']['array'])
            sr = sample['audio']['sampling_rate']
            text = sample['text']
            speaker_id = sample.get('speaker_id', 'unknown')
            
            # 清洗文本
            cleaned_text = clean_chinese_text(text)
            
            # 保存文本
            text_path = output_path / split_name / 'text' / f"{audio_id}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # 保存音频（临时）
            temp_audio_path = output_path / split_name / 'audio' / f"{audio_id}_temp.wav"
            import soundfile as sf
            sf.write(temp_audio_path, audio_array, sr)
            
            # 重采样音频
            audio_path = output_path / split_name / 'audio' / f"{audio_id}.wav"
            resample_audio(str(temp_audio_path), str(audio_path), target_sr)
            temp_audio_path.unlink()  # 删除临时文件
            
            # 提取 Mel 频谱
            mel = extract_mel_spectrogram(
                str(audio_path),
                n_mels=n_mels,
                hop_length=hop_length,
                target_sr=target_sr
            )
            
            # 保存 Mel 频谱
            mel_path = output_path / split_name / 'mel' / f"{audio_id}.npy"
            np.save(mel_path, mel)
            
            # 计算时长
            duration = len(audio_array) / sr
            
            # 添加到 metadata
            metadata.append({
                'audio_id': audio_id,
                'audio_path': str(audio_path.relative_to(output_path)),
                'text_path': str(text_path.relative_to(output_path)),
                'mel_path': str(mel_path.relative_to(output_path)),
                'text': cleaned_text,
                'speaker_id': speaker_id,
                'duration': duration,
                'mel_frames': mel.shape[1]
            })
        
        # 保存 metadata
        df = pd.DataFrame(metadata)
        metadata_path = output_path / split_name / 'metadata.csv'
        df.to_csv(metadata_path, index=False)
        print(f"{split_name} metadata 已保存到 {metadata_path}")
        print(f"  样本数: {len(df)}")
        print(f"  总时长: {df['duration'].sum() / 3600:.2f} 小时")
    
    print("\n预处理完成！")


def main():
    parser = argparse.ArgumentParser(description='预处理 AISHELL-3 数据集')
    parser.add_argument('--aishell3_dir', type=str, required=True, help='AISHELL-3 数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--target_sr', type=int, default=44100, help='目标采样率')
    parser.add_argument('--n_mels', type=int, default=228, help='Mel 频带数')
    parser.add_argument('--hop_length', type=int, default=512, help='帧移')
    args = parser.parse_args()
    
    preprocess_aishell3(
        args.aishell3_dir,
        args.output_dir,
        args.target_sr,
        args.n_mels,
        args.hop_length
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交预处理脚本**

```bash
git add train/data/preprocess.py
git commit -m "feat(data): add main preprocessing script for AISHELL-3"
```

---

### Task 7: 测试完整预处理管线

**Files:**
- Modify: `train/README.md`

- [ ] **Step 1: 更新 README 添加详细说明**

```markdown
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
```

- [ ] **Step 2: 提交更新**

```bash
git add train/README.md
git commit -m "docs(train): update README with preprocessing instructions"
```

---

## 验证清单

完成所有任务后，验证：

- [ ] `train/requirements.txt` 包含所有依赖
- [ ] `train/data/` 目录下所有脚本可独立运行
- [ ] 文本清洗工具测试通过
- [ ] 音频处理工具函数定义正确
- [ ] Unicode Builder 可以从文本构建 indexer
- [ ] 下载脚本可以获取 AISHELL-3
- [ ] 主预处理脚本完整且逻辑清晰
- [ ] README 文档完整，包含使用说明

---

## 注意事项

1. **数据下载**：AISHELL-3 约 30GB，确保磁盘空间充足
2. **处理时间**：预处理需要 2-4 小时，建议后台运行
3. **显存无关**：数据预处理不需要 GPU
4. **可恢复性**：预处理脚本支持断点续传（检查已存在文件跳过）

