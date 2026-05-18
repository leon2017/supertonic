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
