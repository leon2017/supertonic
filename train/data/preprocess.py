import argparse
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from text_cleaner import clean_chinese_text
from audio_processor import resample_audio, extract_mel_spectrogram


def parse_content_txt(content_path: Path) -> dict[str, str]:
    """解析 content.txt，提取文件名到中文文本的映射。"""
    mapping = {}
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            filename = parts[0]
            tokens = parts[1].split()
            chars = [t for i, t in enumerate(tokens) if i % 2 == 0]
            text = "".join(chars)
            mapping[filename] = text
    return mapping


def find_wav_files(wav_dir: Path, text_mapping: dict[str, str]) -> list[dict]:
    """查找所有有对应文本的 WAV 文件。"""
    samples = []
    for speaker_dir in sorted(wav_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for wav_file in sorted(speaker_dir.glob("*.wav")):
            if wav_file.name in text_mapping:
                samples.append({
                    "wav_path": wav_file,
                    "filename": wav_file.name,
                    "speaker_id": speaker_id,
                    "text": text_mapping[wav_file.name],
                })
    return samples


def preprocess_aishell3(
    aishell3_dir: str,
    output_dir: str,
    target_sr: int = 44100,
    n_mels: int = 228,
    hop_length: int = 512,
    seed: int = 42,
) -> None:
    aishell3_path = Path(aishell3_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        (output_path / split / "audio").mkdir(parents=True, exist_ok=True)
        (output_path / split / "mel").mkdir(parents=True, exist_ok=True)

    content_path = aishell3_path / "train" / "content.txt"
    wav_dir = aishell3_path / "train" / "wav"

    print("解析 content.txt...")
    text_mapping = parse_content_txt(content_path)
    print(f"  转录条目: {len(text_mapping)}")

    print("查找 WAV 文件...")
    samples = find_wav_files(wav_dir, text_mapping)
    print(f"  匹配样本: {len(samples)}")
    print(f"  说话人数: {len(set(s['speaker_id'] for s in samples))}")

    random.seed(seed)
    random.shuffle(samples)

    val_size = int(len(samples) * 0.05)
    splits = {
        "val": samples[:val_size],
        "train": samples[val_size:],
    }

    for split_name, split_samples in splits.items():
        print(f"\n处理 {split_name} split ({len(split_samples)} 样本)...")
        metadata = []

        for sample in tqdm(split_samples):
            audio_id = sample["filename"].replace(".wav", "")
            cleaned_text = clean_chinese_text(sample["text"])

            audio_out = output_path / split_name / "audio" / f"{audio_id}.wav"
            resample_audio(str(sample["wav_path"]), str(audio_out), target_sr)

            mel = extract_mel_spectrogram(
                str(audio_out),
                n_mels=n_mels,
                hop_length=hop_length,
                target_sr=target_sr,
            )
            mel_path = output_path / split_name / "mel" / f"{audio_id}.npy"
            np.save(mel_path, mel)

            duration = librosa.get_duration(path=str(audio_out))

            metadata.append({
                "audio_id": audio_id,
                "audio_path": f"{split_name}/audio/{audio_id}.wav",
                "mel_path": f"{split_name}/mel/{audio_id}.npy",
                "text": cleaned_text,
                "speaker_id": sample["speaker_id"],
                "duration": duration,
                "mel_frames": mel.shape[1],
            })

        df = pd.DataFrame(metadata)
        metadata_path = output_path / split_name / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        print(f"  {split_name}: {len(df)} 样本, {df['duration'].sum()/3600:.2f} 小时")

    print("\n预处理完成！")


def main():
    parser = argparse.ArgumentParser(description="预处理 AISHELL-3 数据集")
    parser.add_argument(
        "--aishell3_dir", type=str, required=True,
        help="AISHELL-3 数据目录（包含 train/wav 和 train/content.txt）",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target_sr", type=int, default=44100, help="目标采样率")
    parser.add_argument("--n_mels", type=int, default=228, help="Mel 频带数")
    parser.add_argument("--hop_length", type=int, default=512, help="帧移")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    preprocess_aishell3(
        args.aishell3_dir,
        args.output_dir,
        args.target_sr,
        args.n_mels,
        args.hop_length,
        args.seed,
    )


if __name__ == '__main__':
    main()
