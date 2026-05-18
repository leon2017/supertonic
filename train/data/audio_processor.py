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
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    sf.write(output_path, audio, target_sr)


def extract_mel_spectrogram(
    audio_path: str,
    n_mels: int = 228,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_sr: int = 44100,
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
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=0,
        fmax=sr // 2,
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    return mel


if __name__ == "__main__":
    print("Audio processor utility ready.")
    print("Functions: resample_audio, extract_mel_spectrogram")
