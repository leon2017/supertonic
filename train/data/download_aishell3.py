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

    dataset = load_dataset("AISHELL/AISHELL-3", cache_dir=str(output_path))

    print(f"下载完成！数据保存在 {output_path}")
    print(f"训练集样本数: {len(dataset['train'])}")

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
