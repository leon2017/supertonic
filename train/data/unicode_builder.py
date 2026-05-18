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
    char_counter = Counter()

    print("统计字符频率...")
    for text_file in tqdm(text_files):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
            char_counter.update(text)

    chars = [char for char, freq in char_counter.items() if freq >= min_freq]

    special_tokens = ["<pad>", "<unk>", "<zh>", "</zh>"]

    chinese_punctuation = list("。，！？；：“”‘’、（）《》【】…—")

    all_chars = special_tokens + chinese_punctuation + sorted(set(chars))

    unicode_indexer = {}
    for idx, char in enumerate(all_chars):
        unicode_indexer[ord(char)] = idx

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unicode_indexer, f, ensure_ascii=False, indent=2)

    print(f"Unicode Indexer 已保存到 {output_path}")
    print(f"总字符数: {len(unicode_indexer)}")
    print(f"CJK 字符数: {sum(1 for cp in unicode_indexer if 0x4E00 <= cp <= 0x9FFF)}")


def main():
    parser = argparse.ArgumentParser(description="构建 Unicode Indexer")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output", type=str, default="unicode_indexer.json", help="输出文件")
    parser.add_argument("--min_freq", type=int, default=5, help="最小字符频率")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    text_files = list(data_dir.rglob("*.txt"))

    if not text_files:
        print(f"错误：在 {data_dir} 中未找到文本文件")
        return

    print(f"找到 {len(text_files)} 个文本文件")

    build_unicode_indexer(text_files, args.output, args.min_freq)


if __name__ == "__main__":
    main()
