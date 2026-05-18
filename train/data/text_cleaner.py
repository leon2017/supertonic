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
    text = text.replace('‘', '’')
    text = text.replace('’', '’')

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # 确保以句号结尾（兼容 NFKD 规范化后的 ASCII 标点）
    if text and text[-1] not in '。！？；!?':
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
