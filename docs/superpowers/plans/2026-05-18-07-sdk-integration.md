# SDK 集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将中文支持集成到所有 SDK，更新 AVAILABLE_LANGS 并测试

**Architecture:** 更新 10 个 SDK 文件 + 中文测试用例

**Tech Stack:** Python, Node.js, Go, Rust, C++, Java, C#, Swift, Flutter

---

## File Structure

需要修改的文件：
```
py/helper.py
nodejs/helper.js
web/helper.js
go/helper.go
rust/src/lib.rs (或相关文件)
cpp/helper.cpp
java/Helper.java
csharp/Helper.cs
swift/Sources/**/Helper.swift
flutter/lib/**/helper.dart
```

---

## 关键任务概要

### Task 1: 更新 Python SDK
```python
# py/helper.py
AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi", "na", "zh"  # 添加 zh
]

# 确认 CJK 逻辑已包含 zh
def chunk_text(text: str, lang: str, max_len: int = 300) -> list[str]:
    if lang in ['ko', 'ja', 'zh']:  # 添加 zh
        max_len = 120
    # ... rest of the function
```

### Task 2: 更新 Node.js SDK
```javascript
// nodejs/helper.js
const AVAILABLE_LANGS = [
  "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
  "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
  "sl", "sv", "tr", "uk", "vi", "na", "zh"  // 添加 zh
];

function chunkText(text, lang, maxLen = 300) {
  if (['ko', 'ja', 'zh'].includes(lang)) {  // 添加 zh
    maxLen = 120;
  }
  // ... rest of the function
}
```

### Task 3: 更新其他 8 个 SDK
- Web (JavaScript)
- Go
- Rust
- C++
- Java
- C#
- Swift
- Flutter (Dart)

每个 SDK 的修改模式相同：
1. 在 AVAILABLE_LANGS 列表末尾添加 "zh"
2. 在 CJK 语言判断逻辑中添加 "zh"（如果有 max_len 或分句逻辑）

### Task 4: 创建中文测试用例
```python
# py/test_chinese.py
from helper import TextToSpeech, load_voice_style

def test_chinese_synthesis():
    # 加载模型
    tts = load_text_to_speech(onnx_dir='./assets/onnx', use_gpu=False)
    style = load_voice_style(['./assets/voice_styles/chinese_speaker_01.json'])
    
    # 测试文本
    test_texts = [
        "你好世界。",
        "这是一个中文文本转语音测试。",
        "支持中文标点符号：句号、逗号、感叹号！问号？",
        "长文本测试：" + "这是一个很长的句子。" * 20
    ]
    
    for text in test_texts:
        print(f"合成: {text[:50]}...")
        wav, duration = tts(text, lang='zh', style=style, total_step=8, speed=1.0)
        print(f"  时长: {duration:.2f}s, 音频长度: {len(wav[0])}")
        assert len(wav[0]) > 0, "音频为空"
        assert duration > 0, "时长为 0"
    
    print("✓ 所有中文测试通过")

if __name__ == '__main__':
    test_chinese_synthesis()
```

### Task 5: 更新 README
在主 README.md 中更新支持的语言列表，添加中文（zh）。

---

## 测试命令

```bash
# Python
cd py && python test_chinese.py

# Node.js
cd nodejs && node test_chinese.js

# Go
cd go && go run test_chinese.go

# 其他 SDK 类似
```

---

## 验证清单

- [ ] 所有 10 个 SDK 的 AVAILABLE_LANGS 包含 "zh"
- [ ] CJK 语言逻辑包含 "zh"（max_len=120）
- [ ] 中文测试用例通过
- [ ] 中文标点符号正确处理
- [ ] 长文本自动分句正常
- [ ] 不影响其他语言的推理
- [ ] README 更新

---

## 提交

```bash
git add py/helper.py nodejs/helper.js web/helper.js go/helper.go rust/src/ cpp/helper.cpp java/Helper.java csharp/Helper.cs swift/Sources/ flutter/lib/ README.md
git commit -m "feat(lang): add Chinese (zh) language support across all SDKs"
```

---

## 注意事项

1. **Unicode Indexer**：确保 `unicode_indexer_zh.json` 已放置在 `assets/onnx/` 目录
2. **Voice Style**：需要至少一个中文 voice style JSON 文件用于测试
3. **向后兼容**：添加中文不影响现有 31 种语言的功能
4. **文档更新**：更新所有 SDK 的 README，说明中文支持
