---
name: text-processing
description: Implement or debug text preprocessing, normalization, and sentence splitting for TTS
version: 1.0.0
source: supertonic-tts
domain: nlp
---

# Text Processing

Work with text preprocessing and normalization for Supertonic TTS.

## When to Use

- User is adding language-specific text normalization
- User is debugging text preprocessing issues
- User wants to improve sentence splitting
- User asks about Unicode handling or tokenization

## Preprocessing Pipeline

```python
def _preprocess_text(text: str, lang: str) -> str:
    # 1. NFKD normalization
    text = normalize("NFKD", text)
    
    # 2. Remove emojis
    text = emoji_pattern.sub("", text)
    
    # 3. Replace dashes/symbols → ASCII
    # 4. Remove special symbols
    # 5. Replace expressions (@→"at", etc.)
    # 6. Fix punctuation spacing
    # 7. Remove duplicate quotes
    # 8. Collapse whitespace
    # 9. Add terminal punctuation if missing
    # 10. Wrap: <lang>text</lang>
    
    return text
```

## Sentence Splitting (chunk_text)

Purpose: split long text into chunks that fit the model's context window.

```python
def chunk_text(text: str, max_len: int = 300) -> list[str]:
    # 1. Split by paragraph (double newline)
    # 2. For each paragraph, split by sentence boundaries
    # 3. Accumulate sentences until max_len reached
    # 4. Respect abbreviations (Mr., Dr., etc.)
```

Key parameters:
- Latin scripts: `max_len = 300`
- CJK (ko, ja, zh): `max_len = 120` (characters carry more information)

## Adding Language-Specific Rules

### Chinese (zh) — Example

```python
# Chinese-specific preprocessing:
# 1. No NFKD needed (already in NFC)
# 2. Chinese punctuation → keep as-is (。！？，)
# 3. Sentence splitting on Chinese punctuation: 。！？；
# 4. max_len = 120

# Add to terminal punctuation regex:
r"[.!?;:,'\"')\]}…。！？；」』】〉》›»]$"
```

### Arabic (ar) — Example

```python
# Arabic-specific:
# 1. RTL markers handled by Unicode normalization
# 2. Arabic punctuation: ، ؛ ؟
# 3. Sentence splitting on: . ؟ !
```

## Unicode Indexer Validation

Before adding a language, verify character coverage:

```python
import json

with open("assets/onnx/unicode_indexer.json") as f:
    indexer = json.load(f)

# Test with representative text
test_texts = {
    "zh": "今天天气真好，我想出去散步。这是一个测试句子。",
    "ar": "مرحبا بالعالم. هذا اختبار.",
}

for lang, text in test_texts.items():
    missing = []
    for c in text:
        if str(ord(c)) not in indexer:
            missing.append(f"'{c}' (U+{ord(c):04X})")
    if missing:
        print(f"{lang}: Missing {len(missing)} chars: {missing[:10]}")
    else:
        print(f"{lang}: All characters covered")
```

## Expression Tags

Supertonic supports inline expression tags for natural speech:

```
<laugh>Ha ha</laugh>
<breath>...</breath>
<sigh>...</sigh>
```

10 supported tags — these pass through preprocessing unchanged and are handled by the model directly.

## Debugging Tips

- **Garbled output**: Check NFKD normalization isn't destroying needed characters
- **Missing words**: Check if characters are being stripped by symbol removal
- **Wrong prosody**: Check sentence splitting — bad splits cause unnatural pauses
- **KeyError in indexer**: Character not supported — add to removal/replacement list
- **Empty output after preprocessing**: All characters were stripped — check regex patterns
