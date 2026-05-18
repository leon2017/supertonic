---
name: add-language
description: Add a new language to Supertonic TTS — update AVAILABLE_LANGS across all SDKs, add preprocessing, test
version: 1.0.0
source: supertonic-tts
domain: multilingual
---

# Add Language

Add a new language to Supertonic TTS.

## When to Use

- User wants to add a new language (e.g., Chinese `zh`)
- User asks about language support requirements
- User is debugging language-specific synthesis issues

## Steps

### 1. Verify Unicode Coverage

Check that `unicode_indexer.json` contains all characters needed for the target language:

```python
import json

with open("assets/onnx/unicode_indexer.json") as f:
    indexer = json.load(f)

test_text = "你好世界"  # sample text in target language
missing = [c for c in test_text if str(ord(c)) not in indexer]
print(f"Missing characters: {missing}")
```

### 2. Update AVAILABLE_LANGS in All SDKs

Files to update (add the new lang code in alphabetical position):

- `py/helper.py` — Python list
- `nodejs/helper.js` — JS array
- `web/helper.js` — JS array
- `go/helper.go` — Go slice
- `rust/src/*.rs` — Rust array/vec
- `cpp/helper.cpp` — C++ vector
- `java/Helper.java` — Java array
- `csharp/Helper.cs` — C# array
- `swift/Sources/**/*.swift` — Swift array
- `flutter/lib/**/*.dart` — Dart list

### 3. Add Language-Specific Preprocessing (if needed)

Some languages need special handling in `_preprocess_text`:

- **CJK (zh, ja, ko)**: shorter max_len for chunk_text (120 vs 300)
- **Arabic (ar)**: RTL handling already supported
- **Languages with special punctuation**: add to terminal punctuation regex

### 4. Add Test Text

Create sample text for the new language to verify synthesis works:

```python
test_cases = {
    "zh": "今天天气真好，我想出去散步。",
    # Short, medium, and long samples
}
```

### 5. Update Documentation

- Update README.md language list
- Add language code to the supported languages badge
- Add example in the Quick Start section if it's a major language

## Validation Checklist

- [ ] All SDKs have the new lang in AVAILABLE_LANGS
- [ ] Unicode indexer covers the language's character set
- [ ] Text preprocessing handles the language correctly
- [ ] Sentence splitting works for the language's punctuation
- [ ] Sample synthesis produces audible, reasonable output
- [ ] README updated
