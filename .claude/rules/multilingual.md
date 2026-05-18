# Multilingual Text Processing Rules

## Supported Languages

31 languages + language-agnostic mode (`na`):

```
ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu,
id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv,
tr, uk, vi, na
```

The `AVAILABLE_LANGS` list must be identical across ALL SDKs.

## Language Tag Format

Text is wrapped with XML-style language tags before tokenization:

```
<en>Hello world.</en>
<ko>안녕하세요.</ko>
<ja>こんにちは.</ja>
```

- Tags are part of the token sequence (included in unicode indexer)
- `na` (language-agnostic) mode: `<na>text</na>` — model handles mixed/unknown language

## Text Preprocessing Pipeline

1. NFKD Unicode normalization
2. Remove emojis
3. Replace dashes/symbols with ASCII equivalents
4. Remove special symbols (♥, ☆, ©, etc.)
5. Replace expressions (@→"at", e.g.→"for example")
6. Fix spacing around punctuation
7. Remove duplicate quotes
8. Collapse whitespace
9. Append period if no terminal punctuation
10. Wrap with language tags

## Adding a New Language

When adding language `xx`:

1. Add `"xx"` to `AVAILABLE_LANGS` in ALL SDK helper files:
   - `py/helper.py`
   - `nodejs/helper.js`
   - `web/helper.js`
   - `go/helper.go`
   - `rust/src/*.rs`
   - `cpp/helper.cpp`
   - `java/Helper.java`
   - `csharp/Helper.cs`
   - `swift/Sources/**/*.swift`
   - `flutter/lib/**/*.dart`

2. Verify `unicode_indexer.json` covers the language's character set

3. Add language-specific preprocessing if needed (e.g., CJK sentence splitting)

4. Update README language list

5. Add test text samples for the new language

## Sentence Splitting

- Default max_len: 300 characters (Latin scripts)
- CJK languages (ko, ja): max_len = 120 (characters are denser)
- Split by paragraph first (double newline), then by sentence boundaries
- Respect abbreviations (Mr., Dr., etc.) — don't split mid-abbreviation
- Each chunk is synthesized independently, joined with silence gap (default 0.3s)

## Unicode Indexer

- `unicode_indexer.json` maps Unicode code points to model token IDs
- Characters not in the indexer will cause KeyError — must validate before inference
- Language tags (`<en>`, `</en>`) are also in the indexer as multi-character tokens mapped per-character
