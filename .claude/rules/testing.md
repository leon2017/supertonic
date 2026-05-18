# Testing Rules

## Python SDK

```bash
cd py && python example_onnx.py --text "Hello world" --lang en --n-test 1
```

Verify:
- Output WAV file exists and is non-empty
- Audio duration matches expected range (text length dependent)
- No runtime errors or warnings

## Node.js SDK

```bash
cd nodejs && node example_onnx.js
```

## Web SDK

```bash
cd web && npm run dev
# Open browser, verify synthesis works
```

## Go SDK

```bash
cd go && go run . 
```

## Full Platform Test

```bash
bash test_all.sh
```

## What to Test When Adding a Language

1. Text preprocessing handles the language's Unicode range correctly
2. Language tag wrapping: `<lang>text</lang>` produces valid token IDs
3. All SDKs updated with new lang in `AVAILABLE_LANGS`
4. Sample text synthesizes without error
5. Output audio sounds reasonable (manual listen)

## What to Test When Modifying Text Processing

1. Existing languages still produce identical output (regression)
2. Edge cases: empty string, single character, max length text
3. Punctuation handling for the target language
4. Sentence splitting works correctly for the language's sentence boundaries
5. Special characters don't crash the Unicode indexer

## What to Test When Modifying ONNX Inference

1. Output shape matches expected: `wav` is `(B, T)`, `duration` is `(B,)`
2. Batch inference produces same result as single inference per item
3. Speed parameter affects duration proportionally
4. No NaN/Inf in output arrays

## Audio Output Validation

- Sample rate: 44100 Hz
- Bit depth: 16-bit
- Channels: mono
- Duration: should be proportional to text length and inversely proportional to speed
- No clipping (values within [-1, 1] before int16 conversion)
