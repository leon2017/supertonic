# Cross-Platform SDK Patterns

## Interface Consistency

All SDKs expose the same public interface:

| Component | Purpose |
|-----------|---------|
| `UnicodeProcessor` | Text → token IDs + mask |
| `TextToSpeech` | Main inference class |
| `Style` | Voice style container (style_ttl + style_dp) |
| `load_text_to_speech(onnx_dir, use_gpu)` | Factory function |
| `load_voice_style(paths)` | Load voice style JSON(s) |

## Parameter Conventions

Synthesis call signature (adapt to language idioms):
```
text_to_speech(text, lang, style, total_step=8, speed=1.05)
```

- `text`: string, the input text
- `lang`: string, language code from AVAILABLE_LANGS
- `style`: Style object loaded from voice style JSON
- `total_step`: int, denoising iterations (more = higher quality, slower)
- `speed`: float, speech rate multiplier (>1 = faster, <1 = slower)

## Return Values

- `wav`: audio waveform array, shape (batch, samples), float32 in [-1, 1]
- `duration`: predicted duration in seconds, shape (batch,)

Caller is responsible for trimming wav to actual duration:
```
trimmed = wav[: int(sample_rate * duration)]
```

## Voice Style JSON Format

```json
{
  "style_ttl": {
    "dims": [1, D1, D2],
    "data": [...]
  },
  "style_dp": {
    "dims": [1, D3, D4],
    "data": [...]
  }
}
```

- Batch loading: stack multiple style files along batch dimension
- Dims[0] is always 1 (single speaker per file)

## Assets Management

- Models downloaded from HuggingFace on first run
- Default path: `../assets/onnx/` relative to SDK directory
- Required files: `duration_predictor.onnx`, `text_encoder.onnx`, `vector_estimator.onnx`, `vocoder.onnx`, `tts.json`, `unicode_indexer.json`
- Voice styles: `../assets/voice_styles/*.json`

## Error Handling

- Invalid language → raise/throw with message listing valid languages
- Missing ONNX file → clear error message with download instructions
- Empty text → return empty/zero-length audio (not an error)
- Text too long → auto-split via chunk_text (Python) or equivalent

## Batch Processing

- All SDKs support batch inference: multiple texts in one call
- Batch size = number of voice styles = number of texts = number of langs
- Padding handled internally (text_mask, latent_mask)
