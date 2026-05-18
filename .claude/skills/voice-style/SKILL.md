---
name: voice-style
description: Create, debug, or convert voice style JSON files for Supertonic TTS
version: 1.0.0
source: supertonic-tts
domain: voice
---

# Voice Style

Work with voice style files for Supertonic TTS.

## When to Use

- User wants to create a new voice style
- User is debugging voice quality issues
- User asks about style vector format
- User wants to convert or inspect voice style files

## Voice Style JSON Format

```json
{
  "style_ttl": {
    "dims": [1, D1, D2],
    "data": [0.123, -0.456, ...]
  },
  "style_dp": {
    "dims": [1, D3, D4],
    "data": [0.789, -0.012, ...]
  }
}
```

- `style_ttl`: controls timbre and prosody (used by TextEncoder + VectorEstimator)
- `style_dp`: controls speaking rate/rhythm (used by DurationPredictor)
- `dims[0]` is always 1 (single speaker per file)
- Data is flattened row-major float32 values

## Available Voices

Located in `assets/voice_styles/`:
- M1.json, M3.json, M4.json, M5.json — Male voices
- F1.json, F3.json, F4.json, F5.json — Female voices

## Inspecting a Voice Style

```python
import json
import numpy as np

with open("assets/voice_styles/M1.json") as f:
    style = json.load(f)

ttl = np.array(style["style_ttl"]["data"]).reshape(style["style_ttl"]["dims"])
dp = np.array(style["style_dp"]["data"]).reshape(style["style_dp"]["dims"])

print(f"style_ttl shape: {ttl.shape}, range: [{ttl.min():.3f}, {ttl.max():.3f}]")
print(f"style_dp shape: {dp.shape}, range: [{dp.min():.3f}, {dp.max():.3f}]")
```

## Creating Custom Voices

Custom voices are created via the Voice Builder service:
https://supertonic.supertone.ai/voice_builder

The output is a JSON file in the same format as the built-in voices.

## Debugging Voice Issues

- **Robotic/distorted output**: Check style vector values are in reasonable range (typically [-3, 3])
- **Wrong gender/timbre**: Verify correct style file is loaded
- **Batch mismatch**: Number of style files must equal number of texts in batch mode
- **NaN in style**: Corrupted file — re-download or re-generate
