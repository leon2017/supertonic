---
name: sdk-example
description: Write or modify SDK example code for a specific platform (Python, Node.js, Go, Rust, etc.)
version: 1.0.0
source: supertonic-tts
domain: sdk
---

# SDK Example

Write or modify SDK example code for Supertonic TTS.

## When to Use

- User wants to add a new platform SDK
- User wants to modify an existing SDK example
- User asks about SDK implementation patterns
- User wants to add a new feature to all SDKs

## SDK Structure

Each SDK directory follows this pattern:

```
{lang}/
├── helper.{ext}        # Core logic: UnicodeProcessor, TextToSpeech, utilities
├── example_onnx.{ext}  # CLI example demonstrating usage
├── assets/             # Symlink or copy of ONNX models (gitignored)
├── README.md           # Language-specific usage instructions
└── {build_file}        # Package manager config (requirements.txt, package.json, etc.)
```

## Core Components to Implement

### 1. UnicodeProcessor

- Load `unicode_indexer.json`
- Preprocess text (normalize, clean, wrap with lang tags)
- Convert text to token ID array
- Generate text mask from lengths

### 2. TextToSpeech

- Load 4 ONNX models + config
- Implement inference pipeline: DP → TextEnc → VectorEst (loop) → Vocoder
- Support single and batch inference
- Handle long text via chunking + concatenation

### 3. Style Loading

- Parse voice style JSON
- Extract and reshape style_ttl and style_dp arrays
- Support batch loading (multiple styles)

### 4. Utilities

- `length_to_mask`: convert lengths to binary mask
- `chunk_text`: split long text by sentences
- Timer/profiling helper
- Filename sanitization

## Platform-Specific Notes

| Platform | ONNX Runtime Package | Audio Output |
|----------|---------------------|--------------|
| Python | `onnxruntime` | soundfile (WAV) |
| Node.js | `onnxruntime-node` | wav-encoder |
| Web | `onnxruntime-web` (WebGPU) | AudioContext |
| Go | `onnxruntime_go` | go-audio/wav |
| Rust | `ort` crate | hound (WAV) |
| C++ | ONNX Runtime C API | raw WAV write |
| Java | `ai.onnxruntime` | javax.sound |
| C# | `Microsoft.ML.OnnxRuntime` | NAudio or raw |
| Swift | `onnxruntime-swift` | AVFoundation |
| Flutter | `onnxruntime` (via FFI) | audioplayers |

## Adding a New SDK

1. Create directory: `{lang}/`
2. Implement `helper.{ext}` with all core components
3. Implement `example_onnx.{ext}` as CLI demo
4. Add build/dependency file
5. Write `README.md` with setup and usage instructions
6. Add to `test_all.sh`
7. Update root README.md SDK list
