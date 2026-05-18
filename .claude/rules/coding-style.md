# Coding Style

## Python (primary SDK)

- Formatter: `ruff format` (Black-compatible)
- Linter: `ruff check`
- Line length: 120
- Imports: sorted by `ruff` (isort-compatible)
- Type annotations required for public functions
- Docstrings: only for non-obvious behavior

```python
# Standard library
import json
import os
from pathlib import Path

# Third party
import numpy as np
import onnxruntime as ort

# Local
from helper import TextToSpeech, UnicodeProcessor
```

## JavaScript / TypeScript (Node.js, Web)

- ESM modules (`import/export`)
- No semicolons (follow existing style)
- `const` by default, `let` when reassignment needed
- JSDoc for public functions

## Go

- Standard `gofmt` formatting
- Error handling: always check and return errors
- Naming: exported = PascalCase, unexported = camelCase

## Rust

- `cargo fmt` + `cargo clippy`
- Use `Result<T, E>` for fallible operations
- Avoid `unwrap()` in library code

## C++

- C++17 standard
- CMake build system
- RAII for resource management
- `snake_case` for functions, `PascalCase` for classes

## Java

- Maven build
- PascalCase classes, camelCase methods
- Close resources with try-with-resources

## C#

- .NET SDK / `dotnet` CLI
- PascalCase for public members
- `using` statements for disposables

## Swift

- Swift Package Manager
- camelCase functions, PascalCase types

## Dart / Flutter

- `dart format`
- Follow Flutter style guide
- camelCase for variables/functions, PascalCase for classes

## Cross-SDK Consistency

- All SDKs use the same class/function names where language allows:
  - `UnicodeProcessor` / `unicode_processor`
  - `TextToSpeech` / `text_to_speech`
  - `loadVoiceStyle` / `load_voice_style`
- Parameter order is consistent: `(text, lang, style, total_step, speed)`
- Error messages use the same wording across SDKs
