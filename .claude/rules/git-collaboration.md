# Git & Collaboration Rules

## Branch Strategy

```
main              — stable, released code
feature/*         — new features (e.g., feature/support_chinese)
fix/*             — bug fixes
docs/*            — documentation updates
sdk/*             — SDK-specific changes (e.g., sdk/flutter-improvements)
```

## Commit Message Format

```
<type>(<scope>): <description>
```

Types: feat, fix, refactor, docs, chore, test, perf

Scopes: sdk, lang, model, py, nodejs, web, go, rust, cpp, java, csharp, swift, ios, flutter, ci

Examples:
```
feat(lang): add Chinese (zh) language support
fix(py): handle empty text input in chunk_text
docs(sdk): update Quick Start with soundfile alternative
perf(nodejs): optimize ONNX session options for batch inference
feat(flutter): add macOS platform support
```

## PR Requirements

- Description: what changed, why, how to test
- For new language: include sample audio output comparison
- For SDK changes: verify affected platform runs correctly
- For text processing: include edge case examples

## Large Files

- ONNX models: never commit (distributed via HuggingFace)
- Voice style JSON: OK to commit (small)
- Audio samples: never commit to repo
- Assets directory: gitignored, auto-downloaded at runtime
