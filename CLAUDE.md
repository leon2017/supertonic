# Supertonic — 离线多语言 TTS SDK

## 项目概述

Supertonic 是一个轻量级、高性能的离线多语言文本转语音系统。基于 ONNX Runtime 推理，支持 31 种语言，可在桌面、移动端、浏览器和边缘设备上本地运行，无需网络连接。

核心模型为 99M 参数，输出 44.1kHz 16-bit WAV 音频。

## 架构

推理管线由 4 个 ONNX 模型串联组成：

```
Text → UnicodeProcessor → [DurationPredictor] → duration
                        → [TextEncoder] → text_emb
     → [VectorEstimator] × N steps (flow matching denoising) → latent
     → [Vocoder] → wav (44.1kHz)
```

- **DurationPredictor**: 预测语音时长，输入 text_ids + style_dp
- **TextEncoder**: 编码文本为 embedding，输入 text_ids + style_ttl
- **VectorEstimator**: 迭代去噪（flow matching），输入 noisy_latent + text_emb + style_ttl
- **Vocoder**: 将 latent 解码为波形

## 技术栈

- 推理引擎：ONNX Runtime（跨平台）
- 文本处理：Unicode 索引映射（无 phonemizer 依赖）
- 音频输出：44.1kHz 16-bit PCM WAV
- Voice Style：JSON 格式的 style vector（style_ttl + style_dp）

## 目录结构

```
supertonic/
├── py/          # Python SDK 示例
├── nodejs/      # Node.js SDK 示例
├── web/         # 浏览器 WebGPU 示例
├── java/        # Java SDK 示例
├── cpp/         # C++ SDK 示例
├── csharp/      # C# SDK 示例
├── go/          # Go SDK 示例
├── swift/       # Swift SDK 示例
├── ios/         # iOS App 示例
├── rust/        # Rust SDK 示例
├── flutter/     # Flutter SDK 示例
├── img/         # README 图片资源
└── test_all.sh  # 全平台测试脚本
```

每个 SDK 目录结构一致：
- `helper.*` — 核心推理逻辑（UnicodeProcessor, TextToSpeech）
- `example_onnx.*` — 使用示例
- `assets/` — ONNX 模型和配置文件（gitignore，首次运行自动下载）
- `README.md` — 该语言的使用说明

## 开发命令

```bash
# Python
cd py && pip install -r requirements.txt && python example_onnx.py

# Node.js
cd nodejs && npm install && node example_onnx.js

# Web (Vite dev server)
cd web && npm install && npm run dev

# Go
cd go && go run .

# Rust
cd rust && cargo run --release

# C++
cd cpp && mkdir build && cd build && cmake .. && make && ./example_onnx

# Java
cd java && mvn compile exec:java

# C#
cd csharp && dotnet run

# 全平台测试
bash test_all.sh
```

## 核心概念

### Unicode 处理

文本通过 `unicode_indexer.json` 映射为整数 ID 序列，不依赖外部 phonemizer。文本预处理包括：
- NFKD 规范化
- Emoji 移除
- 标点/符号替换
- 语言标签包裹：`<lang>text</lang>`

### Voice Style

JSON 文件包含两个 style vector：
- `style_ttl`: 控制音色和韵律（用于 TextEncoder 和 VectorEstimator）
- `style_dp`: 控制语速节奏（用于 DurationPredictor）

### Flow Matching 去噪

VectorEstimator 通过 N 步迭代（默认 8 步）将高斯噪声 latent 去噪为语音 latent，类似 diffusion 但使用 ODE 直线路径。

### 长文本分句

超过 max_len 的文本自动按段落和句子边界切分，逐段合成后拼接（中间插入静音）。

## 协作规范

- 所有 SDK 的公共接口保持一致：`TextToSpeech(text, lang, style, total_step, speed)`
- 新增语言时需同步更新所有 SDK 的 `AVAILABLE_LANGS` 列表
- ONNX 模型文件不提交到 git，通过 HuggingFace 分发
- 每个 SDK 的 README 保持统一格式

## 执行原则

- 不接受"最小修复/兜底方案"作为目标，默认提供可执行、可复现、可验证的完整方案。
- 必须遵循第一性原理：先定义问题本质（目标、约束、机制），再设计干预项，不做拍脑袋调参。
- 必须用数据说话：每次优化都要有明确指标、对照组、实验配置和结果记录。
