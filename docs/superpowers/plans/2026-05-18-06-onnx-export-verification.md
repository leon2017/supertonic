# ONNX 导出与验证实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将训练好的 PyTorch 模型导出为 ONNX 格式，并验证精度

**Architecture:** 4 个独立 ONNX 模型 + 精度验证脚本

**Tech Stack:** PyTorch 2.2+, ONNX, onnxruntime

---

## File Structure

```
train/
├── export_onnx.py           # Main export script
├── verify_onnx.py           # Verification script
└── configs/
    └── export.yaml          # Export config
```

---

## 关键任务概要

### Task 1: 实现 Duration Predictor 导出
```python
def export_duration_predictor(model, output_path):
    model.eval()
    
    # Dummy inputs
    text_ids = torch.randint(0, 1000, (1, 100), dtype=torch.int64)
    style_dp = torch.randn(1, 64, 1)
    text_mask = torch.ones(1, 1, 100)
    
    # Export
    torch.onnx.export(
        model,
        (text_ids, style_dp, text_mask),
        output_path,
        input_names=['text_ids', 'style_dp', 'text_mask'],
        output_names=['duration'],
        dynamic_axes={
            'text_ids': {0: 'batch', 1: 'seq_len'},
            'text_mask': {0: 'batch', 2: 'seq_len'},
            'duration': {0: 'batch'}
        },
        opset_version=17
    )
```

### Task 2: 实现 Text Encoder 导出
```python
def export_text_encoder(model, output_path):
    model.eval()
    
    text_ids = torch.randint(0, 1000, (1, 100), dtype=torch.int64)
    style_ttl = torch.randn(1, 512, 50)
    text_mask = torch.ones(1, 1, 100)
    
    torch.onnx.export(
        model,
        (text_ids, style_ttl, text_mask),
        output_path,
        input_names=['text_ids', 'style_ttl', 'text_mask'],
        output_names=['text_emb'],
        dynamic_axes={
            'text_ids': {0: 'batch', 1: 'seq_len'},
            'text_mask': {0: 'batch', 2: 'seq_len'},
            'text_emb': {0: 'batch', 2: 'seq_len'}
        },
        opset_version=17
    )
```

### Task 3: 实现 VF Estimator 导出
```python
def export_vf_estimator(model, output_path):
    model.eval()
    
    noisy_latent = torch.randn(1, 144, 50)
    text_emb = torch.randn(1, 512, 100)
    style_ttl = torch.randn(1, 512, 50)
    text_mask = torch.ones(1, 1, 100)
    latent_mask = torch.ones(1, 1, 50)
    current_step = torch.tensor([0], dtype=torch.int64)
    total_step = torch.tensor([32], dtype=torch.int64)
    
    torch.onnx.export(
        model,
        (noisy_latent, text_emb, style_ttl, text_mask, latent_mask, current_step, total_step),
        output_path,
        input_names=['noisy_latent', 'text_emb', 'style_ttl', 'text_mask', 'latent_mask', 'current_step', 'total_step'],
        output_names=['xt'],
        dynamic_axes={
            'noisy_latent': {0: 'batch', 2: 'latent_len'},
            'text_emb': {0: 'batch', 2: 'text_len'},
            'text_mask': {0: 'batch', 2: 'text_len'},
            'latent_mask': {0: 'batch', 2: 'latent_len'},
            'xt': {0: 'batch', 2: 'latent_len'}
        },
        opset_version=17
    )
```

### Task 4: 实现 Vocoder 导出
```python
def export_vocoder(model, output_path):
    model.eval()
    
    latent = torch.randn(1, 24, 100)
    
    torch.onnx.export(
        model,
        latent,
        output_path,
        input_names=['latent'],
        output_names=['wav'],
        dynamic_axes={
            'latent': {0: 'batch', 2: 'latent_len'},
            'wav': {0: 'batch', 1: 'audio_len'}
        },
        opset_version=17
    )
```

### Task 5: 实现精度验证
```python
def verify_onnx_accuracy(pytorch_model, onnx_path, test_inputs):
    """验证 ONNX 与 PyTorch 输出误差 < 1e-5"""
    import onnxruntime as ort
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(*test_inputs)
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_inputs = {name: inp.numpy() for name, inp in zip(session.get_inputs(), test_inputs)}
    onnx_output = session.run(None, onnx_inputs)[0]
    
    # Compare
    diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-5, f"Accuracy check failed: {diff} >= 1e-5"
    print("✓ Accuracy check passed")
```

---

## 导出命令

```bash
python train/export_onnx.py \
    --duration_checkpoint checkpoints/duration_best.pth \
    --text_encoder_checkpoint checkpoints/text_to_latent_best.pth \
    --vf_estimator_checkpoint checkpoints/text_to_latent_best.pth \
    --vocoder_checkpoint checkpoints/autoencoder_best.pth \
    --output_dir ./assets/onnx/
```

---

## 验证清单

- [ ] 4 个 ONNX 文件成功导出
- [ ] 每个模型精度验证通过（误差 < 1e-5）
- [ ] ONNX 模型可以用 onnxruntime 加载
- [ ] Dynamic axes 正确设置
- [ ] 输入输出名称与 SDK 一致

---

## 输出文件

```
assets/onnx/
├── duration_predictor.onnx
├── text_encoder.onnx
├── vector_estimator.onnx
├── vocoder.onnx
├── tts.json                    # 配置文件
└── unicode_indexer_zh.json     # Unicode 映射
```
