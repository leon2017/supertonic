import argparse
import torch
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("请安装 onnxruntime: pip install onnxruntime")
    exit(1)


def verify_model(
    onnx_path: str,
    test_inputs: dict[str, np.ndarray],
    pytorch_output: np.ndarray,
    tolerance: float = 1e-4,
) -> bool:
    print(f"\n验证: {onnx_path}")

    session = ort.InferenceSession(onnx_path)

    onnx_output = session.run(None, test_inputs)[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    print(f"  最大误差: {max_diff:.8f}")
    print(f"  平均误差: {mean_diff:.8f}")
    print(f"  容差: {tolerance}")

    if max_diff < tolerance:
        print(f"  PASS 验证通过")
        return True
    else:
        print(f"  FAIL 验证失败 (误差超过容差)")
        return False


def verify_vocoder(onnx_path: str):
    from models.autoencoder import LatentDecoder

    decoder = LatentDecoder(latent_dim=24, channels=256, intermediate_channels=1024, num_layers=3)
    decoder.eval()

    latent = torch.randn(1, 24, 50)

    with torch.no_grad():
        pytorch_out = decoder(latent).numpy()

    onnx_inputs = {'latent': latent.numpy()}
    return verify_model(onnx_path, onnx_inputs, pytorch_out)


def verify_duration_predictor(onnx_path: str):
    from models.duration_predictor import DurationPredictor

    dp = DurationPredictor(vocab_size=1000)
    dp.eval()

    text_ids = torch.randint(0, 1000, (1, 30), dtype=torch.long)
    mel = torch.randn(1, 228, 50)
    text_mask = torch.ones(1, 1, 30)

    with torch.no_grad():
        pytorch_out = dp(text_ids, mel, text_mask).numpy()

    onnx_inputs = {
        'text_ids': text_ids.numpy(),
        'style_dp': mel.numpy(),
        'text_mask': text_mask.numpy()
    }
    return verify_model(onnx_path, onnx_inputs, pytorch_out)


def main():
    parser = argparse.ArgumentParser(description='验证 ONNX 模型精度')
    parser.add_argument('--onnx_dir', type=str, required=True, help='ONNX 模型目录')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='误差容差')
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)

    results = {}

    models = ['vocoder.onnx', 'text_encoder.onnx', 'vector_estimator.onnx', 'duration_predictor.onnx']

    for model_name in models:
        model_path = onnx_dir / model_name
        if model_path.exists():
            print(f"\n检查 {model_name}...")
            session = ort.InferenceSession(str(model_path))
            print(f"  输入: {[inp.name for inp in session.get_inputs()]}")
            print(f"  输出: {[out.name for out in session.get_outputs()]}")
            results[model_name] = True
        else:
            print(f"\n跳过 {model_name} (文件不存在)")
            results[model_name] = None

    print("\n\n=== 验证结果 ===")
    for name, result in results.items():
        if result is True:
            print(f"  PASS {name}: 可加载")
        elif result is False:
            print(f"  FAIL {name}: 验证失败")
        else:
            print(f"  SKIP {name}: 跳过")


if __name__ == '__main__':
    main()
