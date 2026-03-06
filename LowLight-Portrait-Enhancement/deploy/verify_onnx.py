#!/usr/bin/env python3
"""
详细的 ONNX 模型验证工具

功能：
1. 对比 PyTorch vs ONNX 输出一致性
2. 性能测试（推理速度对比）
3. 可视化对比结果
4. 生成详细的验证报告

Usage:
    python deploy/verify_onnx.py --onnx deploy/models/retinexformer.onnx --image data/test.png
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RetinexFormerEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description='Verify ONNX model')
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNX model path')
    parser.add_argument('--weights', type=str, default=None,
                        help='PyTorch weights path (default: models/LOL_v2_synthetic.pth)')
    parser.add_argument('--image', type=str, default=None,
                        help='Test image path (optional, for visual comparison)')
    parser.add_argument('--output-dir', type=str, default='deploy/verification',
                        help='Output directory for reports')
    parser.add_argument('--num-tests', type=int, default=10,
                        help='Number of random inputs for testing (default: 10)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    return parser.parse_args()


def verify_numerical_consistency(pytorch_model, onnx_path, input_size=(512, 512), num_tests=10):
    """验证数值一致性"""
    try:
        import onnxruntime as ort
    except ImportError:
        print('Error: onnxruntime not installed. Run: pip install onnxruntime')
        sys.exit(1)

    print(f'\n{"="*60}')
    print('数值一致性验证')
    print(f'{"="*60}')

    # 创建 ONNX Runtime session
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    pytorch_model.eval()
    pytorch_model.to('cpu')

    results = []

    for i in range(num_tests):
        # 生成随机输入
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).cpu().numpy()

        # ONNX 推理
        onnx_output = sess.run(None, {'input': dummy_input.numpy()})[0]

        # 计算差异
        diff = np.abs(pytorch_output - onnx_output)
        mean_diff = diff.mean()
        max_diff = diff.max()

        results.append({
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'passed': mean_diff < 1e-4
        })

        print(f'Test {i+1}/{num_tests}: Mean diff = {mean_diff:.6f}, Max diff = {max_diff:.6f}')

    # 统计结果
    passed = sum(1 for r in results if r['passed'])
    avg_mean_diff = np.mean([r['mean_diff'] for r in results])
    avg_max_diff = np.mean([r['max_diff'] for r in results])

    print(f'\n{"="*60}')
    print(f'通过率: {passed}/{num_tests} ({passed/num_tests*100:.1f}%)')
    print(f'平均 Mean Diff: {avg_mean_diff:.6f}')
    print(f'平均 Max Diff: {avg_max_diff:.6f}')
    print(f'{"="*60}')

    return results, passed == num_tests


def benchmark_performance(pytorch_model, onnx_path, input_size=(512, 512), num_runs=50):
    """性能基准测试"""
    try:
        import onnxruntime as ort
    except ImportError:
        print('Error: onnxruntime not installed')
        sys.exit(1)

    print(f'\n{"="*60}')
    print('性能基准测试')
    print(f'{"="*60}')

    # 创建 ONNX Runtime session
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    pytorch_model.eval()
    pytorch_model.to('cpu')

    # 准备输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Warmup
    print('Warming up...')
    for _ in range(5):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
        _ = sess.run(None, {'input': dummy_input.numpy()})

    # PyTorch 性能测试
    print(f'\nPyTorch 推理测试 ({num_runs} 次)...')
    pytorch_times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
        pytorch_times.append(time.time() - start)

    # ONNX 性能测试
    print(f'ONNX Runtime 推理测试 ({num_runs} 次)...')
    onnx_times = []
    for _ in range(num_runs):
        start = time.time()
        _ = sess.run(None, {'input': dummy_input.numpy()})
        onnx_times.append(time.time() - start)

    # 统计结果
    pytorch_avg = np.mean(pytorch_times) * 1000  # ms
    pytorch_std = np.std(pytorch_times) * 1000
    onnx_avg = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000

    print(f'\n{"="*60}')
    print(f'PyTorch:      {pytorch_avg:.2f} ± {pytorch_std:.2f} ms')
    print(f'ONNX Runtime: {onnx_avg:.2f} ± {onnx_std:.2f} ms')
    print(f'加速比:       {pytorch_avg/onnx_avg:.2f}x')
    print(f'{"="*60}')

    return {
        'pytorch': {'mean': pytorch_avg, 'std': pytorch_std, 'times': pytorch_times},
        'onnx': {'mean': onnx_avg, 'std': onnx_std, 'times': onnx_times}
    }


def visual_comparison(pytorch_model, onnx_path, image_path, output_dir):
    """可视化对比"""
    try:
        import onnxruntime as ort
    except ImportError:
        print('Error: onnxruntime not installed')
        return

    print(f'\n{"="*60}')
    print('可视化对比')
    print(f'{"="*60}')

    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f'Error: Failed to load image: {image_path}')
        return

    print(f'Image size: {image.shape[1]}x{image.shape[0]}')

    # 预处理
    img_rgb = image[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # PyTorch 推理
    print('PyTorch 推理...')
    pytorch_model.eval()
    pytorch_model.to('cpu')
    with torch.no_grad():
        pytorch_output = pytorch_model(img_tensor)
    pytorch_output = torch.clamp(pytorch_output, 0, 1)
    pytorch_result = pytorch_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pytorch_result = (pytorch_result * 255).astype(np.uint8)

    # ONNX 推理
    print('ONNX Runtime 推理...')
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_output = sess.run(None, {'input': img_tensor.numpy()})[0]
    onnx_output = np.clip(onnx_output, 0, 1)
    onnx_result = onnx_output.squeeze(0).transpose(1, 2, 0)
    onnx_result = (onnx_result * 255).astype(np.uint8)

    # 计算差异
    diff = np.abs(pytorch_result.astype(float) - onnx_result.astype(float))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / 'input.png'), image)
    cv2.imwrite(str(output_dir / 'pytorch_output.png'), pytorch_result[:, :, ::-1])
    cv2.imwrite(str(output_dir / 'onnx_output.png'), onnx_result[:, :, ::-1])
    cv2.imwrite(str(output_dir / 'diff.png'), diff_normalized)

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pytorch_result)
    axes[0, 1].set_title('PyTorch Output')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(onnx_result)
    axes[1, 0].set_title('ONNX Output')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(diff_normalized)
    axes[1, 1].set_title(f'Difference (Mean: {diff.mean():.2f})')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'comparison.png'), dpi=150, bbox_inches='tight')
    print(f'\n对比图已保存到: {output_dir / "comparison.png"}')

    # 统计差异
    print(f'\n差异统计:')
    print(f'  Mean: {diff.mean():.4f}')
    print(f'  Max:  {diff.max():.4f}')
    print(f'  Std:  {diff.std():.4f}')


def main():
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = project_root / onnx_path

    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = project_root / 'models' / 'LOL_v2_synthetic.pth'

    # Check files exist
    if not onnx_path.exists():
        print(f'Error: ONNX model not found: {onnx_path}')
        sys.exit(1)
    if not weights_path.exists():
        print(f'Error: Weights not found: {weights_path}')
        sys.exit(1)

    print(f'ONNX Model: {onnx_path}')
    print(f'PyTorch Weights: {weights_path}')

    # Load PyTorch model
    print('\nLoading PyTorch model...')
    enhancer = RetinexFormerEnhancer(str(weights_path), device='cpu')
    model = enhancer.get_model()

    # 1. 数值一致性验证
    results, passed = verify_numerical_consistency(model, onnx_path, num_tests=args.num_tests)

    # 2. 性能基准测试
    if args.benchmark:
        perf_results = benchmark_performance(model, onnx_path)

    # 3. 可视化对比
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        if image_path.exists():
            visual_comparison(model, onnx_path, image_path, args.output_dir)
        else:
            print(f'Warning: Image not found: {image_path}')

    # 总结
    print(f'\n{"="*60}')
    print('验证完成')
    print(f'{"="*60}')
    if passed:
        print('✓ 数值一致性验证通过')
    else:
        print('✗ 数值一致性验证失败')


if __name__ == '__main__':
    main()
