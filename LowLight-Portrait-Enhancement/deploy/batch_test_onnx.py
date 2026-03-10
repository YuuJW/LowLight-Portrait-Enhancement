#!/usr/bin/env python3
"""
批量测试 ONNX 模型在多张图像上的表现

功能：
1. 在多张图像上测试 PyTorch vs ONNX 一致性
2. 统计平均差异和最大差异
3. 生成测试报告

Usage:
    python deploy/batch_test_onnx.py --onnx deploy/models/retinexformer.onnx --image-dir data/LOL/eval15/low
"""

import argparse
import sys
import time
from pathlib import Path
import json

import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_paths import resolve_onnx_path, resolve_project_path, resolve_weights_path
from models import RetinexFormerEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description='Batch test ONNX model')
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNX model path')
    parser.add_argument('--weights', type=str, default=None,
                        help='PyTorch weights path')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output', type=str, default='deploy/batch_test_report.json',
                        help='Output report path')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to test')
    return parser.parse_args()


def test_single_image(pytorch_model, onnx_session, image_path):
    """测试单张图像"""
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # 预处理
    img_rgb = image[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # PyTorch 推理
    pytorch_model.eval()
    with torch.no_grad():
        start = time.time()
        pytorch_output = pytorch_model(img_tensor)
        pytorch_time = time.time() - start

    pytorch_output = torch.clamp(pytorch_output, 0, 1).cpu().numpy()

    # ONNX 推理
    start = time.time()
    onnx_output = onnx_session.run(None, {'input': img_tensor.numpy()})[0]
    onnx_time = time.time() - start

    onnx_output = np.clip(onnx_output, 0, 1)

    # 计算差异
    diff = np.abs(pytorch_output - onnx_output)

    return {
        'image': image_path.name,
        'size': f'{image.shape[1]}x{image.shape[0]}',
        'mean_diff': float(diff.mean()),
        'max_diff': float(diff.max()),
        'std_diff': float(diff.std()),
        'pytorch_time': float(pytorch_time),
        'onnx_time': float(onnx_time),
        'speedup': float(pytorch_time / onnx_time) if onnx_time > 0 else 0
    }


def main():
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    onnx_path = resolve_onnx_path(args.onnx)
    weights_path = resolve_weights_path(args.weights)
    image_dir = resolve_project_path(args.image_dir)

    # Check paths
    if not onnx_path.exists():
        print(f'Error: ONNX model not found: {onnx_path}')
        sys.exit(1)
    if not weights_path.exists():
        print(f'Error: Weights not found: {weights_path}')
        sys.exit(1)
    if not image_dir.exists():
        print(f'Error: Image directory not found: {image_dir}')
        sys.exit(1)

    print(f'ONNX Model: {onnx_path}')
    print(f'PyTorch Weights: {weights_path}')
    print(f'Image Directory: {image_dir}')

    # Load models
    print('\nLoading models...')
    enhancer = RetinexFormerEnhancer(str(weights_path), device='cpu')
    pytorch_model = enhancer.get_model()

    try:
        import onnxruntime as ort
    except ImportError:
        print('Error: onnxruntime not installed')
        sys.exit(1)

    onnx_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f'\nFound {len(image_files)} images')

    # Test images
    results = []
    print('\nTesting images...')
    for image_path in tqdm(image_files):
        result = test_single_image(pytorch_model, onnx_session, image_path)
        if result:
            results.append(result)

    # Statistics
    if not results:
        print('No valid results')
        return

    mean_diffs = [r['mean_diff'] for r in results]
    max_diffs = [r['max_diff'] for r in results]
    speedups = [r['speedup'] for r in results]

    report = {
        'summary': {
            'total_images': len(results),
            'avg_mean_diff': float(np.mean(mean_diffs)),
            'avg_max_diff': float(np.mean(max_diffs)),
            'max_mean_diff': float(np.max(mean_diffs)),
            'max_max_diff': float(np.max(max_diffs)),
            'avg_speedup': float(np.mean(speedups)),
            'passed': sum(1 for d in mean_diffs if d < 1e-4),
            'pass_rate': float(sum(1 for d in mean_diffs if d < 1e-4) / len(mean_diffs) * 100)
        },
        'details': results
    }

    # Save report
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f'\n{"="*60}')
    print('测试报告')
    print(f'{"="*60}')
    print(f'测试图像数: {report["summary"]["total_images"]}')
    print(f'平均 Mean Diff: {report["summary"]["avg_mean_diff"]:.6f}')
    print(f'平均 Max Diff: {report["summary"]["avg_max_diff"]:.6f}')
    print(f'最大 Mean Diff: {report["summary"]["max_mean_diff"]:.6f}')
    print(f'最大 Max Diff: {report["summary"]["max_max_diff"]:.6f}')
    print(f'通过率: {report["summary"]["passed"]}/{report["summary"]["total_images"]} ({report["summary"]["pass_rate"]:.1f}%)')
    print(f'平均加速比: {report["summary"]["avg_speedup"]:.2f}x')
    print(f'\n报告已保存到: {output_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
