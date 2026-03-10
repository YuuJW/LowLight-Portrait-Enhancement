#!/usr/bin/env python3
"""
Test script for RetinexFormer inference.

Usage:
    python tests/test_retinexformer.py --image data/LOL/lol_dataset/eval15/low/1.png
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_paths import resolve_project_path, resolve_weights_path
from models import RetinexFormerEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description='Test RetinexFormer inference')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights path (default: models/LOL_v2_synthetic.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: input_enhanced.png)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    image_path = resolve_project_path(args.image)
    weights_path = resolve_weights_path(args.weights)

    if args.output:
        output_path = resolve_project_path(args.output)
    else:
        output_path = image_path.parent / f'{image_path.stem}_enhanced{image_path.suffix}'

    # Check files exist
    if not image_path.exists():
        print(f'Error: Image not found: {image_path}')
        sys.exit(1)
    if not weights_path.exists():
        print(f'Error: Weights not found: {weights_path}')
        sys.exit(1)

    print(f'Image: {image_path}')
    print(f'Weights: {weights_path}')
    print(f'Output: {output_path}')

    # Load model
    print('\nLoading model...')
    t0 = time.time()
    enhancer = RetinexFormerEnhancer(str(weights_path), device=args.device)
    print(f'Model loaded in {time.time() - t0:.2f}s')
    print(f'Device: {enhancer.device}')

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f'Error: Failed to load image: {image_path}')
        sys.exit(1)
    print(f'Input size: {image.shape[1]}x{image.shape[0]}')

    # Enhance
    print('\nRunning inference...')
    t0 = time.time()
    enhanced = enhancer.enhance(image)
    inference_time = time.time() - t0
    print(f'Inference time: {inference_time:.3f}s')

    # Save result
    cv2.imwrite(str(output_path), enhanced)
    print(f'\nSaved: {output_path}')


if __name__ == '__main__':
    main()
