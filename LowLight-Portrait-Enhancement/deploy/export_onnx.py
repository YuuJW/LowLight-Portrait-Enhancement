#!/usr/bin/env python3
"""
Export RetinexFormer model to ONNX format.

Usage:
    python deploy/export_onnx.py --output deploy/models/retinexformer.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RetinexFormerEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description='Export RetinexFormer to ONNX')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights path (default: models/LOL_v2_synthetic.pth)')
    parser.add_argument('--output', type=str, default='deploy/models/retinexformer.onnx',
                        help='Output ONNX path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512],
                        help='Input size [H, W] (default: 512 512)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnxsim')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX output against PyTorch')
    parser.add_argument('--check-model', action='store_true',
                        help='Check ONNX model validity')
    parser.add_argument('--show-info', action='store_true',
                        help='Show detailed ONNX model information')
    return parser.parse_args()


def export_onnx(model, output_path, input_size, opset_version):
    """Export PyTorch model to ONNX."""
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w).to(next(model.parameters()).device)

    print(f'Exporting ONNX with input size: {h}x{w}')
    print(f'Opset version: {opset_version}')

    # Use dynamo=False to force legacy exporter (avoids onnxscript compatibility issues)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=None,  # Fixed input size for NCNN compatibility
        dynamo=False  # Force legacy exporter for compatibility
    )

    print(f'Exported: {output_path}')


def simplify_onnx(input_path, output_path=None):
    """Simplify ONNX model using onnxsim."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print('Warning: onnxsim not installed. Run: pip install onnxsim')
        return False

    if output_path is None:
        output_path = input_path

    print('Simplifying ONNX model...')
    model = onnx.load(input_path)
    model_simp, check = simplify(model)

    if check:
        onnx.save(model_simp, output_path)
        print(f'Simplified: {output_path}')
        return True
    else:
        print('Warning: Simplification check failed')
        return False


def verify_onnx(pytorch_model, onnx_path, input_size, device):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print('Warning: onnxruntime not installed. Run: pip install onnxruntime')
        return False

    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w).to(device)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX inference
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_output = sess.run(None, {'input': dummy_input.cpu().numpy()})[0]

    # Compare
    diff = np.abs(pytorch_output - onnx_output).mean()
    max_diff = np.abs(pytorch_output - onnx_output).max()

    print(f'\nVerification:')
    print(f'  Mean diff: {diff:.6f}')
    print(f'  Max diff: {max_diff:.6f}')

    if diff < 1e-4:
        print('  Status: ✓ PASSED')
        return True
    else:
        print('  Status: ✗ FAILED (diff too large)')
        return False


def check_onnx_model(onnx_path):
    """Check ONNX model validity."""
    try:
        import onnx
    except ImportError:
        print('Warning: onnx not installed. Run: pip install onnx')
        return False

    print('\nChecking ONNX model...')
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print('  Status: ✓ Model is valid')
        return True
    except Exception as e:
        print(f'  Status: ✗ Model check failed: {e}')
        return False


def show_model_info(onnx_path):
    """Show detailed ONNX model information."""
    try:
        import onnx
    except ImportError:
        print('Warning: onnx not installed')
        return

    print('\n' + '='*60)
    print('ONNX Model Information')
    print('='*60)

    model = onnx.load(str(onnx_path))

    # Basic info
    print(f'\nProducer: {model.producer_name} {model.producer_version}')
    print(f'IR Version: {model.ir_version}')
    print(f'Opset Version: {model.opset_import[0].version}')

    # Graph info
    graph = model.graph
    print(f'\nGraph Name: {graph.name}')

    # Inputs
    print(f'\nInputs ({len(graph.input)}):')
    for inp in graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f'  - {inp.name}: {shape} (type: {dtype})')

    # Outputs
    print(f'\nOutputs ({len(graph.output)}):')
    for out in graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f'  - {out.name}: {shape} (type: {dtype})')

    # Nodes
    print(f'\nNodes: {len(graph.node)}')
    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    print('\nOperator Statistics:')
    for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {op_type}: {count}')

    # Model size
    import os
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'\nModel Size: {size_mb:.2f} MB')

    print('='*60)


def main():
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = project_root / 'models' / 'LOL_v2_synthetic.pth'

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    # Check weights exist
    if not weights_path.exists():
        print(f'Error: Weights not found: {weights_path}')
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Weights: {weights_path}')
    print(f'Output: {output_path}')

    # Load model
    print('\nLoading model...')
    enhancer = RetinexFormerEnhancer(str(weights_path), device='cpu')
    model = enhancer.get_model()

    # Export
    print('\nExporting to ONNX...')
    export_onnx(model, str(output_path), args.input_size, args.opset)

    # Simplify
    if args.simplify:
        simplify_onnx(str(output_path))

    # Verify
    if args.verify:
        verify_onnx(model, output_path, args.input_size, 'cpu')

    # Check model
    if args.check_model:
        check_onnx_model(output_path)

    # Show info
    if args.show_info:
        show_model_info(output_path)

    print('\nDone!')


if __name__ == '__main__':
    main()
