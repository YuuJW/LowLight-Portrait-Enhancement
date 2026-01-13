# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LowLight-Portrait-Enhancement (暗光人像增强系统) - A low-light portrait enhancement system targeting mobile deployment using pre-trained **RetinexFormer (ICCV 2023)**.

**Tech Stack:** PyTorch → ONNX → NCNN (INT8 quantization) → C++ Engine (Tiling + ThreadPool) → Android ARM

## Working Directory

The main project code is in `LowLight-Portrait-Enhancement/`. All Python commands should be run from this subdirectory.

## Build Commands

### Initial Setup (run once after cloning)
```powershell
cd LowLight-Portrait-Enhancement

# Create conda environment
conda create -n lowlight python=3.10
conda activate lowlight

# Install dependencies
pip install torch torchvision onnx onnxruntime opencv-python gdown

# Run setup script (clones RetinexFormer repo + downloads weights)
python scripts/setup.py
```

Setup script options:
- `--skip-weights` - Skip weight download (for manual download)
- `--mirror` - Use mirror for git clone (if network issues)
- `--weight LOL_v2_real` - Select different weight version

### Development Commands
```powershell
# Test RetinexFormer inference (after models/retinexformer.py is implemented)
python tests/test_retinexformer.py --image data/LOL/eval15/low/1.png

# Export to ONNX (after deploy/export_onnx.py is implemented)
python deploy/export_onnx.py --model retinexformer --output deploy/models/
```

### NCNN Conversion Pipeline
```bash
# Simplify ONNX model (required for NCNN compatibility)
python -m onnxsim model.onnx model_sim.onnx

# Convert to NCNN
./onnx2ncnn model_sim.onnx model.param model.bin

# Optimize NCNN model (fp16 storage)
./ncnnoptimize model.param model.bin model_opt.param model_opt.bin 65536

# INT8 quantization
./ncnn2table model_opt.param model_opt.bin calibration/ model.table
./ncnn2int8 model_opt.param model_opt.bin model_int8.param model_int8.bin model.table
```

### C++ Build (reference ISP project)
```powershell
cd references/HDR-ISP-main
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Implementation Status

| Module | Status | Priority |
|--------|--------|----------|
| `models/retinexformer.py` | TODO | P0 |
| `deploy/export_onnx.py` | TODO | P0 |
| `deploy/cpp/` (Tiling + ThreadPool + NCNN) | TODO | P0 |
| `tests/test_retinexformer.py` | TODO | P0 |
| `isp/` (C++ ISP modules) | TODO | P1 |
| `archive/models/` | Archived | Reference only |

## Architecture

### RetinexFormer (ICCV 2023)
- **Paper**: "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement"
- **Architecture**: Illumination Estimator → Illumination-Guided Attention → Reflectance Restoration
- **Retinex decomposition**: I = L × R (Image = Illumination × Reflectance)
- **Reference code**: `references/Retinexformer-master/`

### C++ Inference Engine Design
1. **Tiling**: Split large images (4K/12MP) into 512x512 tiles with 32px overlap
2. **ThreadPool**: C++11 thread pool for parallel tile inference (see `references/ThreadPool/ThreadPool.h`)
3. **Overlap Blending**: Linear alpha blending in overlap regions
4. **NCNN Wrapper**: Per-tile inference with optimized memory management

### Why NCNN (not TensorRT)
Mobile phones use ARM CPUs (no NVIDIA GPU). NCNN is optimized for ARM NEON. TensorRT only works on NVIDIA hardware.

## Key Directories

- `LowLight-Portrait-Enhancement/` - Main project implementation
- `references/Retinexformer-master/` - Official RetinexFormer code (cloned by setup.py)
- `references/ThreadPool/` - C++11 thread pool header (copy directly for use)
- `references/HDR-ISP-main/` - C++ ISP reference implementation
- `references/ncnn/` - NCNN framework reference
- `docs/` - Chinese documentation (environment setup, ISP basics, NCNN deployment)

## Archived Code

`LowLight-Portrait-Enhancement/archive/models/` contains previous custom model implementations:
- `repvgg_block.py` - RepVGG reparameterization module with train→deploy mode switching
- `unet_repvgg.py` - U-Net + RepVGG backbone
- `losses.py` - L1 + Perceptual loss functions

These are preserved for potential future custom training but not currently used.

## Pre-trained Weights

Weights are downloaded by `setup.py` to `deploy/models/`:
- `LOL_v2_synthetic.pth` - PSNR 29.04 (recommended)
- `LOL_v2_real.pth` - PSNR 27.71
- `LOL_v1.pth` - PSNR 27.18

Manual download: [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV) or [百度网盘](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2)
