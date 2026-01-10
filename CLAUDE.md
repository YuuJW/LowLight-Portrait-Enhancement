# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LowLight-Portrait-Enhancement (暗光人像增强系统) - A low-light portrait enhancement system targeting mobile deployment. This is a portfolio project for mobile imaging algorithm development.

**Current Strategy**: Use pre-trained **RetinexFormer (ICCV 2023)** to build deployment pipeline first, then optionally train custom models later.

**Tech Stack:**
- Deep Learning: PyTorch (Pre-trained RetinexFormer)
- Mobile Deployment: NCNN framework with INT8 quantization
- C++ Engine: Tiling + ThreadPool for large image processing
- Target Platform: Android (ARM)
- Optional: C++ ISP modules (BLC, AWB, Demosaic)

## Implementation Status

| Module | Status | Description |
|--------|--------|-------------|
| `models/` | 🔄 In Progress | RetinexFormer integration (pre-trained) |
| `deploy/cpp/` | ❌ TODO | C++ inference engine (Tiling + ThreadPool + NCNN) |
| `deploy/scripts/` | ❌ TODO | ONNX export, NCNN conversion, INT8 quantization |
| `isp/` | ❌ TODO (P1) | C++ ISP modules |
| `archive/models/` | 📦 Archived | Old RepVGG/U-Net code (for reference) |

## Module Priority

- **P0 (Critical)**: deploy/cpp/, deploy/scripts/ - Core deployment engineering
- **P1 (Should Complete)**: isp/, benchmarks/, INT8 quantization optimization
- **P2 (Optional)**: Custom model training, Android JNI, Face Loss

## Build Commands

### Python Environment Setup
```powershell
conda create -n lowlight python=3.10
conda activate lowlight

# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install opencv-python numpy matplotlib onnx onnxruntime-gpu onnxsim tqdm pyyaml
```

### Download RetinexFormer Pretrained Model
```powershell
cd LowLight-Portrait-Enhancement
python scripts/download_pretrained.py --model retinexformer
```

### Test RetinexFormer Inference
```powershell
python tests/test_retinexformer.py --image data/LOL/eval15/low/1.png
```

### Export to ONNX
```powershell
python deploy/export_onnx.py --model retinexformer --output deploy/models/
```

### C++ Build (Reference ISP Project)
```powershell
cd references/HDR-ISP-main
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Model Deployment Pipeline

```
RetinexFormer (.pth) → ONNX → onnx-simplifier → NCNN → INT8 quantization
```

Key commands:
```bash
# Simplify ONNX model (required for NCNN compatibility)
python -m onnxsim model.onnx model_sim.onnx

# Convert to NCNN
./onnx2ncnn model_sim.onnx model.param model.bin

# Optimize NCNN model (fp16 storage)
./ncnnoptimize model.param model.bin model_opt.param model_opt.bin 65536

# INT8 quantization (after preparing calibration images)
./ncnn2table model_opt.param model_opt.bin calibration/ model.table
./ncnn2int8 model_opt.param model_opt.bin model_int8.param model_int8.bin model.table
```

## Architecture Notes

### RetinexFormer (ICCV 2023) - Current Model
- **Paper**: "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement"
- **Architecture**: Illumination Estimator + Illumination-Guided Attention + Reflectance Restoration
- **Parameters**: ~3-4M
- **GitHub**: https://github.com/caiyuanhao1998/Retinexformer

### Why RetinexFormer?
1. **State-of-the-art quality** on LOL dataset (~22-24 dB PSNR)
2. **Physically interpretable** - Retinex decomposition (I = L × R)
3. **Pre-trained weights available** - No training required
4. **NCNN compatible** - Standard Transformer ops supported

### C++ Inference Engine Design (Core Engineering Work)
The deployment engine features:
1. **Tiling**: Split large images (4K/12MP) into 512x512 tiles with 32px overlap
2. **Thread Pool**: C++11 thread pool for parallel tile inference (reference: `references/ThreadPool/ThreadPool.h`)
3. **Overlap Blending**: Linear alpha blending in overlap regions to eliminate seams
4. **NCNN Wrapper**: Per-tile inference with optimized memory management

### Why NCNN (not TensorRT)
Mobile phones use ARM CPUs with Adreno/Mali GPUs - no NVIDIA hardware. NCNN is optimized for ARM NEON instructions. TensorRT only works on NVIDIA GPUs.

### Archived Models (in archive/models/)
Previous custom model implementations (for future reference):
- `repvgg_block.py` - RepVGG reparameterization module
- `unet_repvgg.py` - U-Net + RepVGG backbone
- `losses.py` - L1 + Perceptual Loss

## Project Structure

```
LowLight-Portrait-Enhancement/
├── models/
│   ├── __init__.py
│   └── retinexformer.py      # RetinexFormer model loader
│
├── deploy/
│   ├── export_onnx.py        # ONNX export script
│   ├── models/               # Converted models (.onnx, .param, .bin)
│   ├── calibration/          # INT8 calibration images
│   ├── cpp/                  # C++ inference engine
│   │   ├── include/          # Headers (tiling.h, thread_pool.h, ncnn_inference.h)
│   │   ├── src/              # Implementation
│   │   └── CMakeLists.txt
│   └── scripts/              # Conversion scripts
│
├── scripts/
│   └── download_pretrained.py
│
├── tests/
│   └── test_retinexformer.py
│
├── data/
│   └── LOL/                  # Low-light dataset for validation
│
├── archive/                  # Archived old code
│   └── models/               # RepVGG, U-Net, losses (backup)
│
└── benchmarks/               # Performance testing results
```

## Documentation Index

All docs are in Chinese, located in `docs/`:
- `01_环境配置.md` - Environment setup
- `02_ISP基础.md` - ISP pipeline fundamentals
- `08_NCNN部署.md` - Mobile deployment guide (core)
- `09_数据合成.md` - Low-light data synthesis (optional for custom training)

Interview materials: `interview/07_面试准备.md`

## Reference Projects (in references/)

| Project | Purpose | Usage |
|---------|---------|-------|
| `HDR-ISP-main/` | C++ ISP reference | BLC, AWB, Demosaic code reference |
| `ThreadPool/` | C++11 thread pool | Direct copy for parallel inference |
| `ncnn/` | Mobile inference framework | Model conversion and deployment |

## Key Interview Points

1. **Why RetinexFormer?** - Retinex physical decomposition + Transformer global attention
2. **Transformer on mobile challenges** - Tiling + ThreadPool + mixed-precision quantization
3. **Engineering focus** - 90% deployment optimization, 10% model setup
