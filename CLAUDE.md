# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LowLight-Portrait-Enhancement (暗光人像增强系统) - A low-light portrait enhancement system targeting mobile deployment using reparameterization techniques. This is a portfolio project for mobile imaging algorithm development.

**Tech Stack:**
- ISP Pipeline: C++ (black level correction, white balance, demosaicing)
- Deep Learning: PyTorch (U-Net + RepVGG blocks)
- Mobile Deployment: NCNN framework with INT8 quantization
- Target Platform: Android (ARM)

## Repository Structure

```
img_pro/
├── LowLight-Portrait-Enhancement/   # Main project
│   ├── isp/                         # P1: C++ ISP modules (BLC, AWB, Demosaic)
│   │   ├── CMakeLists.txt
│   │   ├── include/                 # Header files
│   │   └── src/                     # Implementation
│   │
│   ├── models/                      # P0: PyTorch model definitions
│   │   ├── repvgg_block.py          # RepVGG reparameterization
│   │   ├── unet_repvgg.py           # U-Net backbone
│   │   └── losses.py                # L1 + Perceptual (+ Face)
│   │
│   ├── train/                       # P0: Training pipeline
│   │   ├── data_synthesis.py        # Low-light data synthesis
│   │   ├── dataset.py               # Data loader
│   │   ├── train.py                 # Training script
│   │   └── configs/                 # Training configs
│   │
│   ├── deploy/                      # P0: Deployment (engineering highlight)
│   │   ├── export_onnx.py           # ONNX export
│   │   ├── cpp/                     # C++ inference engine
│   │   │   ├── include/
│   │   │   │   ├── tiling.h         # Tiling processor
│   │   │   │   ├── thread_pool.h    # Thread pool
│   │   │   │   └── ncnn_inference.h # NCNN wrapper
│   │   │   ├── src/
│   │   │   └── CMakeLists.txt
│   │   └── scripts/                 # Conversion scripts
│   │
│   ├── data/                        # Datasets
│   │   ├── LOL/                     # Public low-light dataset
│   │   └── FFHQ_lowlight/           # Synthesized data
│   │
│   └── benchmarks/                  # Performance testing
│
├── docs/                            # Technical documentation (Chinese)
├── interview/                       # Interview preparation materials
└── references/HDR-ISP-main/         # Reference C++ ISP implementation
```

## Module Priority

- **P0 (Must Complete)**: models/, train/, deploy/cpp/ - Core training and deployment
- **P1 (Should Complete)**: isp/, INT8 quantization, Face Loss
- **P2 (Optional)**: Android JNI, HDR, AI denoising, AI super-resolution

## Build Commands

### Python Environment Setup
```powershell
conda create -n isp python=3.10
conda activate isp

# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install opencv-python numpy matplotlib scikit-image rawpy imageio onnx onnxruntime-gpu lpips basicsr tqdm pyyaml
```

### C++ Build (Reference ISP Project)
```powershell
cd references/HDR-ISP-main
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### Run ISP Pipeline
```powershell
.\HDR_ISP.exe .\cfgs\isp_config_cannon.json
```

## Model Deployment Pipeline

The deployment flow is critical for this project:

```
PyTorch (.pth) → switch_to_deploy() → ONNX → onnx-simplifier → NCNN → INT8 quantization
```

Key commands:
```bash
# Simplify ONNX model (required for NCNN compatibility)
python -m onnxsim model.onnx model_sim.onnx

# Convert to NCNN
./onnx2ncnn model_sim.onnx model.param model.bin

# Optimize NCNN model (fp16 storage)
./ncnnoptimize model.param model.bin model_opt.param model_opt.bin 65536
```

**Critical**: For RepVGG models, always call `model.switch_to_deploy()` before ONNX export to merge multi-branch training structure into single-branch inference structure.

## Architecture Notes

### ISP Pipeline Modules
The reference implementation in `references/HDR-ISP-main/srcs/sources/modules/` contains 20+ ISP modules:
- BLC (Black Level Correction)
- AWB (Auto White Balance)
- Demosaicing
- CCM (Color Correction Matrix)
- Gamma correction
- Denoising, Sharpening, Saturation, etc.

### Neural Network Architecture
- **Backbone**: U-Net with RepVGG blocks for reparameterization
- **Loss Functions**: L1 + Perceptual Loss (VGG features) + Face Parsing Loss
- **Key Feature**: RepVGG enables multi-branch training (better accuracy) with single-branch inference (faster deployment)

### Why NCNN (not TensorRT)
Mobile phones use ARM CPUs with Adreno/Mali GPUs - no NVIDIA hardware. NCNN is optimized for ARM NEON instructions. TensorRT only works on NVIDIA GPUs (servers/PCs).

## Documentation Index

All docs are in Chinese, located in `docs/`:
- `01_环境配置.md` - Environment setup
- `02_ISP基础.md` - ISP pipeline fundamentals
- `03_AWB模块.md` - Auto white balance implementation
- `04_HDR融合.md` - HDR fusion (optional module)
- `05_AI降噪.md` - AI denoising (optional module)
- `06_AI超分.md` - AI super-resolution (optional module)
- `08_NCNN部署.md` - Mobile deployment guide (core technical content)
- `09_数据合成.md` - Low-light data synthesis strategy (core)

Interview materials: `interview/07_面试准备.md`

## Reference Projects (in references/)

| Project | Purpose | Usage |
|---------|---------|-------|
| `HDR-ISP-main/` | C++ ISP reference implementation | BLC, AWB, Demosaic code reference |
| `ThreadPool/` | C++11 thread pool (~100 lines) | Direct copy for parallel inference |
| `ncnn/` | Mobile inference framework | Model conversion and deployment |
| `BiSeNet/` | Face parsing model | Generate face masks for Face Loss |

## Key Configuration Files

- ISP configs: `references/HDR-ISP-main/cfgs/isp_config_*.json`
- C++ build: `references/HDR-ISP-main/CMakeLists.txt`
- ThreadPool: `references/ThreadPool/ThreadPool.h` (single header)
