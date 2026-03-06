# RetinexFormer 低光照图像增强推理引擎

基于 RetinexFormer (ICCV 2023) 的高性能低光照图像增强推理引擎，采用 C++ + ONNX Runtime 实现。

> **项目定位**: PyTorch → ONNX → C++ 工程化部署链路，专注于桌面端高性能推理。

## 🎯 核心技术亮点

| 技术模块 | 实现方案 | 性能提升 |
|---------|---------|---------|
| **分块推理 (Tiling)** | 512×512 分块 + 32px 重叠 + 线性融合 | 支持任意尺寸输入，降低内存峰值 |
| **并行调度** | 自研 C++11 线程池 + 任务队列 | 多核 CPU 并行处理 |
| **会话池 (SessionPool)** | 多推理实例 + 无锁调度 | 4核 CPU 速度提升 **3-4倍** |
| **ONNX 优化** | onnx-simplifier + ONNX Runtime | 比 PyTorch 快 **1.2倍** |

## 📁 项目结构

```
LowLight-Portrait-Enhancement/
├── models/                   # Python 模型定义
│   ├── retinexformer.py     # RetinexFormer 封装
│   └── archs/               # 模型架构
│
├── deploy/                   # 部署工具链 ⭐
│   ├── export_onnx.py       # ONNX 导出工具
│   ├── verify_onnx.py       # 验证工具（一致性+性能+可视化）
│   ├── batch_test_onnx.py   # 批量测试工具
│   ├── models/              # 模型文件
│   └── cpp/                 # C++ 推理引擎 ⭐⭐⭐
│       ├── include/
│       │   ├── onnx_wrapper.h          # ONNX Runtime 封装
│       │   ├── session_pool.h          # 会话池（核心优化）
│       │   ├── thread_pool.h           # 线程池（自研）
│       │   ├── tiling_manager.h        # 分块管理
│       │   └── retinexformer_engine.h  # 主引擎
│       └── src/
│
├── tests/                    # 测试脚本
├── scripts/                  # 工具脚本
├── archive/                  # 备份代码（早期 RepVGG 实现）
└── benchmarks/               # 性能测试
```

## 🔄 技术架构

```
┌─────────────────────────────────────────────────────────┐
│  PyTorch 模型 (RetinexFormer)                           │
│  - 预训练权重: LOL_v2_synthetic.pth                     │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  ONNX 导出 + 简化                                        │
│  - export_onnx.py: PyTorch → ONNX                       │
│  - onnx-simplifier: 模型优化                            │
│  - verify_onnx.py: 一致性验证                           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  C++ 推理引擎 (ONNX Runtime)                            │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ TilingManager│  │  ThreadPool  │  │ SessionPool  │  │
│  │  分块管理    │  │  线程池调度  │  │  会话池管理  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  特性:                                                  │
│  - 处理任意尺寸输入                                     │
│  - 多线程并行推理                                       │
│  - 无锁并行调度                                         │
│  - 边界伪影消除                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/YuuJW/LowLight-Portrait-Enhancement.git
cd LowLight-Portrait-Enhancement

# 创建 conda 环境
conda create -n IMG_Env python=3.10 -y
conda activate IMG_Env

# 安装依赖
pip install torch torchvision onnx onnxruntime onnxsim opencv-python matplotlib tqdm einops
```

### 2. 下载预训练权重

| 权重文件 | PSNR | 下载链接 |
|---------|------|---------|
| LOL_v2_synthetic.pth | 29.04 | [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV) |
| LOL_v2_real.pth | 27.71 | [百度网盘](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) 提取码: cyh2 |

下载后放到 `models/` 目录。

### 3. Python 端：ONNX 导出

```bash
# 导出 ONNX 模型（包含简化和验证）
python deploy/export_onnx.py \
    --weights models/LOL_v2_synthetic.pth \
    --output deploy/models/retinexformer.onnx \
    --simplify --verify

# 详细验证（一致性 + 性能测试）
python deploy/verify_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --num-tests 10 \
    --benchmark

# 批量测试
python deploy/batch_test_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --image-dir data/LOL/lol_dataset/eval15/low
```

### 4. C++ 端：编译推理引擎

```bash
cd deploy/cpp
mkdir build && cd build

# 配置 CMake
cmake -G "Visual Studio 17 2022" -A x64 ..

# 编译
cmake --build . --config Release

# 运行测试
.\Release\test_engine.exe model.onnx input.png
```

## 📊 性能数据

### ONNX 导出验证

| 指标 | 结果 |
|-----|------|
| 数值一致性 | **100%** 通过 (Mean diff < 1e-6) |
| 模型大小 | **6.4 MB** (简化后) |
| 推理速度 | ONNX Runtime 比 PyTorch 快 **1.19x** |

### C++ 推理引擎性能

| 图像尺寸 | Tile 数量 | 单会话+Mutex | SessionPool | 加速比 |
|---------|---------|-------------|-------------|--------|
| 512×512 | 1 | ~0.5s | ~0.4s | 1.25x |
| 1024×1024 | 4 | ~2.0s | ~0.5s | **4.0x** |
| 2048×2048 | 16 | ~8.0s | ~2.0s | **4.0x** |
| 4096×4096 | 64 | ~32.0s | ~8.0s | **4.0x** |

**关键优化**:
- ✅ SessionPool 消除了 mutex 串行瓶颈
- ✅ CPU 利用率从 25% 提升到 **100%**
- ✅ 4核 CPU 上速度提升 **3-4 倍**

## 🛠️ 技术栈

**Python 端**:
- PyTorch 2.10.0
- ONNX 1.20.1
- ONNX Runtime 1.23.2
- OpenCV, matplotlib

**C++ 端**:
- C++14
- ONNX Runtime
- OpenCV
- CMake

## 📝 使用示例

### Python 推理

```python
from models import RetinexFormerEnhancer
import cv2

# 加载模型
enhancer = RetinexFormerEnhancer('models/LOL_v2_synthetic.pth')

# 增强图像
image = cv2.imread('input.png')
enhanced = enhancer.enhance(image)
cv2.imwrite('output.png', enhanced)
```

### C++ 推理

```cpp
#include "retinexformer_engine.h"

int main() {
    // 创建引擎（4个线程，4个会话）
    RetinexFormerEngine engine("model.onnx", 4);

    // 读取图像
    cv::Mat input = cv::imread("input.png");

    // 增强图像
    cv::Mat output = engine.enhance(input);

    // 保存结果
    cv::imwrite("output.png", output);

    return 0;
}
```

## 📈 进度

- [x] 项目结构设计
- [x] RetinexFormer 模型封装
- [x] ONNX 导出工具
- [x] ONNX 验证工具
- [x] C++ Tiling 模块
- [x] C++ 线程池模块
- [x] C++ 会话池优化
- [x] ONNX Runtime 推理封装
- [ ] 性能可视化工具
- [ ] 完整 Demo 示例

## 🔗 参考

- [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) - ICCV 2023
- [ONNX Runtime](https://onnxruntime.ai/) - 微软推理框架
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/) - 低光照数据集

## 📄 License

本项目仅供学习和研究使用。
