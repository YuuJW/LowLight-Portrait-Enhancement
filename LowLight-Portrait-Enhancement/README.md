# LowLight Portrait Enhancement

面向移动端的高性能低光照图像增强系统

> **策略 (2025-01)**: 采用预训练 RetinexFormer (ICCV 2023) 模型，专注部署工程优化。

## 项目亮点

| 维度 | 展示能力 |
|------|----------|
| 算法 | 理解 RetinexFormer 架构原理 (Retinex 分解 + Transformer) |
| 工程 | C++ 并行编程 (Tiling + ThreadPool)、内存管理、性能优化 |
| 部署 | NCNN 量化 (INT8/FP16)、移动端适配、真机实测 |

## 项目结构

```
LowLight-Portrait-Enhancement/
├── models/                   # 模型定义
│   ├── __init__.py
│   └── retinexformer.py     # RetinexFormer 模型 (TODO)
│
├── deploy/                   # 部署 (工程核心)
│   ├── export_onnx.py       # PyTorch → ONNX
│   ├── models/              # 转换后的模型存放
│   ├── cpp/                 # C++ 推理引擎
│   │   ├── include/
│   │   │   ├── tiling.h         # Tiling 分块处理
│   │   │   ├── thread_pool.h    # 线程池
│   │   │   └── ncnn_inference.h # NCNN 推理封装
│   │   └── src/
│   └── scripts/             # 转换脚本
│       ├── convert_ncnn.ps1
│       └── quantize_int8.ps1
│
├── scripts/                  # 工具脚本
│   └── download_pretrained.py
│
├── tests/                    # 测试
│   └── test_retinexformer.py
│
├── isp/                      # ISP 基础模块 (可选)
│   ├── include/
│   └── src/
│
├── benchmarks/               # 性能测试
│   └── results/
│
├── archive/                  # 备份代码
│   └── models/              # 旧版 RepVGG + U-Net 模型
│
└── data/                     # 数据集
    └── LOL/                 # Low-Light 验证数据集
```

## 技术架构

```
RetinexFormer 预训练权重
        │
        ▼
   ONNX 导出 → onnx-simplifier → NCNN 转换 → INT8 量化
                                      │
                                      ▼
                            ┌─────────────────┐
                            │  C++ 推理引擎   │
                            │                 │
                            │  Tiling 分块    │
                            │  ThreadPool     │
                            │  Overlap 融合   │
                            └─────────────────┘
                                      │
                                      ▼
                              移动端部署 (Android)
```

## 技术栈

- **模型**: RetinexFormer (ICCV 2023, Retinex + Transformer)
- **框架**: PyTorch → ONNX → NCNN
- **部署**: C++ (Tiling + ThreadPool)、INT8 量化
- **平台**: Android (ARM NEON)

## 进度

- [x] 项目结构设计
- [x] 技术方案确定 (RetinexFormer)
- [ ] 预训练模型下载与验证
- [ ] ONNX 导出
- [ ] C++ Tiling 模块
- [ ] C++ 线程池模块
- [ ] NCNN 推理封装
- [ ] INT8 量化
- [ ] 性能基准测试
- [ ] Android 集成 (可选)

## 快速开始

```bash
# 1. 环境配置
conda create -n lowlight python=3.10
conda activate lowlight
pip install torch torchvision onnx onnxruntime opencv-python

# 2. 下载预训练权重
python scripts/download_pretrained.py --model retinexformer

# 3. 验证推理
python tests/test_retinexformer.py --image data/LOL/eval15/low/1.png

# 4. 导出 ONNX
python deploy/export_onnx.py --model retinexformer --output deploy/models/
```

## 参考

- [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) - ICCV 2023
- [NCNN](https://github.com/Tencent/ncnn) - 腾讯移动端推理框架
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/) - 低光照数据集
