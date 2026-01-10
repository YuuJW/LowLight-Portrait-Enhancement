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

### 首次配置（克隆后运行一次）

```bash
# 1. 克隆项目
git clone https://github.com/YuuJW/LowLight-Portrait-Enhancement.git
cd LowLight-Portrait-Enhancement

# 2. 创建 conda 环境
conda create -n lowlight python=3.10
conda activate lowlight

# 3. 安装依赖
pip install torch torchvision onnx onnxruntime opencv-python gdown

# 4. 运行配置脚本（自动克隆 RetinexFormer + 下载权重）
python scripts/setup.py
```

配置脚本会自动完成：
- 克隆 RetinexFormer 官方仓库到 `../references/Retinexformer/`
- 下载预训练权重到 `deploy/models/`
- 创建必要的目录结构

如果网络有问题，可以使用镜像或跳过权重下载：
```bash
python scripts/setup.py --mirror           # 使用镜像加速
python scripts/setup.py --skip-weights     # 跳过权重下载（手动下载）
```

### 手动下载权重（备用）

如果自动下载失败，请手动下载：

| 权重 | PSNR | 下载链接 |
|------|------|----------|
| LOL_v2_synthetic.pth | 29.04 | [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV) |
| LOL_v2_real.pth | 27.71 | [百度网盘](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) 提取码: cyh2 |

下载后放到 `deploy/models/` 目录。

### 开发流程

```bash
# 验证推理
python tests/test_retinexformer.py --image data/LOL/eval15/low/1.png

# 导出 ONNX
python deploy/export_onnx.py --model retinexformer --output deploy/models/
```

## 参考

- [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) - ICCV 2023
- [NCNN](https://github.com/Tencent/ncnn) - 腾讯移动端推理框架
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/) - 低光照数据集
