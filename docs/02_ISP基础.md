# 02 - ISP 基础

本文档介绍 ISP（Image Signal Processor）管线的基本原理，以及如何理解 HDR-ISP 项目代码。

---

## 目录

1. [ISP 概述](#1-isp-概述)
2. [ISP Pipeline 流程](#2-isp-pipeline-流程)
3. [HDR-ISP 项目分析](#3-hdr-isp-项目分析)
4. [各模块原理](#4-各模块原理)
5. [实践任务](#5-实践任务)
6. [面试要点](#6-面试要点)

---

## 1. ISP 概述

### 1.1 什么是 ISP？

ISP（Image Signal Processor）是将相机传感器输出的原始数据（RAW）转换为可视图像的处理单元。

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  镜头   │────▶│ 传感器  │────▶│   ISP   │────▶ 图像
└─────────┘     └─────────┘     └─────────┘
                    │                │
                    ▼                ▼
                RAW 数据        RGB/YUV 图像
```

### 1.2 为什么需要 ISP？

| RAW 数据特点 | ISP 处理目标 |
|-------------|-------------|
| 单通道 Bayer 格式 | 转换为 RGB 三通道 |
| 包含噪声 | 降噪处理 |
| 动态范围有限 | HDR 增强 |
| 色彩不准确 | 白平衡、色彩校正 |
| 对比度低 | Gamma 校正、对比度增强 |

### 1.3 Bayer 格式

相机传感器使用 Bayer 滤镜阵列，每个像素只采集一种颜色：

```
┌───┬───┬───┬───┐
│ R │ G │ R │ G │
├───┼───┼───┼───┤
│ G │ B │ G │ B │
├───┼───┼───┼───┤
│ R │ G │ R │ G │
├───┼───┼───┼───┤
│ G │ B │ G │ B │
└───┴───┴───┴───┘
  RGGB Bayer Pattern
```

常见 Bayer 模式：
- **RGGB**：左上角为 R
- **BGGR**：左上角为 B
- **GRBG**：左上角为 G（R在右）
- **GBRG**：左上角为 G（B在右）

---

## 2. ISP Pipeline 流程

### 2.1 标准 ISP 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW Domain                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAW ──▶ BLC ──▶ DPC ──▶ LSC ──▶ WB ──▶ Demosaic ──▶ RGB       │
│          │       │       │       │        │                      │
│          ▼       ▼       ▼       ▼        ▼                      │
│       黑电平   坏点    镜头阴影  白平衡   去马赛克                │
│       校正     校正    校正     增益                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RGB Domain                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RGB ──▶ CCM ──▶ Gamma ──▶ Denoise ──▶ RGB2YUV ──▶ YUV         │
│          │        │         │            │                       │
│          ▼        ▼         ▼            ▼                       │
│       色彩校正  伽马      降噪        色彩空间                   │
│       矩阵      曲线                  转换                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        YUV Domain                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  YUV ──▶ Contrast ──▶ Sharpen ──▶ Saturation ──▶ YUV2RGB       │
│           │            │            │              │             │
│           ▼            ▼            ▼              ▼             │
│        对比度        锐化         饱和度       输出RGB          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 各模块简介

| 模块 | 英文全称 | 作用 |
|------|---------|------|
| BLC | Black Level Correction | 黑电平校正，去除传感器暗电流 |
| DPC | Defect Pixel Correction | 坏点校正，修复坏像素 |
| LSC | Lens Shading Correction | 镜头阴影校正，修复边缘暗角 |
| WB | White Balance | 白平衡，校正色温 |
| Demosaic | Demosaicing | 去马赛克，Bayer → RGB |
| CCM | Color Correction Matrix | 色彩校正矩阵 |
| Gamma | Gamma Correction | 伽马校正，非线性映射 |
| Denoise | Noise Reduction | 降噪 |
| Sharpen | Sharpening | 锐化 |

---

## 3. HDR-ISP 项目分析

### 3.1 获取代码

```bash
# Fork 项目到自己的 GitHub
# 然后 clone
git clone https://github.com/你的用户名/HDR-ISP.git
cd HDR-ISP
```

### 3.2 项目结构

```
HDR-ISP/
├── CMakeLists.txt          # CMake 配置
├── main.cpp                # 主程序入口
├── parse.cpp               # JSON 配置解析
├── cfgs/                   # 配置文件
│   └── isp_config_cannon.json
├── data/                   # 测试数据
│   └── connan_raw14.raw
├── docs/                   # 文档和图片
└── src/                    # 源代码（如果有）
```

### 3.3 配置文件分析

查看 `cfgs/isp_config_cannon.json`：

```json
{
    "raw_file": "./data/connan_raw14.raw",
    "out_file_path": "./",
    "info": {
        "sensor_name": "cannon",
        "cfa": "RGGB",           // Bayer 模式
        "data_type": "RAW16",    // 数据格式
        "bpp": 16,               // 位深
        "max_bit": 14,           // 有效位数
        "width": 6080,           // 图像宽度
        "height": 4044,          // 图像高度
        "mipi_packed": 0         // MIPI 打包
    },
    // ... 各模块配置
}
```

### 3.4 编译运行

```bash
# 创建 build 目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译
cmake --build . --config Release

# 复制配置和数据
cp -r ../cfgs ./
cp -r ../data ./

# 运行
./HDR_ISP ./cfgs/isp_config_cannon.json
```

### 3.5 代码阅读顺序

建议按以下顺序阅读代码：

1. **main.cpp** - 理解整体流程
2. **parse.cpp** - 理解配置解析
3. **各模块实现** - 按 Pipeline 顺序阅读

---

## 4. 各模块原理

### 4.1 黑电平校正（BLC）

**原理**：传感器即使在无光条件下也会产生暗电流，需要减去这个偏移量。

```cpp
// 黑电平校正
output = input - black_level;
output = max(output, 0);  // 防止负值
```

**参数**：
- `black_level`：黑电平值，通常 64-256（10bit 传感器）

### 4.2 坏点校正（DPC）

**原理**：检测并修复亮点（stuck high）和暗点（stuck low）。

**常用方法**：
1. 使用周围像素的中值替换
2. 使用周围同色像素的均值

```cpp
// 简单的中值滤波坏点校正
if (abs(pixel - median_of_neighbors) > threshold) {
    pixel = median_of_neighbors;
}
```

### 4.3 镜头阴影校正（LSC）

**原理**：镜头边缘亮度下降（cos^4 law），需要补偿。

```cpp
// 镜头阴影校正
// gain_map 是预先标定的增益表
output[x][y] = input[x][y] * gain_map[x][y];
```

### 4.4 白平衡（WB）

**原理**：校正不同光源色温造成的色偏。

```cpp
// 白平衡增益
R_out = R_in * r_gain;
G_out = G_in * g_gain;  // 通常 g_gain = 1
B_out = B_in * b_gain;
```

**详细内容见 [03_AWB模块.md](./03_AWB模块.md)**

### 4.5 去马赛克（Demosaic）

**原理**：将单通道 Bayer 图像转换为三通道 RGB。

**常用算法**：
1. **双线性插值**：简单但会产生伪彩
2. **边缘自适应**：根据梯度选择插值方向
3. **MLRI**：更复杂的优化算法

```cpp
// 双线性插值示例（对于 G 在 R 位置）
G = (G_left + G_right + G_up + G_down) / 4;
```

### 4.6 色彩校正矩阵（CCM）

**原理**：将传感器色彩空间映射到标准色彩空间（如 sRGB）。

```cpp
// 3x3 色彩校正矩阵
[R']   [m00 m01 m02] [R]
[G'] = [m10 m11 m12] [G]
[B']   [m20 m21 m22] [B]
```

**CCM 标定**：使用色卡（如 X-Rite ColorChecker）拍摄并计算矩阵。

### 4.7 伽马校正（Gamma）

**原理**：人眼对亮度的感知是非线性的，需要应用 Gamma 曲线。

```cpp
// 标准 sRGB Gamma（简化版）
output = pow(input, 1.0 / 2.2);
```

**实际实现**通常使用查找表（LUT）加速：

```cpp
// 使用 LUT
output = gamma_lut[input];
```

### 4.8 降噪（Denoise）

**传统方法**：
1. **高斯滤波**：平滑但模糊边缘
2. **双边滤波**：保边降噪
3. **非局部均值（NLM）**：效果好但计算量大

**AI 方法**：见 [05_AI降噪.md](./05_AI降噪.md)

### 4.9 锐化（Sharpen）

**原理**：增强边缘，提升清晰度。

**常用方法**：USM（Unsharp Masking）

```cpp
// USM 锐化
blurred = GaussianBlur(input);
mask = input - blurred;
output = input + amount * mask;
```

---

## 5. 实践任务

### 任务 1：理解 HDR-ISP 代码

**目标**：阅读并理解代码结构

**步骤**：
1. 阅读 main.cpp，画出函数调用流程图
2. 阅读配置文件，理解每个参数的作用
3. 修改参数，观察输出变化

**交付物**：
- 代码结构笔记
- 参数调整实验记录

### 任务 2：跑通 Demo

**目标**：成功编译运行项目

**步骤**：
1. 按照 [01_环境配置.md](./01_环境配置.md) 配置环境
2. 编译项目
3. 运行 Demo
4. 查看输出结果

**交付物**：
- 成功运行的截图
- 输出图像

### 任务 3：分析处理效果

**目标**：理解每个模块的作用

**步骤**：
1. 逐个关闭模块（在配置中设置 enable: false）
2. 对比开启/关闭时的输出差异
3. 记录观察结果

**交付物**：
- 各模块对比图
- 效果分析报告

---

## 6. 面试要点

### 高频问题

**Q1: 描述一下 ISP Pipeline 的主要模块**

**回答要点**：
- RAW 域：BLC → DPC → LSC → WB → Demosaic
- RGB 域：CCM → Gamma → Denoise
- YUV 域：Contrast → Sharpen → Saturation
- 说明每个模块的作用

**Q2: Demosaic 算法有哪些？各有什么优缺点？**

| 算法 | 优点 | 缺点 |
|------|------|------|
| 双线性插值 | 简单快速 | 边缘伪彩明显 |
| 边缘自适应 | 减少伪彩 | 计算量增加 |
| MLRI/VNG | 效果好 | 计算复杂 |
| AI 方法 | 效果最好 | 需要大量计算 |

**Q3: 什么是 CCM？如何标定？**

**回答要点**：
- CCM 是 3x3 色彩校正矩阵
- 用于将传感器色彩空间转换到标准色彩空间
- 标定方法：拍摄色卡，通过最小二乘法计算矩阵

**Q4: Gamma 校正的原理是什么？**

**回答要点**：
- 人眼对亮度的感知是非线性的（Weber-Fechner 定律）
- Gamma 曲线压缩高亮、扩展暗部，符合人眼感知
- 标准 sRGB Gamma ≈ 2.2
- 实现时通常使用 LUT 加速

---

## 下一步

完成 ISP 基础学习后，继续阅读 [03_AWB模块.md](./03_AWB模块.md) 学习如何实现自动白平衡。
