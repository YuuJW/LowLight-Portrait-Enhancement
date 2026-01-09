# 03 - AWB 模块

本文档详细介绍自动白平衡（AWB）的原理和实现方法。

---

## 目录

1. [白平衡概述](#1-白平衡概述)
2. [AWB 算法原理](#2-awb-算法原理)
3. [代码实现](#3-代码实现)
4. [集成到 HDR-ISP](#4-集成到-hdr-isp)
5. [效果评估](#5-效果评估)
6. [面试要点](#6-面试要点)

---

## 1. 白平衡概述

### 1.1 什么是白平衡？

白平衡（White Balance）是校正图像色温的过程，使得白色物体在图像中呈现为白色。

### 1.2 为什么需要白平衡？

不同光源有不同的色温：

| 光源 | 色温 (K) | 色偏 |
|------|----------|------|
| 蜡烛 | 1800 | 偏橙红 |
| 白炽灯 | 2700-3000 | 偏黄 |
| 日光 | 5500-6500 | 中性 |
| 阴天 | 6500-7500 | 偏蓝 |
| 阴影 | 8000-10000 | 偏蓝紫 |

相机传感器捕获的是物理光信号，不会像人眼一样自动适应色温。

### 1.3 白平衡的本质

```cpp
// 白平衡就是对 R、G、B 通道分别乘以增益
R_out = R_in * r_gain;
G_out = G_in * g_gain;  // 通常 g_gain = 1（归一化）
B_out = B_in * b_gain;
```

**AWB 的核心任务**：自动估计 `r_gain` 和 `b_gain`。

---

## 2. AWB 算法原理

### 2.1 灰度世界法（Gray World）

**假设**：场景中所有颜色的平均值应该是灰色（即 R = G = B）。

**算法**：
```
R_avg = mean(R)
G_avg = mean(G)
B_avg = mean(B)

r_gain = G_avg / R_avg
b_gain = G_avg / B_avg
g_gain = 1
```

**优点**：
- 简单快速
- 适合色彩丰富的场景

**缺点**：
- 对单一颜色主导的场景效果差
- 不适合偏色严重的场景

### 2.2 白点检测法（White Patch / Max RGB）

**假设**：场景中最亮的点是白色。

**算法**：
```
R_max = max(R)
G_max = max(G)
B_max = max(B)

r_gain = max(R_max, G_max, B_max) / R_max
g_gain = max(R_max, G_max, B_max) / G_max
b_gain = max(R_max, G_max, B_max) / B_max
```

**优点**：
- 对高光区域敏感

**缺点**：
- 对过曝敏感
- 场景中必须有白色区域

### 2.3 完美反射法（Perfect Reflector）

**假设**：场景中存在完美反射体（白色表面）。

**算法**：
1. 找到亮度最高的前 N% 像素
2. 计算这些像素的平均值
3. 用该平均值估计光源色温

### 2.4 灰度边缘法（Gray Edge）

**假设**：图像边缘的平均颜色应该是灰色。

**算法**：
```
# 计算梯度
grad_R = gradient(R)
grad_G = gradient(G)
grad_B = gradient(B)

# 对梯度应用灰度世界
r_gain = mean(|grad_G|) / mean(|grad_R|)
b_gain = mean(|grad_G|) / mean(|grad_B|)
```

**优点**：
- 对颜色分布不均的场景更鲁棒

### 2.5 基于统计的混合方法

实际应用中通常结合多种方法：

```
1. 剔除过曝/欠曝像素
2. 计算灰度世界增益
3. 计算白点增益
4. 加权融合两种结果
5. 限制增益范围，防止异常值
```

### 2.6 基于学习的方法

现代手机通常使用机器学习方法：
- 使用大量标注数据训练模型
- 输入：图像统计特征
- 输出：增益值或色温

---

## 3. 代码实现

### 3.1 灰度世界法实现

```cpp
// awb_gray_world.cpp
#include <opencv2/opencv.hpp>
#include <iostream>

struct AWBGains {
    float r_gain;
    float g_gain;
    float b_gain;
};

AWBGains grayWorldAWB(const cv::Mat& bayer_img, const std::string& cfa_pattern) {
    // 分离 Bayer 通道
    int height = bayer_img.rows;
    int width = bayer_img.cols;

    double r_sum = 0, g_sum = 0, b_sum = 0;
    int r_count = 0, g_count = 0, b_count = 0;

    // 根据 CFA 模式确定各通道位置
    int r_row, r_col, b_row, b_col;
    if (cfa_pattern == "RGGB") {
        r_row = 0; r_col = 0;
        b_row = 1; b_col = 1;
    } else if (cfa_pattern == "BGGR") {
        r_row = 1; r_col = 1;
        b_row = 0; b_col = 0;
    } else if (cfa_pattern == "GRBG") {
        r_row = 0; r_col = 1;
        b_row = 1; b_col = 0;
    } else { // GBRG
        r_row = 1; r_col = 0;
        b_row = 0; b_col = 1;
    }

    // 统计各通道均值
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = bayer_img.at<float>(y, x);

            if ((y % 2 == r_row) && (x % 2 == r_col)) {
                r_sum += val;
                r_count++;
            } else if ((y % 2 == b_row) && (x % 2 == b_col)) {
                b_sum += val;
                b_count++;
            } else {
                g_sum += val;
                g_count++;
            }
        }
    }

    double r_avg = r_sum / r_count;
    double g_avg = g_sum / g_count;
    double b_avg = b_sum / b_count;

    // 计算增益
    AWBGains gains;
    gains.g_gain = 1.0f;
    gains.r_gain = static_cast<float>(g_avg / r_avg);
    gains.b_gain = static_cast<float>(g_avg / b_avg);

    // 限制增益范围 [0.5, 4.0]
    gains.r_gain = std::max(0.5f, std::min(4.0f, gains.r_gain));
    gains.b_gain = std::max(0.5f, std::min(4.0f, gains.b_gain));

    std::cout << "Gray World AWB Gains: R=" << gains.r_gain
              << ", G=" << gains.g_gain
              << ", B=" << gains.b_gain << std::endl;

    return gains;
}
```

### 3.2 白点检测法实现

```cpp
// awb_white_patch.cpp
AWBGains whitePatchAWB(const cv::Mat& bayer_img, const std::string& cfa_pattern,
                        float percentile = 0.95) {
    int height = bayer_img.rows;
    int width = bayer_img.cols;

    std::vector<float> r_vals, g_vals, b_vals;

    // 根据 CFA 模式分离通道（与灰度世界相同）
    int r_row, r_col, b_row, b_col;
    // ... 省略 CFA 模式判断

    // 收集各通道像素值
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = bayer_img.at<float>(y, x);
            if ((y % 2 == r_row) && (x % 2 == r_col)) {
                r_vals.push_back(val);
            } else if ((y % 2 == b_row) && (x % 2 == b_col)) {
                b_vals.push_back(val);
            } else {
                g_vals.push_back(val);
            }
        }
    }

    // 排序
    std::sort(r_vals.begin(), r_vals.end());
    std::sort(g_vals.begin(), g_vals.end());
    std::sort(b_vals.begin(), b_vals.end());

    // 取高百分位值
    int r_idx = static_cast<int>(r_vals.size() * percentile);
    int g_idx = static_cast<int>(g_vals.size() * percentile);
    int b_idx = static_cast<int>(b_vals.size() * percentile);

    float r_white = r_vals[r_idx];
    float g_white = g_vals[g_idx];
    float b_white = b_vals[b_idx];

    // 计算增益
    float max_white = std::max({r_white, g_white, b_white});

    AWBGains gains;
    gains.r_gain = max_white / r_white;
    gains.g_gain = max_white / g_white;
    gains.b_gain = max_white / b_white;

    // 归一化到 G=1
    gains.r_gain /= gains.g_gain;
    gains.b_gain /= gains.g_gain;
    gains.g_gain = 1.0f;

    // 限制范围
    gains.r_gain = std::max(0.5f, std::min(4.0f, gains.r_gain));
    gains.b_gain = std::max(0.5f, std::min(4.0f, gains.b_gain));

    return gains;
}
```

### 3.3 应用增益

```cpp
// apply_wb_gains.cpp
void applyWBGains(cv::Mat& bayer_img, const AWBGains& gains,
                  const std::string& cfa_pattern) {
    int height = bayer_img.rows;
    int width = bayer_img.cols;

    int r_row, r_col, b_row, b_col;
    // ... CFA 模式判断

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if ((y % 2 == r_row) && (x % 2 == r_col)) {
                bayer_img.at<float>(y, x) *= gains.r_gain;
            } else if ((y % 2 == b_row) && (x % 2 == b_col)) {
                bayer_img.at<float>(y, x) *= gains.b_gain;
            } else {
                bayer_img.at<float>(y, x) *= gains.g_gain;
            }
        }
    }
}
```

### 3.4 Python 版本

```python
# awb.py
import numpy as np

def gray_world_awb(bayer_img, cfa_pattern='RGGB'):
    """
    灰度世界自动白平衡

    Args:
        bayer_img: Bayer 格式图像 (H, W)
        cfa_pattern: Bayer 模式

    Returns:
        gains: (r_gain, g_gain, b_gain)
    """
    h, w = bayer_img.shape

    # 根据 CFA 模式提取通道
    if cfa_pattern == 'RGGB':
        r = bayer_img[0::2, 0::2]
        g = (bayer_img[0::2, 1::2] + bayer_img[1::2, 0::2]) / 2
        b = bayer_img[1::2, 1::2]
    elif cfa_pattern == 'BGGR':
        b = bayer_img[0::2, 0::2]
        g = (bayer_img[0::2, 1::2] + bayer_img[1::2, 0::2]) / 2
        r = bayer_img[1::2, 1::2]
    # ... 其他模式

    # 计算均值
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)

    # 计算增益
    r_gain = g_avg / r_avg
    b_gain = g_avg / b_avg
    g_gain = 1.0

    # 限制范围
    r_gain = np.clip(r_gain, 0.5, 4.0)
    b_gain = np.clip(b_gain, 0.5, 4.0)

    return (r_gain, g_gain, b_gain)


def apply_wb_gains(bayer_img, gains, cfa_pattern='RGGB'):
    """
    应用白平衡增益

    Args:
        bayer_img: Bayer 格式图像
        gains: (r_gain, g_gain, b_gain)
        cfa_pattern: Bayer 模式

    Returns:
        wb_img: 白平衡后的图像
    """
    r_gain, g_gain, b_gain = gains
    wb_img = bayer_img.copy().astype(np.float32)

    if cfa_pattern == 'RGGB':
        wb_img[0::2, 0::2] *= r_gain  # R
        wb_img[0::2, 1::2] *= g_gain  # G
        wb_img[1::2, 0::2] *= g_gain  # G
        wb_img[1::2, 1::2] *= b_gain  # B
    # ... 其他模式

    return wb_img
```

---

## 4. 集成到 HDR-ISP

### 4.1 添加 AWB 模块

在 HDR-ISP 项目中添加 AWB 功能：

**步骤**：
1. 创建 `awb.h` 和 `awb.cpp` 文件
2. 在 Pipeline 中调用 AWB（在 Demosaic 之前）
3. 在 JSON 配置中添加 AWB 参数

### 4.2 配置文件修改

```json
{
    "awb": {
        "enable": true,
        "method": "gray_world",  // "gray_world" 或 "white_patch"
        "gain_limit_min": 0.5,
        "gain_limit_max": 4.0
    }
}
```

### 4.3 Pipeline 调用

```cpp
// main.cpp 中的 Pipeline
void runISPPipeline(cv::Mat& raw_img, const Config& cfg) {
    // 1. BLC
    applyBLC(raw_img, cfg.blc);

    // 2. DPC
    applyDPC(raw_img, cfg.dpc);

    // 3. AWB（新增）
    if (cfg.awb.enable) {
        AWBGains gains;
        if (cfg.awb.method == "gray_world") {
            gains = grayWorldAWB(raw_img, cfg.info.cfa);
        } else {
            gains = whitePatchAWB(raw_img, cfg.info.cfa);
        }
        applyWBGains(raw_img, gains, cfg.info.cfa);
    }

    // 4. Demosaic
    cv::Mat rgb_img = demosaic(raw_img, cfg.demosaic);

    // ... 后续处理
}
```

---

## 5. 效果评估

### 5.1 测试场景

准备不同光源下的测试图片：
- 日光场景
- 室内白炽灯场景
- 室内荧光灯场景
- 混合光源场景

### 5.2 评估方法

**主观评估**：
- 白色区域是否呈白色
- 肤色是否自然
- 整体色彩是否协调

**客观评估**：
- 使用色卡（ColorChecker）计算色差
- Delta E（CIE Lab 色差）

### 5.3 对比实验

| 场景 | 无 AWB | Gray World | White Patch |
|------|--------|------------|-------------|
| 日光 | 偏蓝 | 正常 | 正常 |
| 白炽灯 | 偏黄 | 正常 | 略过曝 |
| 荧光灯 | 偏绿 | 正常 | 正常 |

---

## 6. 面试要点

### 高频问题

**Q1: AWB 的常用算法有哪些？**

**回答**：
1. **灰度世界法**：假设场景平均颜色为灰色
2. **白点检测法**：假设最亮点是白色
3. **灰度边缘法**：假设边缘平均颜色为灰色
4. **基于学习的方法**：使用神经网络估计色温

**Q2: 灰度世界假设有什么局限性？**

**回答**：
- 对单一颜色主导的场景失效（如蓝天、绿草地）
- 对严重偏色的场景效果差
- 需要场景中有足够多的颜色分布

**Q3: 如何改进灰度世界法？**

**回答**：
1. 剔除过曝/欠曝像素
2. 只使用中间亮度区域的像素
3. 结合边缘信息（灰度边缘法）
4. 使用加权平均（根据像素可靠性）
5. 与其他方法融合

**Q4: 手机厂商的 AWB 是怎么实现的？**

**回答**：
- 通常使用多种方法融合
- 利用机器学习模型
- 结合场景检测（人脸、天空等）
- 利用历史帧信息（视频场景）
- 在 ISP 芯片中硬件加速

---

## 下一步

完成 AWB 模块后，继续阅读 [04_HDR融合.md](./04_HDR融合.md) 学习多帧 HDR 融合。
