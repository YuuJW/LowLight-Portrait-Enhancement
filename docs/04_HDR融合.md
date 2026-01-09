# 04 - HDR 融合

本文档详细介绍多帧 HDR（High Dynamic Range）融合的原理和实现方法。

---

## 目录

1. [HDR 概述](#1-hdr-概述)
2. [多帧 HDR 原理](#2-多帧-hdr-原理)
3. [图像对齐](#3-图像对齐)
4. [曝光融合算法](#4-曝光融合算法)
5. [鬼影处理](#5-鬼影处理)
6. [代码实现](#6-代码实现)
7. [面试要点](#7-面试要点)

---

## 1. HDR 概述

### 1.1 什么是 HDR？

HDR（High Dynamic Range，高动态范围）是一种扩展图像亮度范围的技术，使得图像能同时保留亮部和暗部的细节。

### 1.2 为什么需要 HDR？

| 场景 | 问题 | HDR 解决方案 |
|------|------|-------------|
| 逆光人像 | 人脸暗，背景过曝 | 融合多曝光，两者都清晰 |
| 室内窗户 | 室内暗，窗外过曝 | 保留室内细节和窗外景色 |
| 日落场景 | 天空过曝或地面欠曝 | 同时保留天空和地面细节 |

### 1.3 动态范围

- **人眼动态范围**：约 10^14:1（14 档）
- **相机传感器**：约 10^3 ~ 10^4:1（10-14 档）
- **显示器**：约 10^2 ~ 10^3:1（8-10 档）

### 1.4 HDR 技术分类

```
HDR 技术
├── 单帧 HDR
│   ├── 传感器 HDR（Staggered HDR、DOL-HDR）
│   └── 单帧算法增强
│
└── 多帧 HDR
    ├── 多曝光融合（Exposure Fusion）
    └── HDR 合成 + 色调映射（HDR + Tone Mapping）
```

---

## 2. 多帧 HDR 原理

### 2.1 多曝光拍摄

快速连续拍摄多张不同曝光时间的照片：

```
┌─────────────────────────────────────────────────────────┐
│                    多曝光序列                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   短曝光（亮部细节）  中曝光        长曝光（暗部细节）   │
│   ┌─────────────┐   ┌─────────────┐  ┌─────────────┐    │
│   │  天空清晰   │   │   均衡     │   │  阴影清晰  │    │
│   │  地面欠曝   │   │            │   │  天空过曝  │    │
│   └─────────────┘   └─────────────┘  └─────────────┘    │
│                                                          │
│                         ↓ 融合                           │
│                                                          │
│                   ┌─────────────┐                        │
│                   │  HDR 结果   │                        │
│                   │  全部清晰   │                        │
│                   └─────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 HDR 处理流程

```
多曝光图像 → 图像对齐 → 曝光融合 → 色调映射 → 输出
                │           │           │
                ▼           ▼           ▼
            处理手抖     合成 HDR    压缩动态范围
```

### 2.3 两种融合方法

| 方法 | 特点 | 优缺点 |
|------|------|--------|
| **曝光融合（Mertens）** | 直接融合 LDR 图像 | 简单快速，无需色调映射 |
| **HDR 合成 + Tone Mapping** | 先合成 HDR，再映射 | 灵活，但计算量大 |

---

## 3. 图像对齐

### 3.1 为什么需要对齐？

多帧拍摄之间可能存在：
- **相机抖动**：手持拍摄导致的位移
- **物体运动**：场景中移动的物体

### 3.2 对齐方法

#### 方法1：基于特征点

```python
# 使用 ORB 特征点对齐
import cv2
import numpy as np

def align_images_orb(reference, target, max_features=500):
    """
    使用 ORB 特征点对齐图像

    Args:
        reference: 参考图像（通常选中曝光）
        target: 需要对齐的图像

    Returns:
        aligned: 对齐后的图像
    """
    # 转换为灰度
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # 检测 ORB 特征点
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 取最好的匹配
    good_matches = matches[:int(len(matches) * 0.5)]

    # 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 应用变换
    h, w = reference.shape[:2]
    aligned = cv2.warpPerspective(target, H, (w, h))

    return aligned
```

#### 方法2：基于光流

```python
def align_images_flow(reference, target):
    """
    使用光流对齐图像
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(
        tgt_gray, ref_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # 应用光流
    h, w = reference.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

    map_x += flow[:, :, 0]
    map_y += flow[:, :, 1]

    aligned = cv2.remap(target, map_x, map_y, cv2.INTER_LINEAR)

    return aligned
```

#### 方法3：ECC（Enhanced Correlation Coefficient）

```python
def align_images_ecc(reference, target):
    """
    使用 ECC 算法对齐图像
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # 定义变换类型
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    # 计算变换
    _, warp_matrix = cv2.findTransformECC(ref_gray, tgt_gray, warp_matrix, warp_mode, criteria)

    # 应用变换
    h, w = reference.shape[:2]
    aligned = cv2.warpAffine(target, warp_matrix, (w, h))

    return aligned
```

---

## 4. 曝光融合算法

### 4.1 Mertens 曝光融合

**原理**：根据每个像素的质量（对比度、饱和度、曝光度）计算权重，加权融合。

```python
def mertens_fusion(images):
    """
    Mertens 曝光融合

    Args:
        images: 多曝光图像列表 [img1, img2, img3, ...]

    Returns:
        fused: 融合后的图像
    """
    # 创建融合器
    merge_mertens = cv2.createMergeMertens()

    # 融合
    fused = merge_mertens.process(images)

    # 转换为 8-bit
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    return fused
```

**自定义实现**：

```python
def mertens_fusion_custom(images, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0):
    """
    自定义 Mertens 融合实现
    """
    weights = []

    for img in images:
        img_float = img.astype(np.float32) / 255.0

        # 1. 对比度权重（使用 Laplacian）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = np.abs(laplacian)

        # 2. 饱和度权重
        saturation = img_float.std(axis=2)

        # 3. 曝光度权重（接近 0.5 的值权重高）
        exposure = np.exp(-0.5 * ((img_float - 0.5) ** 2) / (0.2 ** 2))
        exposure = np.prod(exposure, axis=2)

        # 组合权重
        weight = (contrast ** contrast_weight) * \
                 (saturation ** saturation_weight) * \
                 (exposure ** exposure_weight)
        weight += 1e-6  # 防止除零
        weights.append(weight)

    # 归一化权重
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    # 金字塔融合
    fused = pyramid_blend(images, weights)

    return fused


def pyramid_blend(images, weights, levels=5):
    """
    使用拉普拉斯金字塔进行融合
    """
    # 构建高斯金字塔
    weight_pyramids = []
    for w in weights:
        gp = [w]
        for _ in range(levels - 1):
            gp.append(cv2.pyrDown(gp[-1]))
        weight_pyramids.append(gp)

    # 构建拉普拉斯金字塔
    laplacian_pyramids = []
    for img in images:
        img_float = img.astype(np.float32)
        gp = [img_float]
        for _ in range(levels - 1):
            gp.append(cv2.pyrDown(gp[-1]))

        lp = [gp[-1]]
        for i in range(levels - 1, 0, -1):
            up = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
            lp.append(gp[i-1] - up)
        lp.reverse()
        laplacian_pyramids.append(lp)

    # 融合各层
    fused_pyramid = []
    for level in range(levels):
        fused_level = np.zeros_like(laplacian_pyramids[0][level])
        for i in range(len(images)):
            w = weight_pyramids[i][level]
            if len(fused_level.shape) == 3:
                w = w[:, :, np.newaxis]
            fused_level += laplacian_pyramids[i][level] * w
        fused_pyramid.append(fused_level)

    # 重建图像
    fused = fused_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        fused = cv2.pyrUp(fused, dstsize=(fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))
        fused += fused_pyramid[i]

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused
```

### 4.2 Debevec HDR 合成

```python
def debevec_hdr(images, exposure_times):
    """
    Debevec 方法合成 HDR

    Args:
        images: 多曝光图像列表
        exposure_times: 曝光时间列表

    Returns:
        hdr: HDR 图像（浮点）
    """
    # 计算相机响应曲线
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, np.array(exposure_times, dtype=np.float32))

    # 合成 HDR
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, np.array(exposure_times, dtype=np.float32), response)

    return hdr


def tone_mapping(hdr, method='reinhard'):
    """
    色调映射

    Args:
        hdr: HDR 图像
        method: 映射方法

    Returns:
        ldr: LDR 图像
    """
    if method == 'reinhard':
        tonemap = cv2.createTonemapReinhard(gamma=2.2)
    elif method == 'drago':
        tonemap = cv2.createTonemapDrago(gamma=2.2)
    elif method == 'mantiuk':
        tonemap = cv2.createTonemapMantiuk(gamma=2.2)

    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    return ldr
```

---

## 5. 鬼影处理

### 5.1 什么是鬼影？

当场景中有运动物体时，多帧融合会产生"鬼影"（Ghosting）。

```
┌─────────────────────────────────────────────────────────┐
│                    鬼影问题示意                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   帧1              帧2              帧3                  │
│   ┌─────┐          ┌─────┐          ┌─────┐            │
│   │ 🚶  │          │  🚶 │          │   🚶│            │
│   └─────┘          └─────┘          └─────┘            │
│                                                          │
│                    ↓ 直接融合                            │
│                                                          │
│                    ┌─────┐                               │
│                    │🚶🚶🚶│  ← 鬼影                      │
│                    └─────┘                               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 鬼影检测

```python
def detect_ghost(reference, target, threshold=30):
    """
    检测鬼影区域

    Args:
        reference: 参考帧
        target: 目标帧
        threshold: 差异阈值

    Returns:
        ghost_mask: 鬼影区域 mask
    """
    # 转换为灰度
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 考虑曝光差异，归一化
    ref_norm = ref_gray / (ref_gray.mean() + 1e-6)
    tgt_norm = tgt_gray / (tgt_gray.mean() + 1e-6)

    # 计算差异
    diff = np.abs(ref_norm - tgt_norm)

    # 阈值化
    ghost_mask = (diff > threshold / 255.0).astype(np.uint8)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ghost_mask = cv2.morphologyEx(ghost_mask, cv2.MORPH_CLOSE, kernel)
    ghost_mask = cv2.morphologyEx(ghost_mask, cv2.MORPH_OPEN, kernel)

    return ghost_mask
```

### 5.3 鬼影消除

**策略1**：在鬼影区域只使用参考帧

```python
def remove_ghost_simple(fused, reference, ghost_mask):
    """
    简单鬼影消除：用参考帧替换鬼影区域
    """
    result = fused.copy()
    mask_3ch = np.stack([ghost_mask] * 3, axis=2)
    result = np.where(mask_3ch > 0, reference, result)
    return result
```

**策略2**：降低鬼影区域的融合权重

```python
def fusion_with_ghost_handling(images, reference_idx=1):
    """
    带鬼影处理的融合
    """
    reference = images[reference_idx]
    weights = []

    for i, img in enumerate(images):
        # 计算基础权重（Mertens）
        base_weight = compute_mertens_weight(img)

        # 检测鬼影
        if i != reference_idx:
            ghost_mask = detect_ghost(reference, img)
            # 降低鬼影区域权重
            base_weight = base_weight * (1 - ghost_mask * 0.9)

        weights.append(base_weight)

    # 归一化并融合
    # ...
```

---

## 6. 代码实现

### 6.1 完整 HDR 融合流程

```python
# hdr_fusion.py
import cv2
import numpy as np
from typing import List

class HDRFusion:
    def __init__(self):
        self.align_method = 'ecc'  # 'orb', 'flow', 'ecc'
        self.fusion_method = 'mertens'  # 'mertens', 'debevec'
        self.ghost_detection = True

    def process(self, images: List[np.ndarray],
                exposure_times: List[float] = None) -> np.ndarray:
        """
        HDR 融合主流程

        Args:
            images: 多曝光图像列表
            exposure_times: 曝光时间（Debevec 方法需要）

        Returns:
            result: 融合后的图像
        """
        if len(images) < 2:
            return images[0]

        # 1. 选择参考帧（通常选中间曝光）
        ref_idx = len(images) // 2
        reference = images[ref_idx]

        # 2. 图像对齐
        aligned_images = [reference]
        for i, img in enumerate(images):
            if i != ref_idx:
                aligned = self.align(reference, img)
                aligned_images.append(aligned)
            else:
                aligned_images.append(img)

        # 重新排序
        aligned_images = [aligned_images[i] for i in range(len(images))]

        # 3. 鬼影检测
        ghost_masks = []
        if self.ghost_detection:
            for i, img in enumerate(aligned_images):
                if i != ref_idx:
                    mask = detect_ghost(reference, img)
                    ghost_masks.append(mask)
                else:
                    ghost_masks.append(np.zeros_like(img[:, :, 0]))

        # 4. 融合
        if self.fusion_method == 'mertens':
            result = self.mertens_fusion(aligned_images, ghost_masks, ref_idx)
        else:
            result = self.debevec_fusion(aligned_images, exposure_times)

        return result

    def align(self, reference, target):
        if self.align_method == 'orb':
            return align_images_orb(reference, target)
        elif self.align_method == 'flow':
            return align_images_flow(reference, target)
        else:
            return align_images_ecc(reference, target)

    def mertens_fusion(self, images, ghost_masks, ref_idx):
        merge = cv2.createMergeMertens()
        fused = merge.process(images)
        fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

        # 鬼影处理
        if ghost_masks:
            combined_mask = np.zeros_like(ghost_masks[0])
            for mask in ghost_masks:
                combined_mask = np.maximum(combined_mask, mask)

            if combined_mask.sum() > 0:
                fused = remove_ghost_simple(fused, images[ref_idx], combined_mask)

        return fused

    def debevec_fusion(self, images, exposure_times):
        hdr = debevec_hdr(images, exposure_times)
        ldr = tone_mapping(hdr, method='reinhard')
        return ldr


# 使用示例
if __name__ == '__main__':
    # 读取多曝光图像
    img1 = cv2.imread('short_exposure.jpg')
    img2 = cv2.imread('medium_exposure.jpg')
    img3 = cv2.imread('long_exposure.jpg')

    images = [img1, img2, img3]

    # HDR 融合
    hdr_fusion = HDRFusion()
    result = hdr_fusion.process(images)

    cv2.imwrite('hdr_result.jpg', result)
```

---

## 7. 面试要点

### 高频问题

**Q1: HDR 成像的原理是什么？**

**回答**：
- 拍摄多张不同曝光的照片
- 短曝光保留亮部细节，长曝光保留暗部细节
- 通过对齐和融合，合成高动态范围图像
- 最后通过色调映射转换为可显示的 LDR 图像

**Q2: 多帧 HDR 如何处理运动物体造成的鬼影？**

**回答**：
1. **检测鬼影**：比较多帧之间的差异
2. **处理策略**：
   - 在鬼影区域只使用参考帧
   - 降低鬼影区域的融合权重
   - 使用光流估计运动并补偿
3. **高级方法**：使用深度学习检测和修复

**Q3: Mertens 融合和 Debevec 方法有什么区别？**

| 方面 | Mertens | Debevec |
|------|---------|---------|
| 输入要求 | 只需图像 | 需要曝光时间 |
| 中间表示 | 直接融合 | 生成 HDR 图像 |
| 后处理 | 无需色调映射 | 需要色调映射 |
| 灵活性 | 简单固定 | 可调参数多 |
| 计算量 | 较小 | 较大 |

**Q4: 手机 HDR 是如何实现实时预览的？**

**回答**：
- 使用硬件 ISP 进行实时多帧融合
- 采用简化的融合算法
- 利用 NPU 加速
- 分帧率处理（预览用低分辨率，拍摄用高分辨率）

---

## 下一步

完成 HDR 融合后，继续阅读 [05_AI降噪.md](./05_AI降噪.md) 学习 AI 降噪模块。
