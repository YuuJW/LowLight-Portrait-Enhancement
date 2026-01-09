# 08 - NCNN 移动端部署

本文档详细介绍如何将 PyTorch 模型部署到 Android 手机端。

---

## 目录

1. [为什么用 NCNN](#1-为什么用-ncnn)
2. [环境准备](#2-环境准备)
3. [模型转换流程](#3-模型转换流程)
4. [INT8 量化](#4-int8-量化)
5. [Android 集成](#5-android-集成)
6. [常见问题与解决](#6-常见问题与解决)
7. [面试要点](#7-面试要点)

---

## 1. 为什么用 NCNN

### 1.1 移动端推理框架对比

| 框架 | 公司 | 适用平台 | 特点 |
|------|------|---------|------|
| **NCNN** | 腾讯 | ARM CPU | 轻量、专为移动端优化 |
| **MNN** | 阿里 | ARM CPU/GPU | 全面、支持 GPU |
| **TFLite** | Google | ARM CPU/GPU/NPU | 与 TensorFlow 兼容 |
| TensorRT | NVIDIA | NVIDIA GPU | ❌ 手机没有 NVIDIA GPU |

### 1.2 为什么不能用 TensorRT

```
┌─────────────────────────────────────────────────────────┐
│  PC/服务器                    │  手机                    │
├──────────────────────────────┼──────────────────────────┤
│  Intel/AMD CPU               │  ARM CPU (Snapdragon等)  │
│  NVIDIA GPU (RTX/Tesla)      │  Adreno GPU / Mali GPU   │
│  ✅ 可以用 TensorRT          │  ❌ 没有 NVIDIA GPU      │
│  ✅ 可以用 CUDA              │  ✅ 用 NCNN/MNN/TFLite   │
└─────────────────────────────────────────────────────────┘
```

**关键点**：手机使用的是 ARM 架构，配合高通 Adreno、ARM Mali 或华为达芬奇等 GPU/NPU，没有 NVIDIA 显卡。

### 1.3 NCNN 的优势

- **轻量**：无第三方依赖，纯 C++ 实现
- **ARM 优化**：针对 ARM NEON 指令集深度优化
- **跨平台**：支持 Android、iOS、Linux、Windows
- **腾讯背书**：大量商业项目验证

---

## 2. 环境准备

### 2.1 编译 NCNN

**Linux/Mac**：
```bash
# 下载 NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# 编译
mkdir build && cd build
cmake ..
make -j$(nproc)

# 工具在 build/tools/onnx/ 目录
ls build/tools/onnx/
# onnx2ncnn  ncnnoptimize
```

**Windows**（使用 VS2022）：
```powershell
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### 2.2 安装 onnx-simplifier

```bash
pip install onnx-simplifier
```

---

## 3. 模型转换流程

### 3.1 完整流程

```
PyTorch Model (.pth)
       ↓
   [1] 切换到部署模式（RepVGG 必须）
       ↓
   [2] 导出 ONNX
       ↓
   [3] onnx-simplifier 简化
       ↓
   [4] onnx2ncnn 转换
       ↓
   [5] ncnnoptimize 优化
       ↓
   [6] (可选) INT8 量化
       ↓
NCNN Model (.param + .bin)
```

### 3.2 Step 1: 切换部署模式（RepVGG）

```python
# ⚠️ 关键步骤！如果模型使用了 RepVGG，必须先切换到部署模式

import torch
from models.unet_repvgg import LowLightNet

# 加载模型
model = LowLightNet()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# 切换到部署模式（将多分支合并为单卷积）
model.switch_to_deploy()  # 关键！

# 验证：此时模型应该没有 Add 分支了
```

### 3.3 Step 2: 导出 ONNX

```python
# export_onnx.py
import torch

def export_onnx(model, output_path, input_size=(1, 3, 256, 256)):
    """
    导出 ONNX 模型

    Args:
        model: PyTorch 模型（已切换到部署模式）
        output_path: 输出路径
        input_size: 输入尺寸
    """
    model.eval()
    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=11  # NCNN 支持良好
    )
    print(f'Exported to {output_path}')


if __name__ == '__main__':
    from models.unet_repvgg import LowLightNet

    model = LowLightNet()
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.switch_to_deploy()  # 关键！

    export_onnx(model, 'deploy/model.onnx')
```

### 3.4 Step 3: 简化 ONNX

```bash
# 使用 onnx-simplifier 简化模型
# 这一步可以解决大部分 NCNN 兼容性问题

python -m onnxsim deploy/model.onnx deploy/model_sim.onnx

# 验证简化效果
# 可以用 Netron (https://netron.app) 可视化对比
```

### 3.5 Step 4: 转换为 NCNN

```bash
# 转换
./onnx2ncnn deploy/model_sim.onnx deploy/model.param deploy/model.bin

# 如果报错 "xxx op not supported"，检查：
# 1. 是否运行了 onnx-simplifier
# 2. 是否切换了部署模式（RepVGG）
# 3. opset 版本是否为 11
```

### 3.6 Step 5: 优化 NCNN 模型

```bash
# 优化（层融合、内存优化）
./ncnnoptimize deploy/model.param deploy/model.bin deploy/model_opt.param deploy/model_opt.bin 65536

# 65536 = fp16 存储，减少一半体积
# 0 = fp32 存储
```

---

## 4. INT8 量化

### 4.1 为什么要量化

| 精度 | 模型大小 | 推理速度 | 精度损失 |
|------|---------|---------|---------|
| FP32 | 100% | 基准 | 0 |
| FP16 | 50% | 1.5-2x | ~0.1% |
| **INT8** | **25%** | **2-3x** | **<0.5%** |

### 4.2 准备校准数据

```python
# prepare_calibration.py
import os
import cv2
import numpy as np

def prepare_calibration_images(input_dir, output_dir, num_images=1000):
    """
    准备 INT8 量化校准数据

    需要 1000+ 张有代表性的图片
    """
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)[:num_images]

    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(input_dir, img_name))
        img = cv2.resize(img, (256, 256))

        # 保存为 NCNN 要求的格式（BGR，不需要归一化）
        output_path = os.path.join(output_dir, f'{i:04d}.jpg')
        cv2.imwrite(output_path, img)

    print(f'Prepared {len(images)} calibration images')


if __name__ == '__main__':
    prepare_calibration_images('data/train/', 'deploy/calibration/', num_images=1000)
```

### 4.3 生成量化表

```bash
# 生成量化校准表
./ncnn2table deploy/model_opt.param deploy/model_opt.bin deploy/calibration/ deploy/model.table

# 转换为 INT8 模型
./ncnn2int8 deploy/model_opt.param deploy/model_opt.bin \
            deploy/model_int8.param deploy/model_int8.bin \
            deploy/model.table
```

### 4.4 验证量化精度

```python
# validate_quantization.py
import cv2
import numpy as np

def compare_models(fp32_output, int8_output):
    """比较 FP32 和 INT8 模型输出"""
    # PSNR
    mse = np.mean((fp32_output - int8_output) ** 2)
    psnr = 10 * np.log10(255**2 / mse)

    # 相对误差
    relative_error = np.mean(np.abs(fp32_output - int8_output) / (np.abs(fp32_output) + 1e-6))

    print(f'PSNR: {psnr:.2f} dB')
    print(f'Relative Error: {relative_error*100:.2f}%')

    # 如果 PSNR > 40 dB 或相对误差 < 1%，量化成功
```

---

## 5. Android 集成

### 5.1 项目结构

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/
│   │   │   └── com/example/lowlight/
│   │   │       └── MainActivity.java
│   │   ├── cpp/
│   │   │   ├── CMakeLists.txt
│   │   │   └── lowlight_jni.cpp
│   │   └── assets/
│   │       ├── model_int8.param
│   │       └── model_int8.bin
│   └── build.gradle
└── ncnn-android-lib/      # NCNN Android 预编译库
```

### 5.2 JNI 接口

```cpp
// lowlight_jni.cpp
#include <jni.h>
#include <android/bitmap.h>
#include <ncnn/net.h>

static ncnn::Net net;
static bool net_loaded = false;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_lowlight_LowLightEnhancer_loadModel(
    JNIEnv *env, jobject thiz,
    jobject assetManager, jstring paramPath, jstring binPath) {

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);

    // 加载模型
    net.opt.use_vulkan_compute = false;  // CPU 推理
    net.opt.num_threads = 4;

    int ret = net.load_param(mgr, param);
    ret |= net.load_model(mgr, bin);

    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);

    net_loaded = (ret == 0);
    return net_loaded;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_lowlight_LowLightEnhancer_enhance(
    JNIEnv *env, jobject thiz, jintArray pixels, jint width, jint height) {

    if (!net_loaded) return nullptr;

    // 获取输入像素
    jint* input_pixels = env->GetIntArrayElements(pixels, nullptr);

    // 转换为 NCNN Mat (BGR)
    ncnn::Mat in = ncnn::Mat::from_pixels((unsigned char*)input_pixels,
                                          ncnn::Mat::PIXEL_RGBA2BGR, width, height);

    // 归一化
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 推理
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    // 后处理：反归一化
    // ...

    // 转换回像素
    ncnn::Mat out_rgb;
    out.to_pixels(out_rgb.data, ncnn::Mat::PIXEL_BGR2RGBA);

    // 返回结果
    jintArray result = env->NewIntArray(width * height);
    env->SetIntArrayRegion(result, 0, width * height, (jint*)out_rgb.data);

    env->ReleaseIntArrayElements(pixels, input_pixels, 0);

    return result;
}
```

### 5.3 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(lowlight_jni)

# NCNN
set(NCNN_DIR ${CMAKE_SOURCE_DIR}/../ncnn-android-lib/${ANDROID_ABI})
include_directories(${NCNN_DIR}/include)
link_directories(${NCNN_DIR}/lib)

# JNI
add_library(lowlight_jni SHARED lowlight_jni.cpp)

target_link_libraries(lowlight_jni
    ncnn
    android
    jnigraphics
    log
)
```

### 5.4 Java 接口

```java
// LowLightEnhancer.java
package com.example.lowlight;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class LowLightEnhancer {
    static {
        System.loadLibrary("lowlight_jni");
    }

    public native boolean loadModel(AssetManager assetManager,
                                    String paramPath, String binPath);
    public native int[] enhance(int[] pixels, int width, int height);

    public Bitmap enhance(Bitmap input) {
        int width = input.getWidth();
        int height = input.getHeight();

        int[] pixels = new int[width * height];
        input.getPixels(pixels, 0, width, 0, 0, width, height);

        int[] output = enhance(pixels, width, height);

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);

        return result;
    }
}
```

---

## 6. 常见问题与解决

### 6.1 问题排查表

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| "Op not supported" | ONNX 算子 NCNN 不支持 | 用 onnx-simplifier |
| 模型很大、速度没提升 | 没切换部署模式 | `model.switch_to_deploy()` |
| 导出后结构复杂 | RepVGG 多分支没合并 | 检查 switch_to_deploy |
| INT8 精度严重下降 | 校准集不足 | 用 1000+ 张图 |
| Android 崩溃 | 内存对齐/线程数 | 检查 num_threads |
| 输出全黑/全白 | 归一化参数错误 | 检查 mean/norm_vals |

### 6.2 onnx-simplifier 用法

```bash
# 基本用法
python -m onnxsim input.onnx output.onnx

# 跳过优化（如果某些优化导致问题）
python -m onnxsim input.onnx output.onnx --skip-fuse-bn

# 指定输入形状
python -m onnxsim input.onnx output.onnx --input-shape input:1,3,256,256
```

### 6.3 Netron 可视化

```bash
# 安装
pip install netron

# 使用
netron model.onnx
# 会在浏览器打开可视化界面
```

---

## 7. 面试要点

### 高频问题

**Q1: 为什么用 NCNN 而不是 TensorRT？**

> TensorRT 是 NVIDIA 的框架，只能在 NVIDIA GPU 上运行（服务器/PC）。
> 手机使用 ARM 架构 CPU + 高通/联发科/华为的 NPU，没有 NVIDIA 显卡。
> NCNN 是腾讯开源的移动端推理框架，专门针对 ARM NEON 指令集优化。

**Q2: INT8 量化的原理是什么？**

> 将 FP32 权重和激活值映射到 INT8 范围 [-128, 127]。
> 通过校准数据集统计激活值分布，确定缩放因子（scale）和零点（zero point）。
> 模型大小压缩 75%，推理速度提升 2-3 倍，精度损失通常 < 0.5%。

**Q3: 模型部署遇到过什么问题？**

> 1. "Op not supported" 错误 → 用 onnx-simplifier 简化模型
> 2. RepVGG 导出后速度没提升 → 导出前必须调用 switch_to_deploy()
> 3. INT8 精度下降严重 → 增加校准数据集到 1000+ 张

**Q4: NCNN 和 MNN 有什么区别？**

| 方面 | NCNN | MNN |
|------|------|-----|
| 公司 | 腾讯 | 阿里 |
| GPU 支持 | Vulkan（可选） | OpenCL/Metal/Vulkan |
| 依赖 | 无 | 无 |
| 文档 | 较少 | 较多 |
| 适用场景 | CPU 密集 | GPU 加速 |

**Q5: 如何展示你的部署能力？**

> 准备好：
> 1. 模型大小对比（FP32 vs INT8）
> 2. 推理速度对比（单位 ms/frame）
> 3. 两张 Netron 截图（训练时多分支 vs 部署时单分支）
> 4. Android Demo 视频（如果有）

---

## 附录：部署检查清单

- [ ] RepVGG 模型调用了 `switch_to_deploy()`
- [ ] ONNX 使用 opset_version=11
- [ ] 运行了 onnx-simplifier
- [ ] NCNN 转换成功（无 Op not supported 错误）
- [ ] INT8 校准数据 >= 1000 张
- [ ] 量化后精度损失 < 1%
- [ ] Android JNI 编译通过
- [ ] 手机端测试推理速度
