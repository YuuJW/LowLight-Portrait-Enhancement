# ONNX 导出和验证工具使用指南

本目录包含完整的 ONNX 模型导出、验证和测试工具链。

## 📁 文件说明

| 文件 | 功能 |
|------|------|
| `export_onnx.py` | PyTorch → ONNX 导出工具 |
| `verify_onnx.py` | ONNX 模型详细验证工具 |
| `batch_test_onnx.py` | 批量测试工具 |

## 🚀 快速开始

### 1. 导出 ONNX 模型

**基础导出：**
```bash
python deploy/export_onnx.py \
    --weights models/LOL_v2_synthetic.pth \
    --output deploy/models/retinexformer.onnx
```

**完整导出（包含简化和验证）：**
```bash
python deploy/export_onnx.py \
    --weights models/LOL_v2_synthetic.pth \
    --output deploy/models/retinexformer.onnx \
    --simplify \
    --verify \
    --check-model \
    --show-info
```

**参数说明：**
- `--weights`: PyTorch 权重文件路径
- `--output`: 输出 ONNX 文件路径
- `--input-size`: 输入尺寸，默认 512 512
- `--opset`: ONNX opset 版本，默认 11
- `--simplify`: 使用 onnx-simplifier 简化模型
- `--verify`: 验证 PyTorch vs ONNX 输出一致性
- `--check-model`: 检查 ONNX 模型有效性
- `--show-info`: 显示模型详细信息

### 2. 详细验证 ONNX 模型

**数值一致性验证：**
```bash
python deploy/verify_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --weights models/LOL_v2_synthetic.pth \
    --num-tests 10
```

**性能基准测试：**
```bash
python deploy/verify_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --weights models/LOL_v2_synthetic.pth \
    --benchmark
```

**可视化对比：**
```bash
python deploy/verify_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --weights models/LOL_v2_synthetic.pth \
    --image data/test.png \
    --output-dir deploy/verification
```

**完整验证：**
```bash
python deploy/verify_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --weights models/LOL_v2_synthetic.pth \
    --image data/test.png \
    --num-tests 20 \
    --benchmark \
    --output-dir deploy/verification
```

**参数说明：**
- `--onnx`: ONNX 模型路径（必需）
- `--weights`: PyTorch 权重路径
- `--image`: 测试图像路径（可选）
- `--output-dir`: 输出目录，默认 deploy/verification
- `--num-tests`: 随机输入测试次数，默认 10
- `--benchmark`: 运行性能基准测试

### 3. 批量测试

**在多张图像上测试：**
```bash
python deploy/batch_test_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --weights models/LOL_v2_synthetic.pth \
    --image-dir data/LOL/eval15/low \
    --output deploy/batch_test_report.json
```

**限制测试图像数量：**
```bash
python deploy/batch_test_onnx.py \
    --onnx deploy/models/retinexformer.onnx \
    --image-dir data/LOL/eval15/low \
    --max-images 10
```

**参数说明：**
- `--onnx`: ONNX 模型路径（必需）
- `--weights`: PyTorch 权重路径
- `--image-dir`: 测试图像目录（必需）
- `--output`: 输出报告路径
- `--max-images`: 最大测试图像数

## 📊 输出说明

### export_onnx.py 输出

```
Weights: models/LOL_v2_synthetic.pth
Output: deploy/models/retinexformer.onnx

Loading model...

Exporting to ONNX...
Exporting ONNX with input size: 512x512
Opset version: 11
Exported: deploy/models/retinexformer.onnx

Simplifying ONNX model...
Simplified: deploy/models/retinexformer.onnx

Verification:
  Mean diff: 0.000012
  Max diff: 0.000234
  Status: ✓ PASSED

Checking ONNX model...
  Status: ✓ Model is valid

============================================================
ONNX Model Information
============================================================

Producer: pytorch 2.0.0
IR Version: 8
Opset Version: 11

Graph Name: torch_jit

Inputs (1):
  - input: [1, 3, 512, 512] (type: 1)

Outputs (1):
  - output: [1, 3, 512, 512] (type: 1)

Nodes: 245

Operator Statistics:
  Conv: 48
  Add: 32
  Mul: 28
  ...

Model Size: 15.23 MB
============================================================

Done!
```

### verify_onnx.py 输出

```
============================================================
数值一致性验证
============================================================
Test 1/10: Mean diff = 0.000012, Max diff = 0.000234
Test 2/10: Mean diff = 0.000015, Max diff = 0.000198
...
Test 10/10: Mean diff = 0.000011, Max diff = 0.000245

============================================================
通过率: 10/10 (100.0%)
平均 Mean Diff: 0.000013
平均 Max Diff: 0.000215
============================================================

============================================================
性能基准测试
============================================================
Warming up...

PyTorch 推理测试 (50 次)...
ONNX Runtime 推理测试 (50 次)...

============================================================
PyTorch:      125.34 ± 5.23 ms
ONNX Runtime: 98.76 ± 3.45 ms
加速比:       1.27x
============================================================

============================================================
可视化对比
============================================================
Image size: 1024x768
PyTorch 推理...
ONNX Runtime 推理...

差异统计:
  Mean: 0.0234
  Max:  1.2345
  Std:  0.0456

对比图已保存到: deploy/verification/comparison.png
```

### batch_test_onnx.py 输出

```
Found 15 images

Testing images...
100%|████████████████████| 15/15 [00:45<00:00,  3.02s/it]

============================================================
测试报告
============================================================
测试图像数: 15
平均 Mean Diff: 0.000234
平均 Max Diff: 0.001234
最大 Mean Diff: 0.000456
最大 Max Diff: 0.002345
通过率: 15/15 (100.0%)
平均加速比: 1.25x

报告已保存到: deploy/batch_test_report.json
============================================================
```

## 🔍 验证标准

### 数值一致性
- **Mean Diff < 1e-4**: 通过 ✓
- **Mean Diff >= 1e-4**: 失败 ✗

### 性能要求
- ONNX Runtime 应该比 PyTorch 快或相当
- 典型加速比：1.2x - 1.5x

## 📝 常见问题

### Q1: 导出失败，提示 "No module named 'onnx'"
```bash
pip install onnx onnxruntime onnxsim
```

### Q2: 验证失败，差异过大
可能原因：
1. 模型权重不匹配
2. ONNX opset 版本不兼容
3. 导出时使用了不支持的操作

解决方案：
```bash
# 尝试不同的 opset 版本
python deploy/export_onnx.py --opset 12 --verify
```

### Q3: 性能测试显示 ONNX 更慢
可能原因：
1. 使用了 CPU 而非 GPU
2. 未进行 warmup
3. 模型未优化

解决方案：
```bash
# 确保使用简化后的模型
python deploy/export_onnx.py --simplify
```

## 🎯 最佳实践

1. **总是使用 --simplify**
   - 减小模型大小
   - 提升推理速度
   - 移除冗余操作

2. **总是使用 --verify**
   - 确保数值一致性
   - 及早发现问题

3. **使用批量测试**
   - 在真实数据上验证
   - 统计平均性能

4. **保存验证报告**
   - 记录模型版本
   - 追踪性能变化

## 📚 相关文档

- [ONNX 官方文档](https://onnx.ai/)
- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
