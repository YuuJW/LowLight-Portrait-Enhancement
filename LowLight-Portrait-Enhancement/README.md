# LowLight Portrait Enhancement

基于重参数化与移动端部署的暗光人像增强系统

## 项目结构

```
LowLight-Portrait-Enhancement/
├── isp/                      # ISP 基础模块
│   ├── blc.cpp              # 黑电平校正
│   ├── awb.cpp              # 自动白平衡
│   └── raw_reader.cpp       # RAW 数据读取
│
├── models/                   # 暗光增强网络
│   ├── unet_repvgg.py       # U-Net + RepVGG Block
│   ├── repvgg_block.py      # 重参数化模块
│   └── losses.py            # Face Parsing Loss + Perceptual Loss
│
├── train/                    # 训练相关
│   ├── train.py             # 训练脚本
│   ├── dataset.py           # 数据集加载
│   └── configs/             # 配置文件
│
├── deploy/                   # 移动端部署
│   ├── export_onnx.py       # PyTorch → ONNX
│   ├── onnx2ncnn.py         # ONNX → NCNN
│   ├── quantize.py          # INT8 量化
│   └── android/             # Android 项目
│
└── data/                     # 数据集
    ├── LOL/                 # Low-Light 数据集
    └── FFHQ_dark/           # 暗光人脸数据（合成）
```

## 技术栈

- **网络**: U-Net + RepVGG Block
- **损失**: L1 + Perceptual + Face Parsing Loss
- **部署**: NCNN + INT8 量化
- **平台**: Android (ARM)

## 进度

- [ ] ISP 基础模块
- [ ] 暗光增强网络
- [ ] 数据合成
- [ ] 模型训练
- [ ] NCNN 部署
- [ ] Android 集成
