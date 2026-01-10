# Archive - 备份代码

此目录包含项目早期开发的自研模型代码，现已切换到使用预训练的 RetinexFormer 模型。

## 备份内容

### models/
- `repvgg_block.py` - RepVGG 重参数化模块实现
- `unet_repvgg.py` - U-Net + RepVGG 网络架构
- `losses.py` - L1 + Perceptual Loss 损失函数

## 为什么备份而不删除

1. 这些代码展示了对 RepVGG 重参数化原理的理解
2. 未来可能需要训练自定义模型时可以恢复使用
3. 面试时可以讨论这些设计决策

## 恢复使用

如果需要恢复使用这些模型：
```bash
cp archive/models/*.py models/
```

## 当前策略

项目现在使用 RetinexFormer (ICCV 2023) 预训练模型，专注于：
- 部署工程 (NCNN + Tiling + ThreadPool)
- INT8 量化优化
- 移动端性能调优
