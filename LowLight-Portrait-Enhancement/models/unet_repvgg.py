"""
U-Net + RepVGG 网络

使用 RepVGG Block 替换所有卷积模块的 U-Net 架构，
以实现最优的训练精度和推理速度。

网络结构:
- 编码器: 3个 RepVGG Block + 最大池化
- 瓶颈层: 1个 RepVGG Block
- 解码器: 3个上采样 + 跳跃连接 + RepVGG Block
- 输出层: 1x1 卷积生成 RGB 图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .repvgg_block import RepVGGBlock


class UNetRepVGG(nn.Module):
    """
    使用 RepVGG Block 的 U-Net 网络，用于低光图像增强

    输入: 低光 RGB 图像 (B, 3, H, W)
    输出: 增强后的 RGB 图像 (B, 3, H, W)

    参数:
        in_channels: 输入通道数 (默认: 3 用于 RGB)
        out_channels: 输出通道数 (默认: 3 用于 RGB)
        base_channels: 基础通道数 (默认: 64)
        deploy: 是否使用部署模式
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, deploy=False):
        super(UNetRepVGG, self).__init__()

        self.deploy = deploy
        c = base_channels  # 基础通道数

        # 编码器 (下采样路径)
        self.enc1 = RepVGGBlock(in_channels, c, stride=1, deploy=deploy)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = RepVGGBlock(c, c * 2, stride=1, deploy=deploy)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = RepVGGBlock(c * 2, c * 4, stride=1, deploy=deploy)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = RepVGGBlock(c * 4, c * 8, stride=1, deploy=deploy)

        # 解码器 (上采样路径)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = RepVGGBlock(c * 8 + c * 4, c * 4, stride=1, deploy=deploy)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = RepVGGBlock(c * 4 + c * 2, c * 2, stride=1, deploy=deploy)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = RepVGGBlock(c * 2 + c, c, stride=1, deploy=deploy)

        # 输出层
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入低光图像 (B, 3, H, W)

        返回:
            增强后的图像 (B, 3, H, W)
        """
        # 编码器
        e1 = self.enc1(x)        # (B, 64, H, W)
        e1_pool = self.pool1(e1)  # (B, 64, H/2, W/2)

        e2 = self.enc2(e1_pool)   # (B, 128, H/2, W/2)
        e2_pool = self.pool2(e2)  # (B, 128, H/4, W/4)

        e3 = self.enc3(e2_pool)   # (B, 256, H/4, W/4)
        e3_pool = self.pool3(e3)  # (B, 256, H/8, W/8)

        # 瓶颈层
        b = self.bottleneck(e3_pool)  # (B, 512, H/8, W/8)

        # 解码器 + 跳跃连接
        d3 = self.up3(b)  # (B, 512, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512+256, H/4, W/4)
        d3 = self.dec3(d3)  # (B, 256, H/4, W/4)

        d2 = self.up2(d3)  # (B, 256, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256+128, H/2, W/2)
        d2 = self.dec2(d2)  # (B, 128, H/2, W/2)

        d1 = self.up1(d2)  # (B, 128, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 128+64, H, W)
        d1 = self.dec1(d1)  # (B, 64, H, W)

        # 输出
        out = self.out_conv(d1)  # (B, 3, H, W)

        # 残差连接: 输出 = 网络输出 + 输入
        out = out + x

        # 裁剪到 [0, 1] 范围
        out = torch.clamp(out, 0, 1)

        return out

    def switch_to_deploy(self):
        """
        将所有 RepVGG Block 切换到部署模式

        在导出 ONNX 前调用此方法可获得最佳推理速度
        """
        if self.deploy:
            return

        # 切换所有 RepVGG Block
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

        self.deploy = True
        print("已切换到部署模式: 所有 RepVGG Block 已融合")


if __name__ == '__main__':
    # 单元测试
    print("测试 U-Net + RepVGG...")

    # 测试1: 训练模式前向传播
    model_train = UNetRepVGG(in_channels=3, out_channels=3, base_channels=64, deploy=False)
    x = torch.randn(2, 3, 256, 256)

    print(f"输入形状: {x.shape}")

    out_train = model_train(x)
    print(f"训练模式输出形状: {out_train.shape}")
    assert out_train.shape == (2, 3, 256, 256), "输出形状不匹配"
    assert out_train.min() >= 0 and out_train.max() <= 1, "输出不在 [0, 1] 范围内"

    # 测试2: 参数量统计
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    params_train = count_params(model_train)
    print(f"训练模式参数量: {params_train:,}")

    # 测试3: 切换到部署模式
    model_train.eval()
    with torch.no_grad():
        out_before = model_train(x)

    model_train.switch_to_deploy()

    with torch.no_grad():
        out_after = model_train(x)

    # 检查重参数化后输出一致性
    diff = (out_before - out_after).abs().mean()
    print(f"部署后输出差异: {diff.item():.6f}")
    assert diff < 1e-4, f"部署模式误差过大: {diff.item()}"

    params_deploy = count_params(model_train)
    print(f"部署模式参数量: {params_deploy:,}")
    print(f"参数减少: {params_train - params_deploy:,} ({100*(params_train - params_deploy)/params_train:.1f}%)")

    # 测试4: 直接创建部署模式
    model_deploy = UNetRepVGG(in_channels=3, out_channels=3, base_channels=64, deploy=True)
    out_deploy = model_deploy(x)
    print(f"部署模式 (直接创建) 输出形状: {out_deploy.shape}")

    # 测试5: 不同输入尺寸
    for size in [128, 256, 512]:
        x_test = torch.randn(1, 3, size, size)
        out_test = model_deploy(x_test)
        assert out_test.shape == (1, 3, size, size), f"尺寸 {size} 测试失败"
    print("不同输入尺寸测试: 通过")

    # 测试6: 模型大小估算
    print(f"\n模型大小估算:")
    print(f"  32位浮点: {params_deploy * 4 / 1024 / 1024:.2f} MB")
    print(f"  16位浮点: {params_deploy * 2 / 1024 / 1024:.2f} MB")
    print(f"  8位整数: {params_deploy * 1 / 1024 / 1024:.2f} MB")

    print("\n所有测试通过!")