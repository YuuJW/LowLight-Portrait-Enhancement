"""
RepVGG Block 实现

核心重参数化模块，训练时使用多分支结构以提升精度，
推理时融合为单分支结构以提升速度。

参考论文: RepVGG: Making VGG-style ConvNets Great Again
         https://arxiv.org/abs/2101.03697
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RepVGGBlock(nn.Module):
    """
    RepVGG 结构重参数化模块

    训练模式: 3x3卷积 + 1x1卷积 + 恒等映射 (多分支)
    推理模式: 单个3x3卷积 + 偏置 (融合后)

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长 (默认: 1)
        padding: 3x3卷积的填充 (默认: 1)
        dilation: 卷积膨胀率 (默认: 1)
        groups: 分组卷积的组数 (默认: 1)
        deploy: 是否使用部署模式 (默认: False)
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 padding=1, dilation=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        if deploy:
            # 推理模式: 单个3x3卷积 + 偏置
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True
            )
        else:
            # 训练模式: 多分支结构

            # 分支1: 3x3卷积 + BN
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # 分支2: 1x1卷积 + BN
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # 分支3: 恒等映射 (仅当输入输出通道相等且步长为1时)
            if in_channels == out_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播"""
        if self.deploy:
            # 推理模式: 单分支
            return self.relu(self.rbr_reparam(x))

        # 训练模式: 多分支求和
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.relu(
            self.rbr_dense(x) +
            self.rbr_1x1(x) +
            id_out
        )

    def _fuse_bn_tensor(self, branch):
        """
        将 BatchNorm 参数融合到卷积权重中

        返回:
            kernel: 融合后的卷积核
            bias: 融合后的偏置
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # 恒等映射分支 (仅有 BN)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                # 创建恒等映射卷积核
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        # 将 BN 参数融合到卷积权重
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """将 1x1 卷积核填充为 3x3"""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def switch_to_deploy(self):
        """
        将多分支训练结构转换为单分支推理结构

        这是核心的重参数化步骤:
        1. 将每个分支的 BN 参数融合到卷积权重
        2. 将 1x1 卷积填充为 3x3
        3. 将所有分支的卷积核和偏置相加
        4. 创建单个卷积层并加载融合后的参数
        5. 删除训练分支以节省内存
        """
        if self.deploy:
            return

        # 从每个分支获取融合后的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # 将 1x1 卷积核填充为 3x3
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)

        # 将所有卷积核和偏置相加
        final_kernel = kernel3x3 + kernel1x1 + kernelid
        final_bias = bias3x3 + bias1x1 + biasid

        # 创建部署用的卷积层
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, bias=True
        )

        # 加载融合后的参数
        self.rbr_reparam.weight.data = final_kernel
        self.rbr_reparam.bias.data = final_bias

        # 删除训练分支以节省内存
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        self.deploy = True


if __name__ == '__main__':
    # 单元测试
    print("测试 RepVGG Block...")

    # 测试1: 训练模式前向传播
    block_train = RepVGGBlock(64, 128, stride=1, deploy=False)
    x = torch.randn(2, 64, 32, 32)
    out_train = block_train(x)
    print(f"训练模式输出形状: {out_train.shape}")
    assert out_train.shape == (2, 128, 32, 32), "训练模式形状不匹配"

    # 测试2: 切换到部署模式
    block_train.eval()
    with torch.no_grad():
        out_before = block_train(x)

    block_train.switch_to_deploy()

    with torch.no_grad():
        out_after = block_train(x)

    # 检查重参数化后输出是否一致
    diff = (out_before - out_after).abs().mean()
    print(f"重参数化后输出差异: {diff.item():.6f}")
    assert diff < 1e-4, f"重参数化误差过大: {diff.item()}"

    # 测试3: 直接创建部署模式
    block_deploy = RepVGGBlock(64, 128, stride=1, deploy=True)
    out_deploy = block_deploy(x)
    print(f"部署模式输出形状: {out_deploy.shape}")

    # 测试4: 检查参数量减少
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    block_before = RepVGGBlock(64, 128, deploy=False)
    params_before = count_params(block_before)
    block_before.switch_to_deploy()
    params_after = count_params(block_before)

    print(f"部署前参数量: {params_before}")
    print(f"部署后参数量: {params_after}")
    print(f"参数减少量: {params_before - params_after}")

    print("\n所有测试通过!")
