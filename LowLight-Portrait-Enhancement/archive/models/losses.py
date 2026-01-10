"""
低光图像增强损失函数

实现:
1. L1 Loss - 像素级重建损失
2. Perceptual Loss - VGG 特征匹配，提升视觉质量
3. Combined Loss - 上述损失的加权组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    感知损失 (Perceptual Loss)，使用预训练 VGG16 特征

    从中间层提取特征，计算预测图像和目标图像特征之间的 MSE 损失

    参数:
        layer_weights: 不同 VGG 层的权重 (默认: {'relu3_3': 1.0})
        device: 运行 VGG 模型的设备 (默认: 自动检测)
    """

    def __init__(self, layer_weights=None, device=None):
        super(PerceptualLoss, self).__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if layer_weights is None:
            # 默认: 使用 relu3_3 层
            layer_weights = {'relu3_3': 1.0}
        self.layer_weights = layer_weights

        # 加载预训练 VGG16
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()

        # 冻结 VGG 参数
        for param in vgg.parameters():
            param.requires_grad = False

        # 提取到 relu3_3 的层 (索引 15)
        # VGG16 层索引:
        # relu1_2: 3, relu2_2: 8, relu3_3: 15, relu4_3: 22, relu5_3: 29
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }

        # 获取所需的最大层索引
        max_idx = max(self.layer_indices[k] for k in layer_weights.keys())
        self.vgg_layers = vgg[:max_idx + 1].to(device)

        # VGG 的归一化参数 (ImageNet 统计值)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """对图像进行 VGG 归一化 (ImageNet 归一化)"""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def extract_features(self, x):
        """
        从指定层提取 VGG 特征

        参数:
            x: 输入图像 (B, 3, H, W)，范围 [0, 1]

        返回:
            features: 特征字典 {层名: 特征张量}
        """
        x = self.normalize(x)
        features = {}

        for name, idx in self.layer_indices.items():
            if name in self.layer_weights:
                # 提取该层的特征
                feat = x
                for layer in self.vgg_layers[:idx + 1]:
                    feat = layer(feat)
                features[name] = feat

        return features

    def forward(self, pred, target):
        """
        计算感知损失

        参数:
            pred: 预测图像 (B, 3, H, W)
            target: 目标图像 (B, 3, H, W)

        返回:
            loss: 感知损失值
        """
        # 提取特征
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        # 计算加权损失
        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            loss += weight * F.mse_loss(
                pred_features[layer_name],
                target_features[layer_name]
            )

        return loss


class CombinedLoss(nn.Module):
    """
    低光增强组合损失函数

    Loss = lambda_l1 * L1_loss + lambda_p * Perceptual_loss

    参数:
        lambda_l1: L1 损失权重 (默认: 1.0)
        lambda_p: 感知损失权重 (默认: 0.1)
        vgg_layer: 用于感知损失的 VGG 层 (默认: 'relu3_3')
        device: 运行设备
    """

    def __init__(self, lambda_l1=1.0, lambda_p=0.1, vgg_layer='relu3_3', device=None):
        super(CombinedLoss, self).__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p

        # L1 损失
        self.l1_loss = nn.L1Loss()

        # 感知损失
        if lambda_p > 0:
            layer_weights = {vgg_layer: 1.0}
            self.perceptual_loss = PerceptualLoss(layer_weights=layer_weights, device=device)
        else:
            self.perceptual_loss = None

    def forward(self, pred, target):
        """
        计算组合损失

        参数:
            pred: 预测增强图像 (B, 3, H, W)
            target: 真实增强图像 (B, 3, H, W)

        返回:
            loss: 总损失
            loss_dict: 各损失分量的字典
        """
        # L1 损失
        loss_l1 = self.l1_loss(pred, target)

        # 总损失
        loss = self.lambda_l1 * loss_l1

        loss_dict = {
            'loss_l1': loss_l1.item(),
            'loss_total': loss.item()
        }

        # 感知损失
        if self.perceptual_loss is not None and self.lambda_p > 0:
            loss_p = self.perceptual_loss(pred, target)
            loss += self.lambda_p * loss_p
            loss_dict['loss_perceptual'] = loss_p.item()
            loss_dict['loss_total'] = loss.item()

        return loss, loss_dict


if __name__ == '__main__':
    # 单元测试
    print("测试损失函数...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 测试1: 感知损失
    print("\n1. 测试感知损失...")
    perceptual_loss = PerceptualLoss(layer_weights={'relu3_3': 1.0}, device=device)

    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)

    loss_p = perceptual_loss(pred, target)
    print(f"   感知损失: {loss_p.item():.6f}")
    assert loss_p.item() >= 0, "感知损失应为非负值"

    # 测试2: 相同图像的感知损失应接近0
    loss_p_same = perceptual_loss(pred, pred)
    print(f"   相同图像的感知损失: {loss_p_same.item():.6f}")
    assert loss_p_same.item() < 1e-5, "相同图像的感知损失应接近0"

    # 测试3: 组合损失
    print("\n2. 测试组合损失...")
    combined_loss = CombinedLoss(lambda_l1=1.0, lambda_p=0.1, device=device)

    pred = torch.rand(2, 3, 256, 256).to(device)  # 范围 [0, 1]
    target = torch.rand(2, 3, 256, 256).to(device)

    loss, loss_dict = combined_loss(pred, target)
    print(f"   总损失: {loss.item():.6f}")
    print(f"   损失分量: {loss_dict}")

    assert 'loss_l1' in loss_dict, "缺少 L1 损失"
    assert 'loss_perceptual' in loss_dict, "缺少感知损失"
    assert 'loss_total' in loss_dict, "缺少总损失"

    # 测试4: 相同图像的损失应接近0
    loss_same, loss_dict_same = combined_loss(pred, pred)
    print(f"\n   相同图像的组合损失: {loss_same.item():.6f}")
    print(f"   损失分量: {loss_dict_same}")
    assert loss_same.item() < 1e-5, "相同图像的损失应接近0"

    # 测试5: 仅 L1 损失 (lambda_p = 0)
    print("\n3. 测试仅 L1 损失...")
    l1_only_loss = CombinedLoss(lambda_l1=1.0, lambda_p=0.0, device=device)
    loss_l1, loss_dict_l1 = l1_only_loss(pred, target)
    print(f"   L1 损失: {loss_l1.item():.6f}")
    assert 'loss_perceptual' not in loss_dict_l1, "不应包含感知损失"

    # 测试6: 多层感知损失
    print("\n4. 测试多层感知损失...")
    multi_layer_loss = PerceptualLoss(
        layer_weights={'relu2_2': 0.5, 'relu3_3': 1.0, 'relu4_3': 0.5},
        device=device
    )
    loss_multi = multi_layer_loss(pred, target)
    print(f"   多层感知损失: {loss_multi.item():.6f}")

    # 测试7: 梯度流
    print("\n5. 测试梯度流...")
    pred = torch.rand(1, 3, 128, 128, requires_grad=True).to(device)
    target = torch.rand(1, 3, 128, 128).to(device)

    loss, _ = combined_loss(pred, target)
    loss.backward()

    assert pred.grad is not None, "梯度应该通过损失传播"
    print(f"   梯度范数: {pred.grad.norm().item():.6f}")

    print("\n所有测试通过!")