"""
Low-Light Portrait Enhancement Models

This package contains the neural network architectures for low-light image enhancement.

Current Strategy: Using pre-trained RetinexFormer (ICCV 2023)
- Retinex-based decomposition + Transformer architecture
- Pre-trained weights available from official repository
- Focus on deployment engineering (NCNN, quantization, mobile optimization)

Archived Models (in archive/models/):
- RepVGGBlock: Reparameterization module (for future custom training)
- UNetRepVGG: U-Net + RepVGG backbone
- CombinedLoss: L1 + Perceptual loss functions
"""

# RetinexFormer model will be imported after creation
# from .retinexformer import RetinexFormer

__all__ = []  # Will be updated when RetinexFormer is implemented
