"""
RetinexFormer Model Loader

Provides a simple interface for loading and running RetinexFormer model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from .archs import RetinexFormer


class RetinexFormerEnhancer:
    """
    RetinexFormer low-light image enhancer.

    Args:
        weights_path: Path to pretrained weights (.pth file)
        device: Device to run inference on ('cuda' or 'cpu')
        n_feat: Number of features (default: 40)
        stage: Number of stages (default: 1)
        num_blocks: Number of blocks per level (default: [1,2,2])
    """

    def __init__(
        self,
        weights_path: str,
        device: str = None,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks: list = [1, 2, 2]
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Create model
        self.model = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=n_feat,
            stage=stage,
            num_blocks=num_blocks
        )

        # Load weights
        self._load_weights(weights_path)

        # Set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weights_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=True)

    def _pad_to_multiple(self, img: torch.Tensor, factor: int = 4):
        """Pad image to be divisible by factor."""
        _, _, h, w = img.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        return img, h, w

    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a low-light image.

        Args:
            image: Input image as numpy array (H, W, C), BGR format, uint8 [0-255]

        Returns:
            Enhanced image as numpy array (H, W, C), BGR format, uint8 [0-255]
        """
        # BGR to RGB
        img_rgb = image[:, :, ::-1].copy()

        # HWC to CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Pad to multiple of 4
        img_padded, orig_h, orig_w = self._pad_to_multiple(img_tensor, factor=4)

        # Inference
        output = self.model(img_padded)

        # Remove padding
        output = output[:, :, :orig_h, :orig_w]

        # Clamp and convert back
        output = torch.clamp(output, 0, 1)
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255).astype(np.uint8)

        # RGB to BGR
        output_bgr = output[:, :, ::-1].copy()

        return output_bgr

    def get_model(self) -> RetinexFormer:
        """Get the underlying PyTorch model."""
        return self.model
