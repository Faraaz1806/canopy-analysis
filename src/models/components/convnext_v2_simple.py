"""
Simplified ConvNeXt-V2 implementation for canopy height estimation.
Pure PyTorch implementation without heavy dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from src.models.components.utils import (
    SimpleSegmentationHead,
    infer_output,
    set_first_layer,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps."""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise conv and feedforward."""
    
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout2d(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.drop_path(x)
        x = input + x
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt stage with multiple blocks."""
    
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        
        # Downsampling layer
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        
        # ConvNeXt blocks
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(out_channels) for _ in range(depth)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization from ConvNeXt-V2.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        
        # Transpose for broadcasting
        x_t = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_t = self.gamma * Nx.permute(0, 2, 3, 1) * x_t + self.beta + x_t
        return x_t.permute(0, 3, 1, 2)  # Back to [B, C, H, W]


class CanopyHeightHead(nn.Module):
    """
    Regression head for canopy height estimation.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        dropout_rate: float = 0.1,
        img_size: int = 512,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Progressive upsampling
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Final regression
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final_conv = nn.Conv2d(128, num_classes, 1)
        
        # Activation for height (ensure positive values)
        self.height_activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get input size to maintain aspect ratio
        input_size = x.shape[-2:]  # [H, W]
        
        # Progressive upsampling
        x = self.relu1(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Final upsampling to match original input resolution
        # Calculate target size based on stride reduction (4x4 stem + 2x downsamples)
        target_size = (self.img_size, self.img_size)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Regression
        x = self.dropout(x)
        x = self.final_conv(x)
        
        # Ensure positive height values
        x = self.height_activation(x)
        
        return x


class ConvNeXtV2(nn.Module):
    """
    Simplified ConvNeXt-V2 for canopy height estimation.
    Pure PyTorch implementation without heavy dependencies.
    """
    
    def __init__(
        self,
        backbone: str = "convnextv2_base",  # Ignored - using our implementation
        num_classes: int = 1,
        num_channels: int = 4,  # RGBI for canopy data
        pretrained: bool = True,  # Ignored for now
        img_size: int = 512,
        use_fpn: bool = False,  # Simplified - no FPN
        fpn_channels: int = 256,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_fpn = use_fpn
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=4, stride=4),
            LayerNorm2d(96)
        )
        
        # Stages - ConvNeXt-V2 Base configuration (simplified)
        self.stages = nn.ModuleList([
            ConvNeXtStage(96, 96, depth=3),      # Stage 0
            ConvNeXtStage(96, 192, depth=3),     # Stage 1  
            ConvNeXtStage(192, 384, depth=9),    # Stage 2 (simplified from 27)
            ConvNeXtStage(384, 768, depth=3),    # Stage 3
        ])
        
        # Global Response Normalization
        self.grn = GlobalResponseNorm(768)  # Final stage output channels
        
        # Regression head for canopy height
        self.regression_head = CanopyHeightHead(
            in_channels=768,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            img_size=img_size
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        log.info(f"ConvNeXt-V2 initialized: {num_classes} classes, {num_channels} channels")
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, metas=None) -> torch.Tensor:
        """
        Forward pass for canopy height estimation.
        
        Args:
            x: Input tensor [B, C, H, W]
            metas: Metadata (not used but required for compatibility)
            
        Returns:
            Height predictions [B, num_classes, H, W]
        """
        # Store original input size
        original_size = x.shape[-2:]  # [H, W]
        
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Global Response Normalization
        x = self.grn(x)
        
        # Height regression (will upsample to original input size)
        height_pred = self.regression_head(x)
        
        # Ensure output matches input size exactly
        if height_pred.shape[-2:] != original_size:
            height_pred = F.interpolate(
                height_pred, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return height_pred


def create_convnext_v2_model(variant: str = "base", num_channels: int = 3, pretrained: bool = True, **kwargs):
    """Create ConvNeXt-V2 model without timm dependency."""
    
    return ConvNeXtV2(
        backbone=f"convnextv2_{variant}",
        num_channels=num_channels,
        pretrained=pretrained,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    model = create_convnext_v2_model("base", num_channels=3)
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")