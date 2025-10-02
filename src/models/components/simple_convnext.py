"""
Simplified ConvNeXt-V2 implementation for canopy height estimation.
No timm dependency - pure PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class SimpleConvNeXt(nn.Module):
    """Simplified ConvNeXt for canopy height estimation."""
    
    def __init__(self, num_channels: int = 3, num_classes: int = 1):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=4, stride=4),
            LayerNorm2d(96)
        )
        
        # Stages
        self.stages = nn.ModuleList([
            ConvNeXtStage(96, 96, depth=3),      # Stage 1
            ConvNeXtStage(96, 192, depth=3),     # Stage 2  
            ConvNeXtStage(192, 384, depth=9),    # Stage 3
            ConvNeXtStage(384, 768, depth=3),    # Stage 4
        ])
        
        # Head for height regression
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            LayerNorm2d(768),
            nn.Linear(768, num_classes),
            nn.ReLU()  # Ensure positive height values
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, metas=None) -> torch.Tensor:
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.head(x)
        
        # Reshape to match expected output [B, 1, H, W]
        B = x.shape[0]
        # Use bilinear upsampling to match input resolution
        x = x.view(B, 1, 1, 1)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        return x


def create_simple_convnext_model(num_channels: int = 3, num_classes: int = 1, **kwargs):
    """Create simplified ConvNeXt model."""
    return SimpleConvNeXt(num_channels=num_channels, num_classes=num_classes)


if __name__ == "__main__":
    # Test the model
    model = create_simple_convnext_model(num_channels=3)
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")