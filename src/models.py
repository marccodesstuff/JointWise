"""
Custom FPN backbones using timm pretrained models.

This module defines Feature Pyramid Network (FPN) architectures with various
backbone options (Xception, DenseNet, ResNeXt, EfficientNet) for YOLO detection.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required. Install with: pip install timm")

from ultralytics.nn import tasks as ytasks

from .config import NUM_CLASSES


# =============================================================================
# Helper Modules
# =============================================================================
class ConvBNAct(nn.Sequential):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, 
                 p: int = None, act: bool = True):
        if p is None:
            p = (k - 1) // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch)
        ]
        if act:
            layers.append(nn.SiLU(inplace=False))
        super().__init__(*layers)


def _ensure_no_inplace(module: nn.Module) -> None:
    """Disable inplace operations for gradient compatibility."""
    if isinstance(module, (nn.SiLU, nn.ReLU)):
        module.inplace = False


# =============================================================================
# Feature Pyramid Network
# =============================================================================
class FPN(nn.Module):
    """
    Feature Pyramid Network neck for multi-scale feature fusion.
    
    Takes features from 3 backbone stages (C3, C4, C5) and produces
    pyramid features (P3, P4, P5) at different scales.
    """
    
    def __init__(self, c3: int, c4: int, c5: int, out: int = 256):
        super().__init__()
        # Lateral connections
        self.l3 = ConvBNAct(c3, out, k=1, act=False)
        self.l4 = ConvBNAct(c4, out, k=1, act=False)
        self.l5 = ConvBNAct(c5, out, k=1, act=False)
        
        # Output convolutions
        self.o3 = ConvBNAct(out, out, k=3)
        self.o4 = ConvBNAct(out, out, k=3)
        self.o5 = ConvBNAct(out, out, k=3)

    def forward(self, c3, c4, c5):
        # Top-down pathway with lateral connections
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        
        return [self.o3(p3), self.o4(p4), self.o5(p5)]


# =============================================================================
# Timm Feature Extractor Factory
# =============================================================================
def create_timm_features(name: str, in_chans: int = 1, pretrained: bool = True):
    """
    Create a timm model configured for feature extraction.
    
    Args:
        name: timm model name (e.g., 'xception41', 'densenet121')
        in_chans: Number of input channels (1 for grayscale)
        pretrained: Whether to use pretrained weights
    
    Returns:
        Feature extraction model with outputs at indices (2, 3, 4)
    """
    model = timm.create_model(
        name,
        features_only=True,
        in_chans=in_chans,
        pretrained=pretrained,
        out_indices=(2, 3, 4),
        act_layer=nn.SiLU
    )
    # Ensure no inplace operations
    for mod in model.modules():
        _ensure_no_inplace(mod)
    return model


# =============================================================================
# FPN Backbone Classes
# =============================================================================
class BaseFPNBackbone(nn.Module):
    """Base class for FPN backbones with common functionality."""
    
    def __init__(self, backbone_name: str, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        self.backbone = create_timm_features(backbone_name, in_chans=1, pretrained=pretrained)
        c3, c4, c5 = self.backbone.feature_info.channels()
        self.neck = FPN(c3, c4, c5, out_channels)
        self.c2 = [out_channels] * 3

    def forward(self, x):
        # Convert to single channel if needed
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)
        
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3.contiguous(), c4.contiguous(), c5.contiguous())
        return [p3, p4, p5]


class XceptionFPN(BaseFPNBackbone):
    """Xception41 backbone with FPN neck."""
    
    def __init__(self, name: str = "xception41", pretrained: bool = True, out_channels: int = 256):
        super().__init__(name, pretrained, out_channels)


class DenseNetFPN(BaseFPNBackbone):
    """DenseNet121 backbone with FPN neck."""
    
    def __init__(self, name: str = "densenet121", pretrained: bool = True, out_channels: int = 256):
        super().__init__(name, pretrained, out_channels)


class ResNeXtFPN(BaseFPNBackbone):
    """ResNeXt50 backbone with FPN neck."""
    
    def __init__(self, name: str = "resnext50_32x4d", pretrained: bool = True, out_channels: int = 256):
        super().__init__(name, pretrained, out_channels)


class EfficientNetFPN(BaseFPNBackbone):
    """EfficientNet-B0 backbone with FPN neck."""
    
    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True, out_channels: int = 256):
        super().__init__(name, pretrained, out_channels)


# =============================================================================
# Index/Take Module for Ultralytics
# =============================================================================
class Take(nn.Module):
    """
    Select a specific output from a list of tensors.
    
    Used in YOLO model YAML configs to extract specific FPN outputs.
    Handles various YAML argument styles for compatibility.
    """
    
    def __init__(self, c1=None, c2=None, i=0, *args, **kwargs):
        super().__init__()
        
        # Parse index from various argument formats
        if 'i' in kwargs:
            self.i = int(kwargs['i'])
        elif len(args) > 0:
            try:
                self.i = int(args[-1])
            except Exception:
                self.i = int(i) if i is not None else 0
        else:
            self.i = int(i) if i is not None else 0
        
        # Determine output channels
        out_ch = None
        if isinstance(c1, (list, tuple)) and len(c1) > 0:
            idx = min(max(0, self.i), len(c1) - 1)
            try:
                out_ch = int(c1[idx])
            except Exception:
                out_ch = None
        elif isinstance(c1, (int, float)) and c1:
            out_ch = int(c1)
        
        self.c2 = out_ch if (out_ch and out_ch > 0) else 256
        self._warned = False

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"Take[{self.i}] expects list/tuple, got {type(x)}")
        
        if self.i >= len(x):
            if not self._warned:
                print(f"[Take] Index {self.i} > {len(x)-1}; clamping to last element")
                self._warned = True
            return x[-1]
        
        return x[self.i]


class Index(Take):
    """Alias for Take module."""
    pass


# =============================================================================
# Model YAML Definitions
# =============================================================================
def get_model_yaml(backbone_name: str) -> str:
    """
    Generate YOLO model YAML configuration for a given backbone.
    
    Args:
        backbone_name: One of 'xception', 'resnext', 'densenet', 'efficientnet'
    
    Returns:
        YAML configuration string
    """
    backbone_map = {
        "xception": "XceptionFPN",
        "resnext": "ResNeXtFPN",
        "densenet": "DenseNetFPN",
        "efficientnet": "EfficientNetFPN",
    }
    
    if backbone_name not in backbone_map:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(backbone_map.keys())}")
    
    fpn_class = backbone_map[backbone_name]
    
    return (
        "task: detect\n"
        f"nc: {NUM_CLASSES}\n"
        "ch: 1\n"
        "backbone:\n"
        f"  - [-1, 1, {fpn_class}, []]\n"
        "  - [0, 1, Index, [256, 0]]\n"
        "  - [0, 1, Index, [256, 1]]\n"
        "  - [0, 1, Index, [256, 2]]\n"
        "head:\n"
        "  - [[1, 2, 3], 1, Detect, [nc]]\n"
    )


# Pre-defined YAML configurations for convenience
XCEPTION_YAML = get_model_yaml("xception")
RESNEXT_YAML = get_model_yaml("resnext")
DENSENET_YAML = get_model_yaml("densenet")
EFFICIENTNET_YAML = get_model_yaml("efficientnet")


# =============================================================================
# Register Custom Modules with Ultralytics
# =============================================================================
def register_custom_modules():
    """Register custom modules with Ultralytics task registry."""
    ytasks.XceptionFPN = XceptionFPN
    ytasks.DenseNetFPN = DenseNetFPN
    ytasks.ResNeXtFPN = ResNeXtFPN
    ytasks.EfficientNetFPN = EfficientNetFPN
    ytasks.Take = Take
    ytasks.Index = Index


# Auto-register on import
register_custom_modules()
