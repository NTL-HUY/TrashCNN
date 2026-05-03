"""
  1. Cập nhật SimpleBackbone chết
  2. Backbone mới: ResNet-like với 4 stage (C2–C5), deeper và đúng hơn
  3. Thêm Feature Pyramid Network (FPN) → detect multi-scale objects
  4. Kaiming initialization đúng chuẩn cho tất cả Conv / BN
  5. Anchor sizes khớp với FPN levels (P2–P6)
  6. Tách build_model / build_inference_model (threshold khác nhau)
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)


# ─────────────────────────── Building blocks ────────────────────────────

class ResidualBlock(nn.Module):
    """
    Basic residual block (2 × 3×3 conv).
    Shortcut = 1×1 conv khi stride != 1 hoặc channels thay đổi.
    bias=False vì BatchNorm đã xử lý bias.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut để match shape
        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


def _make_layer(in_channels: int, out_channels: int,
                num_blocks: int, stride: int = 1) -> nn.Sequential:
    """Tạo 1 stage gồm nhiều ResidualBlock, chỉ block đầu dùng stride."""
    blocks = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(1, num_blocks):
        blocks.append(ResidualBlock(out_channels, out_channels, 1))
    return nn.Sequential(*blocks)


# ──────────────────────── Custom Backbone + FPN ─────────────────────────

class SimpleBackbone(nn.Module):
    """
    Architecture:
        Stem   : 7×7 conv, BN, ReLU, MaxPool  → /4  (64ch)
        Layer1 : 2× ResBlock(64  → 128, stride=1) → /4  (C2)
        Layer2 : 2× ResBlock(128 → 256, stride=2) → /8  (C3)
        Layer3 : 3× ResBlock(256 → 512, stride=2) → /16 (C4)
        Layer4 : 2× ResBlock(512 → 512, stride=2) → /32 (C5)
        FPN    : lateral projections + top-down → 256ch mỗi level
                 LastLevelMaxPool thêm P6 cho object to

    Tại sao FPN?
        - Detect được vật nhỏ (P2, P3) lẫn vật lớn (P5, P6)
        - Backbone cũ chỉ trả 1 feature map → blind với nhiều scale
    """

    FPN_OUT_CHANNELS = 256

    def __init__(self):
        super().__init__()

        # Stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Backbone stages
        self.layer1 = _make_layer(64, 128, num_blocks=2, stride=1)  # C2 /4
        self.layer2 = _make_layer(128, 256, num_blocks=2, stride=2)  # C3 /8
        self.layer3 = _make_layer(256, 512, num_blocks=3, stride=2)  # C4 /16
        self.layer4 = _make_layer(512, 512, num_blocks=2, stride=2)  # C5 /32

        # FPN: nhận C2–C5, output channels đều = FPN_OUT_CHANNELS
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 512],
            out_channels=self.FPN_OUT_CHANNELS,
            extra_blocks=LastLevelMaxPool(),  # thêm P6 = MaxPool(P5)
        )

        # FasterRCNN check attribute này
        self.out_channels = self.FPN_OUT_CHANNELS

        self._init_weights()

    # ── Weight initialization ──────────────────────────────────────────
    def _init_weights(self):
        """
        Kaiming Normal cho Conv (fan_out, relu) → tránh vanishing/exploding gradient
        BN weight=1, bias=0 (standard)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        x = self.stem(x)

        c2 = self.layer1(x)  # stride /4
        c3 = self.layer2(c2)  # stride /8
        c4 = self.layer3(c3)  # stride /16
        c5 = self.layer4(c4)  # stride /32

        # FPN nhận OrderedDict; key phải là string
        feat_maps = OrderedDict([
            ("0", c2),
            ("1", c3),
            ("2", c4),
            ("3", c5),
        ])
        return self.fpn(feat_maps)  # → {"0":P2, "1":P3, "2":P4, "3":P5, "pool":P6}


# ─────────────────────────── Build helpers ──────────────────────────────

def _make_anchor_generator() -> AnchorGenerator:
    """
    5 FPN levels (P2–P6) → 5 anchor size tuples.
    Sizes tăng dần theo từng level.
    aspect_ratios: 0.5 (tall), 1.0 (square), 2.0 (wide)
    """
    sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def build_model(num_classes: int = 7) -> FasterRCNN:
    """
    Build model để TRAIN.
    box_score_thresh thấp (0.05) để loss không bị thiếu proposal.
    """
    backbone = SimpleBackbone()
    anchor_generator = _make_anchor_generator()

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # --- RPN hyperparams ---
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        # --- ROI head ---
        box_score_thresh=0.05,  # thấp khi train; raise lúc deploy
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        # --- Image size ---
        min_size=416,
        max_size=640,
    )
    return model


def build_inference_model(num_classes: int = 7,
                          score_thresh: float = 0.4) -> FasterRCNN:
    """
    Build model để INFERENCE / DEPLOY.
    score_thresh cao hơn để lọc bớt false positive.
    """
    model = build_model(num_classes)
    model.roi_heads.score_thresh = score_thresh
    return model
