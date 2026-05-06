"""
model.py - Faster R-CNN with ResNet-50 + FPN (trained from scratch, no pretrained weights)
Architecture:
  Backbone  : ResNet-50 with Batch Normalization
  Neck      : Feature Pyramid Network (FPN)
  RPN       : Region Proposal Network
  Head      : RoI Align + Two-stage classifier/regressor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    FeaturePyramidNetwork,
    MultiScaleRoIAlign,
    box_iou,
    clip_boxes_to_image,
    nms,
    batched_nms,
    remove_small_boxes,
)
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import FasterRCNN
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import math


# ─────────────────────────────────────────────
# ResNet-50 Backbone (from scratch)
# ─────────────────────────────────────────────
class Bottleneck(nn.Module):
    """ResNet Bottleneck block (3-layer: 1x1, 3x3, 1x1 conv)."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet-50 backbone returning multi-scale feature maps for FPN.
    Returns: {"0": C2, "1": C3, "2": C4, "3": C5}
    Strides  :  4       8      16      32
    """
    LAYERS = [3, 4, 6, 3]  # ResNet-50 configuration

    def __init__(self):
        super().__init__()
        # Stem
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64,  64,  self.LAYERS[0], stride=1)  # → C2, stride=4
        self.layer2 = self._make_layer(256, 128, self.LAYERS[1], stride=2)  # → C3, stride=8
        self.layer3 = self._make_layer(512, 256, self.LAYERS[2], stride=2)  # → C4, stride=16
        self.layer4 = self._make_layer(1024,512, self.LAYERS[3], stride=2)  # → C5, stride=32

        self._init_weights()

    def _make_layer(self, in_channels, mid_channels, num_blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(in_channels, mid_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)])

    @property
    def out_channels(self):
        """Output channel sizes for each feature level."""
        return {"0": 256, "1": 512, "2": 1024, "3": 2048}


# ─────────────────────────────────────────────
# ResNet50 + FPN Backbone
# ─────────────────────────────────────────────
class ResNet50FPN(nn.Module):
    """
    ResNet-50 + Feature Pyramid Network.
    Returns 5 feature levels: P2, P3, P4, P5, P6 (via extra pooling)
    All at 256 channels.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.body = ResNet50()
        self.fpn  = FeaturePyramidNetwork(
            in_channels_list=list(self.body.out_channels.values()),  # [256,512,1024,2048]
            out_channels=out_channels,
            extra_blocks=None,
        )
        self.out_channels = out_channels

        # Extra level P6 via max-pooling on P5 (for large anchors)
        self.extra_pool = nn.MaxPool2d(1, stride=2)

        self._init_fpn()

    def _init_fpn(self):
        for m in self.fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.body(x)
        pyramid  = self.fpn(features)            # {"0":P2,"1":P3,"2":P4,"3":P5}
        pyramid["pool"] = self.extra_pool(pyramid["3"])  # P6
        return pyramid


# ─────────────────────────────────────────────
# Box Predictor Head
# ─────────────────────────────────────────────
class FastRCNNPredictor(nn.Module):
    """Two FC layers → class logits + bbox regression."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)


class TwoMLPHead(nn.Module):
    """Standard 2-layer MLP head after RoI pooling."""
    def __init__(self, in_channels: int, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc6.weight, a=1)
        nn.init.kaiming_uniform_(self.fc7.weight, a=1)
        nn.init.zeros_(self.fc6.bias)
        nn.init.zeros_(self.fc7.bias)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


# ─────────────────────────────────────────────
# Build Faster R-CNN
# ─────────────────────────────────────────────
def build_faster_rcnn(
    num_classes: int = 6,
    # Image transform
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    # RPN
    rpn_anchor_sizes: Optional[Tuple] = None,
    rpn_aspect_ratios: Optional[Tuple] = None,
    rpn_fg_iou_thresh: float = 0.7,
    rpn_bg_iou_thresh: float = 0.3,
    rpn_batch_size_per_image: int = 256,
    rpn_positive_fraction: float = 0.5,
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    rpn_score_thresh: float = 0.0,
    # RoI
    box_roi_pool_output_size: int = 7,
    box_representation_size: int = 1024,
    box_fg_iou_thresh: float = 0.5,
    box_bg_iou_thresh: float = 0.5,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    # Regression weights
    box_bbox_reg_weights: Optional[Tuple[float, ...]] = None,
) -> FasterRCNN:
    """
    Builds a Faster R-CNN model with ResNet-50 + FPN backbone.
    All weights are initialized from scratch (no pretrained weights).
    """
    # ── Backbone ──
    backbone = ResNet50FPN(out_channels=256)

    # ── Anchors ──
    if rpn_anchor_sizes is None:
        rpn_anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    if rpn_aspect_ratios is None:
        rpn_aspect_ratios = ((0.5, 1.0, 2.0),) * len(rpn_anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=rpn_anchor_sizes,
        aspect_ratios=rpn_aspect_ratios,
    )

    # ── RPN Head ──
    rpn_head = RPNHead(
        in_channels=backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
    )

    # ── RPN ──
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=rpn_fg_iou_thresh,
        bg_iou_thresh=rpn_bg_iou_thresh,
        batch_size_per_image=rpn_batch_size_per_image,
        positive_fraction=rpn_positive_fraction,
        pre_nms_top_n={"training": rpn_pre_nms_top_n_train, "testing": rpn_pre_nms_top_n_test},
        post_nms_top_n={"training": rpn_post_nms_top_n_train, "testing": rpn_post_nms_top_n_test},
        nms_thresh=rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )

    # ── RoI Pooling ──
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=box_roi_pool_output_size,
        sampling_ratio=2,
    )

    # ── Box Head ──
    resolution = box_roi_pool_output_size
    in_channels = backbone.out_channels * resolution ** 2  # 256 * 7 * 7 = 12544
    box_head = TwoMLPHead(
        in_channels=in_channels,
        representation_size=box_representation_size,
    )

    # ── Box Predictor ──
    box_predictor = FastRCNNPredictor(
        in_channels=box_representation_size,
        num_classes=num_classes,
    )

    # ── RoI Heads ──
    if box_bbox_reg_weights is None:
        box_bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)

    roi_heads = RoIHeads(
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=box_fg_iou_thresh,
        bg_iou_thresh=box_bg_iou_thresh,
        batch_size_per_image=box_batch_size_per_image,
        positive_fraction=box_positive_fraction,
        bbox_reg_weights=box_bbox_reg_weights,
        score_thresh=box_score_thresh,
        nms_thresh=box_nms_thresh,
        detections_per_img=box_detections_per_img,
    )

    # ── Image Transforms ──
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
    )

    # ── Assemble Model ──
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None,   # pass None; we override rpn/roi_heads below
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
    )
    # Override with our custom modules
    model.rpn       = rpn
    model.roi_heads = roi_heads
    model.transform = transform

    return model


# ─────────────────────────────────────────────
# Model Info
# ─────────────────────────────────────────────
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary(model: nn.Module):
    params = count_parameters(model)
    print("=" * 60)
    print(f"  Model    : Faster R-CNN (ResNet-50 + FPN)")
    print(f"  Backbone : ResNet-50 (from scratch)")
    print(f"  Neck     : Feature Pyramid Network")
    print(f"  Total    : {params['total']:,} parameters")
    print(f"  Trainable: {params['trainable']:,} parameters")
    print("=" * 60)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import NUM_CLASSES, TARGET_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_faster_rcnn(num_classes=NUM_CLASSES).to(device)
    model_summary(model)

    # Dummy forward pass
    model.eval()
    dummy = [torch.rand(3, 800, 600).to(device)]
    with torch.no_grad():
        out = model(dummy)
    print(f"\nTest output: {len(out[0]['boxes'])} detections")
    print(f"Classes    : {TARGET_CLASSES}")
