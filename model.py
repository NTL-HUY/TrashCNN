"""
model.py - Faster R-CNN with ResNet-50 + FPN (trained from scratch, no pretrained weights)
Architecture:
  Backbone  : ResNet-50 with Group Normalization (Fix for small batch size)
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


# ─────────────────────────────────────────────
# ResNet-50 Backbone (from scratch with GroupNorm)
# ─────────────────────────────────────────────
class Bottleneck(nn.Module):
    """ResNet Bottleneck block (3-layer: 1x1, 3x3, 1x1 conv) with GroupNorm."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, groups=32):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.gn1   = nn.GroupNorm(groups, mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(groups, mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.gn3   = nn.GroupNorm(groups, out_channels)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet-50 backbone using GroupNorm for training from scratch with small batch sizes.
    Returns: {"0": C2, "1": C3, "2": C4, "3": C5}
    """
    LAYERS = [3, 4, 6, 3]

    def __init__(self, groups=32):
        super().__init__()
        self.groups = groups

        # Stem
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.gn1     = nn.GroupNorm(self.groups, 64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64,  64,  self.LAYERS[0], stride=1)
        self.layer2 = self._make_layer(256, 128, self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(512, 256, self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(1024,512, self.LAYERS[3], stride=2)

        self._init_weights()

    def _make_layer(self, in_channels, mid_channels, num_blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(self.groups, out_channels),
            )
        layers = [Bottleneck(in_channels, mid_channels, stride, downsample, groups=self.groups)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, mid_channels, groups=self.groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.maxpool(self.relu(self.gn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)])

    @property
    def out_channels(self):
        return {"0": 256, "1": 512, "2": 1024, "3": 2048}


# ─────────────────────────────────────────────
# ResNet50 + FPN Backbone
# ─────────────────────────────────────────────
class ResNet50FPN(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.body = ResNet50()
        self.fpn  = FeaturePyramidNetwork(
            in_channels_list=list(self.body.out_channels.values()),
            out_channels=out_channels,
            extra_blocks=None,
        )
        self.out_channels = out_channels
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
        pyramid  = self.fpn(features)
        pyramid["pool"] = self.extra_pool(pyramid["3"])
        return pyramid


# ─────────────────────────────────────────────
# Box Predictor Head
# ─────────────────────────────────────────────
class FastRCNNPredictor(nn.Module):
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
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
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
    box_roi_pool_output_size: int = 7,
    box_representation_size: int = 1024,
    box_fg_iou_thresh: float = 0.5,
    box_bg_iou_thresh: float = 0.5,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    box_bbox_reg_weights: Optional[Tuple[float, ...]] = None,
) -> FasterRCNN:

    backbone = ResNet50FPN(out_channels=256)

    if rpn_anchor_sizes is None:
        rpn_anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    if rpn_aspect_ratios is None:
        rpn_aspect_ratios = ((0.5, 1.0, 2.0),) * len(rpn_anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=rpn_anchor_sizes,
        aspect_ratios=rpn_aspect_ratios,
    )

    rpn_head = RPNHead(
        in_channels=backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
    )

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

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=box_roi_pool_output_size,
        sampling_ratio=2,
    )

    resolution = box_roi_pool_output_size
    in_channels = backbone.out_channels * resolution ** 2
    box_head = TwoMLPHead(
        in_channels=in_channels,
        representation_size=box_representation_size,
    )

    box_predictor = FastRCNNPredictor(
        in_channels=box_representation_size,
        num_classes=num_classes,
    )

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

    model = FasterRCNN(
        backbone=backbone,
        num_classes=None,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
    )
    model.rpn       = rpn
    model.roi_heads = roi_heads
    model.transform = transform

    return model

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def model_summary(model: nn.Module):
    params = count_parameters(model)
    print("=" * 60)
    print(f"  Model    : Faster R-CNN (ResNet-50 + FPN)")
    print(f"  Backbone : ResNet-50 (from scratch, GroupNorm)")
    print(f"  Neck     : Feature Pyramid Network")
    print(f"  Total    : {params['total']:,} parameters")
    print(f"  Trainable: {params['trainable']:,} parameters")
    print("=" * 60)