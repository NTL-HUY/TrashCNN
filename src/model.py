"""
FasterRCNN
=============================================
Sơ đồ kiến trúc:
  Image
    │
    ▼
  ResNet50Backbone
    │  layer1 → C1 (256 ch)
    │  layer2 → C2 (512 ch)
    │  layer3 → C3 (1024 ch)
    │  layer4 → C4 (2048 ch)
    ▼
  FPN (Feature Pyramid Network) ← tự build
    │  P2 (256 ch, stride 4)
    │  P3 (256 ch, stride 8)
    │  P4 (256 ch, stride 16)
    │  P5 (256 ch, stride 32)
    │  P6 (256 ch, stride 64) ← maxpool từ P5
    ▼
  RPN (Region Proposal Network) ← tự build
    │  • AnchorGenerator: tạo anchor boxes trên mỗi FPN level
    │  • RPNHead: conv → objectness score + bbox delta
    │  • Lọc proposal qua NMS
    ▼
  ROI Align + Box Head + Box Predictor ← tự build
    │  • ROIAlign: crop + resize feature từ FPN
    │  • BoxHead: 2 FC layers + Dropout
    │  • BoxPredictor: FC → class scores + bbox regression
    ▼
  Output: boxes, labels, scores
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    nms, roi_align, box_iou,
    clip_boxes_to_image, batched_nms,
)

from src.config import (
    NUM_CLASSES,
    RESNET_LAYERS,
    FPN_OUT_CHANNELS,
    RPN_ANCHOR_SIZES, RPN_ANCHOR_RATIOS,
    RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N,
    RPN_NMS_THRESH,
    RPN_FG_IOU_THRESH, RPN_BG_IOU_THRESH,
    RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION,
    ROI_BOX_SCORE_THRESH, ROI_NMS_THRESH,
    ROI_DETECTIONS_PER_IMG,
    ROI_FG_IOU_THRESH,
    ROI_BG_IOU_THRESH_HI, ROI_BG_IOU_THRESH_LO,
    ROI_BATCH_SIZE_PER_IMAGE, ROI_POSITIVE_FRACTION,
    ROI_POOLER_OUTPUT_SIZE, ROI_POOLER_SAMPLING_RATIO,
    DROPOUT_RATE,
    LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, LR_MIN,
)


# ===========================================================================
# 1. ResNet50 Backbone
# ===========================================================================

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck block: 1×1 → 3×3 → 1×1 conv với residual connection.
    expansion = 4: channel out = planes * 4
    """
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        # 1×1 conv: giảm channel
        self.conv1 = nn.Conv2d(in_channels, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        # 3×3 conv: xử lý spatial
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # 1×1 conv: tăng channel
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)

        # Shortcut: điều chỉnh channel/stride khi cần
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, planes * self.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone trả về feature maps từ layer1–4.
    Output: {"layer1": C1, "layer2": C2, "layer3": C3, "layer4": C4}
      C1: [B, 256,  H/4,  W/4]
      C2: [B, 512,  H/8,  W/8]
      C3: [B, 1024, H/16, W/16]
      C4: [B, 2048, H/32, W/32]
    """

    def __init__(self, layers: List[int] = RESNET_LAYERS):
        super().__init__()
        # Stem: conv 7×7 + maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 4 stages
        self.layer1 = self._make_layer(64,   64,  layers[0], stride=1)
        self.layer2 = self._make_layer(256,  128, layers[1], stride=2)
        self.layer3 = self._make_layer(512,  256, layers[2], stride=2)
        self.layer4 = self._make_layer(1024, 512, layers[3], stride=2)

        self._init_weights()

    def _make_layer(self, in_ch: int, planes: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [Bottleneck(in_ch, planes, stride)]
        for _ in range(1, n_blocks):
            blocks.append(Bottleneck(planes * 4, planes))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming init cho conv, constant init cho BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}

    @property
    def out_channels(self) -> Dict[str, int]:
        return {"layer1": 256, "layer2": 512,
                "layer3": 1024, "layer4": 2048}


# ===========================================================================
# 2. FPN
# ===========================================================================

class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Bottom-up:  C1(256) → C2(512) → C3(1024) → C4(2048)
    Lateral:    1×1 conv để đưa về FPN_OUT_CHANNELS
    Top-down:   upsample + add
    Output:     P2, P3, P4, P5 (cùng 256 ch) + P6 (maxpool P5)
    """

    def __init__(
        self,
        in_channels_list: List[int],   # [256, 512, 1024, 2048]
        out_channels: int = FPN_OUT_CHANNELS,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Bottom-up features theo thứ tự từ nhỏ đến lớn (C1→C4)
        c_list = [
            features["layer1"], features["layer2"],
            features["layer3"], features["layer4"],
        ]

        # Lateral connections
        lat = [conv(c) for conv, c in zip(self.lateral_convs, c_list)]

        # Top-down pathway: bắt đầu từ P5 (nhỏ nhất), upsample lên
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], size=lat[i - 1].shape[-2:], mode="nearest"
            )

        # Output convs (3×3 để smooth)
        out = [conv(f) for conv, f in zip(self.output_convs, lat)]

        # P6: maxpool từ P5, dùng cho anchor size lớn nhất
        p6 = F.max_pool2d(out[-1], 1, stride=2)

        return OrderedDict([
            ("P2", out[0]),
            ("P3", out[1]),
            ("P4", out[2]),
            ("P5", out[3]),
            ("P6", p6),
        ])


# ===========================================================================
# 3. Anchor Generator
# ===========================================================================

class AnchorGenerator(nn.Module):
    """
    Tạo anchor boxes cho mỗi FPN level.

    Mỗi level có 1 anchor size × 3 ratios = 3 anchors/cell.
    Tổng: 5 levels × 3 = 15 anchor templates.
    """

    def __init__(
        self,
        sizes:  Tuple = RPN_ANCHOR_SIZES,
        ratios: Tuple = RPN_ANCHOR_RATIOS,
    ):
        super().__init__()
        self.sizes  = sizes
        self.ratios = ratios
        self._cache: Dict[Tuple, torch.Tensor] = {}

    def _base_anchors(self, size: int, ratios: Tuple,
                      device: torch.device) -> torch.Tensor:
        """Tạo anchor templates (x1,y1,x2,y2) tại gốc tọa độ."""
        key = (size, ratios, str(device))
        if key in self._cache:
            return self._cache[key]

        ratios_t = torch.tensor(ratios, dtype=torch.float32, device=device)
        h_ratios  = torch.sqrt(ratios_t)
        w_ratios  = 1.0 / h_ratios

        ws = (w_ratios * size).round()
        hs = (h_ratios * size).round()

        anchors = torch.stack([
            -ws / 2, -hs / 2, ws / 2, hs / 2
        ], dim=1)
        self._cache[key] = anchors
        return anchors

    def _grid_anchors(
        self,
        base: torch.Tensor,
        feat_h: int, feat_w: int,
        stride: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Đặt anchors lên mọi cell của feature map."""
        shifts_x = torch.arange(0, feat_w, device=device) * stride
        shifts_y = torch.arange(0, feat_h, device=device) * stride
        sy, sx   = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts   = torch.stack([sx.ravel(), sy.ravel(),
                                sx.ravel(), sy.ravel()], dim=1).float()
        # [num_cells, 1, 4] + [1, num_base, 4] → [num_cells * num_base, 4]
        return (shifts[:, None] + base[None]).reshape(-1, 4)

    def forward(
        self,
        feature_maps: Dict[str, torch.Tensor],
        image_size:   Tuple[int, int],          # (H, W)
    ) -> List[torch.Tensor]:
        """Trả về list anchor tensor, 1 per FPN level."""
        device  = next(iter(feature_maps.values())).device
        strides = [4, 8, 16, 32, 64]           # tương ứng P2–P6
        anchors_all = []

        for (name, fmap), size, stride in zip(
            feature_maps.items(), self.sizes, strides
        ):
            fh, fw = fmap.shape[-2:]
            base   = self._base_anchors(size, self.ratios, device)
            grid   = self._grid_anchors(base, fh, fw, stride, device)
            anchors_all.append(grid)

        return anchors_all   # List[Tensor[N_i, 4]]


# ===========================================================================
# 4. RPN Head & RPN
# ===========================================================================

class RPNHead(nn.Module):
    """
    Shared conv head cho RPN.
    Input:  feature map [B, C, H, W]
    Output: objectness [B, num_anchors, H, W]
            bbox_delta [B, num_anchors*4, H, W]
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.obj_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.conv[0], self.obj_logits, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        obj_list, delta_list = [], []
        for f in features:
            t = self.conv(f)
            obj_list.append(self.obj_logits(t))
            delta_list.append(self.bbox_pred(t))
        return obj_list, delta_list


def _decode_boxes(
    anchors: torch.Tensor,   # [N, 4] x1y1x2y2
    deltas:  torch.Tensor,   # [N, 4] dx dy dw dh
) -> torch.Tensor:
    """Giải mã bbox regression delta → tọa độ tuyệt đối."""
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    dx, dy, dw, dh = deltas.unbind(1)
    # Clamp dw/dh để tránh exp overflow
    dw = dw.clamp(max=math.log(1000.0 / 16))
    dh = dh.clamp(max=math.log(1000.0 / 16))

    x = dx * wa + xa
    y = dy * ha + ya
    w = torch.exp(dw) * wa
    h = torch.exp(dh) * ha

    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)


def _encode_boxes(
    anchors:  torch.Tensor,   # [N, 4]
    gt_boxes: torch.Tensor,   # [N, 4]
) -> torch.Tensor:
    """Encode ground truth boxes thành delta."""
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    wg = gt_boxes[:, 2] - gt_boxes[:, 0]
    hg = gt_boxes[:, 3] - gt_boxes[:, 1]
    xg = gt_boxes[:, 0] + 0.5 * wg
    yg = gt_boxes[:, 1] + 0.5 * hg

    dx = (xg - xa) / wa
    dy = (yg - ya) / ha
    dw = torch.log(wg / wa)
    dh = torch.log(hg / ha)
    return torch.stack([dx, dy, dw, dh], dim=1)


class RPN(nn.Module):
    """
    Region Proposal Network.
    - Training: trả về loss_objectness + loss_rpn_box_reg
    - Inference: trả về list proposals đã qua NMS
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.head = RPNHead(in_channels, num_anchors)

    def _assign_targets(
        self,
        anchors:  torch.Tensor,   # [N, 4]
        gt_boxes: torch.Tensor,   # [M, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gán nhãn fg/bg cho mỗi anchor.
        Returns:
            labels: [N]  1=fg, 0=bg, -1=ignore
            matched_gt: [N, 4]  GT box tương ứng
        """
        if gt_boxes.numel() == 0:
            labels     = torch.zeros(len(anchors), dtype=torch.long,
                                     device=anchors.device)
            matched_gt = torch.zeros_like(anchors)
            return labels, matched_gt

        iou = box_iou(anchors, gt_boxes)          # [N, M]
        max_iou, best_gt = iou.max(dim=1)         # per anchor

        labels = torch.full((len(anchors),), -1,
                            dtype=torch.long, device=anchors.device)
        labels[max_iou >= RPN_FG_IOU_THRESH] = 1
        labels[max_iou <  RPN_BG_IOU_THRESH] = 0

        # Đảm bảo mỗi GT có ít nhất 1 anchor fg
        best_anchor_per_gt = iou.max(dim=0)[1]   # [M]
        labels[best_anchor_per_gt] = 1

        # Sampling: giới hạn số lượng fg/bg
        n_fg = int(RPN_BATCH_SIZE_PER_IMAGE * RPN_POSITIVE_FRACTION)
        fg_idx = (labels == 1).nonzero(as_tuple=False).squeeze(1)
        if fg_idx.numel() > n_fg:
            perm = torch.randperm(fg_idx.numel(), device=anchors.device)
            labels[fg_idx[perm[n_fg:]]] = -1

        n_bg = RPN_BATCH_SIZE_PER_IMAGE - (labels == 1).sum()
        bg_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
        if bg_idx.numel() > n_bg:
            perm = torch.randperm(bg_idx.numel(), device=anchors.device)
            labels[bg_idx[perm[n_bg:]]] = -1

        matched_gt = gt_boxes[best_gt]
        return labels, matched_gt

    def _filter_proposals(
        self,
        proposals:  torch.Tensor,   # [N, 4]
        obj_logits: torch.Tensor,   # [N]
        image_size: Tuple[int, int],
        training:   bool,
    ) -> torch.Tensor:
        """Clip → lọc pre-NMS → NMS → lấy top-K."""
        pre_k  = RPN_PRE_NMS_TOP_N["training" if training else "testing"]
        post_k = RPN_POST_NMS_TOP_N["training" if training else "testing"]

        scores = obj_logits.sigmoid()
        proposals = clip_boxes_to_image(proposals, image_size)

        # Lọc box có w/h quá nhỏ
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        keep = (ws >= 1) & (hs >= 1)
        proposals, scores = proposals[keep], scores[keep]

        # Top pre_k
        if scores.numel() > pre_k:
            top_idx   = scores.topk(pre_k)[1]
            proposals = proposals[top_idx]
            scores    = scores[top_idx]

        keep = nms(proposals, scores, RPN_NMS_THRESH)[:post_k]
        return proposals[keep]

    def forward(
        self,
        features:    Dict[str, torch.Tensor],
        anchors_all: List[torch.Tensor],
        targets:     Optional[List[Dict]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        feat_list = list(features.values())
        obj_list, delta_list = self.head(feat_list)

        # Flatten tất cả FPN levels
        all_anchors     = torch.cat(anchors_all, dim=0)          # [N_all, 4]
        all_obj         = torch.cat([o.permute(0,2,3,1).reshape(o.shape[0], -1)
                                     for o in obj_list], dim=1)  # [B, N_all]
        all_delta       = torch.cat([d.permute(0,2,3,1).reshape(d.shape[0], -1, 4)
                                     for d in delta_list], dim=1) # [B, N_all, 4]

        batch_size = all_obj.shape[0]
        losses: Dict[str, torch.Tensor] = {}
        proposals_list: List[torch.Tensor] = []

        for i in range(batch_size):
            props = _decode_boxes(all_anchors, all_delta[i])

            img_sz = image_sizes[i] if image_sizes else (800, 800)
            filtered = self._filter_proposals(
                props, all_obj[i], img_sz, self.training
            )
            proposals_list.append(filtered)

            if self.training and targets is not None:
                gt_boxes = targets[i]["boxes"]
                labels, matched_gt = self._assign_targets(all_anchors, gt_boxes)

                sampled = labels >= 0
                loss_obj = F.binary_cross_entropy_with_logits(
                    all_obj[i][sampled],
                    labels[sampled].float(),
                )

                fg_mask   = labels == 1
                if fg_mask.any():
                    target_delta = _encode_boxes(all_anchors[fg_mask],
                                                 matched_gt[fg_mask])
                    loss_reg = F.smooth_l1_loss(
                        all_delta[i][fg_mask], target_delta,
                        beta=1.0 / 9, reduction="sum",
                    ) / max(fg_mask.sum().item(), 1)
                else:
                    loss_reg = all_delta[i][fg_mask].sum() * 0.0

                losses[f"loss_rpn_obj_{i}"]     = loss_obj
                losses[f"loss_rpn_box_reg_{i}"] = loss_reg

        # Tổng hợp loss trung bình
        if losses:
            obj_losses = [v for k, v in losses.items() if "obj" in k]
            reg_losses = [v for k, v in losses.items() if "reg" in k]
            final_losses = {
                "loss_objectness":    sum(obj_losses) / len(obj_losses),
                "loss_rpn_box_reg":   sum(reg_losses) / len(reg_losses),
            }
        else:
            final_losses = {}

        return proposals_list, final_losses


# ===========================================================================
# 5. ROI Head
# ===========================================================================

class BoxHead(nn.Module):
    """
    Hai FC layers sau ROIAlign.
    Input:  [N, C * pool_h * pool_w]  (flattened ROI features)
    Output: [N, 1024]
    """

    def __init__(
        self,
        in_channels: int,
        pool_size:   int = ROI_POOLER_OUTPUT_SIZE,
        dropout:     float = DROPOUT_RATE,
    ):
        super().__init__()
        flat = in_channels * pool_size * pool_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BoxPredictor(nn.Module):
    """
    Đầu ra cuối cùng của ROI Head.
    Input:  [N, 1024]
    Output: cls_logits [N, num_classes], bbox_delta [N, num_classes*4]
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.cls_score  = nn.Linear(in_features, num_classes)
        self.bbox_pred  = nn.Linear(in_features, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls_score(x), self.bbox_pred(x)


class ROIHead(nn.Module):
    """
    ROI Align + BoxHead + BoxPredictor.
    Thực hiện:
      1. ROIAlign: crop feature từ FPN theo proposal
      2. BoxHead: 2 FC + Dropout
      3. BoxPredictor: cls + reg
      4. (training) tính loss
      5. (inference) giải mã box + NMS
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.box_head      = BoxHead(in_channels)
        self.box_predictor = BoxPredictor(1024, num_classes)
        self.num_classes   = num_classes

        # FPN level mapping dựa trên diện tích proposal
        self.fpn_strides = [4, 8, 16, 32]   # P2–P5
        self.pool_size   = ROI_POOLER_OUTPUT_SIZE
        self.sampling    = ROI_POOLER_SAMPLING_RATIO

    def _map_to_fpn_level(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Gán mỗi ROI vào FPN level dựa theo diện tích.
        Công thức: k = floor(k0 + log2(sqrt(area) / 224))
        k0 = 4, clamp về [2, 5] → index [0, 3] tương ứng P2–P5.
        """
        areas  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        levels = torch.floor(
            4 + torch.log2(areas.sqrt().clamp(min=1e-6) / 224.0)
        ).clamp(2, 5).long() - 2   # → 0-indexed [0,3]
        return levels

    def _roi_align_multi_level(
        self,
        features:   Dict[str, torch.Tensor],
        proposals:  List[torch.Tensor],          # per image
    ) -> torch.Tensor:
        """
        ROIAlign trên feature map FPN level phù hợp, gộp lại theo thứ tự gốc.
        """
        fpn_keys = ["P2", "P3", "P4", "P5"]
        strides  = [4, 8, 16, 32]

        # Gộp proposals + lưu image index
        all_boxes, img_ids = [], []
        for i, boxes in enumerate(proposals):
            all_boxes.append(boxes)
            img_ids.append(torch.full((len(boxes),), i,
                                      dtype=torch.long, device=boxes.device))
        all_boxes = torch.cat(all_boxes, dim=0)   # [N_total, 4]
        img_ids   = torch.cat(img_ids,   dim=0)   # [N_total]

        levels = self._map_to_fpn_level(all_boxes)
        pooled = torch.zeros(
            len(all_boxes), list(features.values())[0].shape[1],
            self.pool_size, self.pool_size,
            device=all_boxes.device,
        )

        for lvl, (key, stride) in enumerate(zip(fpn_keys, strides)):
            if key not in features:
                continue
            mask = levels == lvl
            if not mask.any():
                continue
            boxes_lvl = all_boxes[mask]
            ids_lvl   = img_ids[mask].float()

            rois = torch.cat([ids_lvl[:, None], boxes_lvl], dim=1)
            pooled[mask] = roi_align(
                features[key], rois,
                output_size=self.pool_size,
                spatial_scale=1.0 / stride,
                sampling_ratio=self.sampling,
                aligned=True,
            )

        return pooled   # [N_total, C, pool, pool]

    def _assign_targets(
        self,
        proposals: torch.Tensor,   # [N, 4]
        gt_boxes:  torch.Tensor,   # [M, 4]
        gt_labels: torch.Tensor,   # [M]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gt_boxes.numel() == 0:
            labels     = torch.zeros(len(proposals), dtype=torch.long,
                                     device=proposals.device)
            matched_gt = torch.zeros_like(proposals)
            return labels, matched_gt, torch.zeros_like(proposals)

        iou = box_iou(proposals, gt_boxes)
        max_iou, best_gt = iou.max(dim=1)

        labels = torch.zeros(len(proposals), dtype=torch.long,
                             device=proposals.device)

        fg_mask = max_iou >= ROI_FG_IOU_THRESH
        bg_mask = (max_iou >= ROI_BG_IOU_THRESH_LO) & \
                  (max_iou <  ROI_BG_IOU_THRESH_HI)

        labels[fg_mask] = gt_labels[best_gt[fg_mask]]
        labels[bg_mask] = 0   # background

        # Ignore cả fg lẫn bg không xác định
        ignore = ~fg_mask & ~bg_mask
        labels[ignore] = -1

        # Sampling
        n_fg = int(ROI_BATCH_SIZE_PER_IMAGE * ROI_POSITIVE_FRACTION)
        fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(1)
        if fg_idx.numel() > n_fg:
            perm = torch.randperm(fg_idx.numel(), device=proposals.device)
            labels[fg_idx[perm[n_fg:]]] = -1

        n_bg = ROI_BATCH_SIZE_PER_IMAGE - (labels > 0).sum()
        bg_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
        if bg_idx.numel() > n_bg:
            perm = torch.randperm(bg_idx.numel(), device=proposals.device)
            labels[bg_idx[perm[n_bg:]]] = -1

        matched_gt = gt_boxes[best_gt]
        return labels, matched_gt, gt_boxes[best_gt]

    def forward(
        self,
        features:   Dict[str, torch.Tensor],
        proposals:  List[torch.Tensor],
        targets:    Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:

        losses: Dict[str, torch.Tensor] = {}

        if self.training and targets is not None:
            proposals = [
                torch.cat([p, t["boxes"]], dim=0)
                for p, t in zip(proposals, targets)
            ]

            # Gán target cho từng proposal
            all_proposals, all_labels, all_gt_boxes = [], [], []
            for i, (props, tgt) in enumerate(zip(proposals, targets)):
                lbl, matched, _ = self._assign_targets(
                    props, tgt["boxes"], tgt["labels"]
                )
                sampled = lbl >= 0
                all_proposals.append(props[sampled])
                all_labels.append(lbl[sampled])
                all_gt_boxes.append(matched[sampled])

            proposals    = all_proposals
            sampled_lbls = torch.cat(all_labels)
            sampled_gts  = torch.cat(all_gt_boxes)

        # ROIAlign
        roi_feats = self._roi_align_multi_level(features, proposals)
        box_feats = self.box_head(roi_feats)
        cls_logits, bbox_deltas = self.box_predictor(box_feats)

        if self.training and targets is not None:
            # Classification loss
            valid = sampled_lbls >= 0
            loss_cls = F.cross_entropy(
                cls_logits[valid], sampled_lbls[valid]
            )

            # Regression loss (chỉ fg)
            fg_mask = sampled_lbls > 0
            if fg_mask.any():
                fg_labels = sampled_lbls[fg_mask]
                # Lấy delta của đúng class
                idx       = torch.arange(fg_mask.sum(), device=fg_labels.device)
                fg_deltas = bbox_deltas[fg_mask].reshape(-1, self.num_classes, 4)
                fg_deltas = fg_deltas[idx, fg_labels]

                # Proposal tương ứng
                flat_props = torch.cat(proposals, dim=0)
                n_per_img  = [len(p) for p in proposals]
                flat_gts   = sampled_gts

                # Encode GT
                fg_props   = flat_props[fg_mask]
                target_enc = _encode_boxes(fg_props, flat_gts[fg_mask])

                loss_box = F.smooth_l1_loss(
                    fg_deltas, target_enc,
                    beta=1.0, reduction="sum",
                ) / max(fg_mask.sum().item(), 1)
            else:
                loss_box = cls_logits[fg_mask].sum() * 0.0

            losses = {
                "loss_classifier": loss_cls,
                "loss_box_reg":    loss_box,
            }
            return [], losses

        # Inference: giải mã + NMS
        results = []
        offset  = 0
        for i, props in enumerate(proposals):
            n = len(props)
            if n == 0:
                results.append({"boxes": props, "labels": props.new_zeros(0, dtype=torch.long), "scores": props.new_zeros(0)})
                continue

            cls_i   = cls_logits[offset:offset + n]
            delta_i = bbox_deltas[offset:offset + n].reshape(n, self.num_classes, 4)
            scores_i = F.softmax(cls_i, dim=1)[:, 1:]   # bỏ background

            boxes_list, score_list, label_list = [], [], []
            for cls_idx in range(1, self.num_classes):
                sc   = scores_i[:, cls_idx - 1]
                dl   = delta_i[:, cls_idx]
                bx   = _decode_boxes(props, dl)
                keep = sc >= ROI_BOX_SCORE_THRESH
                boxes_list.append(bx[keep])
                score_list.append(sc[keep])
                label_list.append(torch.full((keep.sum(),), cls_idx,
                                             dtype=torch.long,
                                             device=props.device))

            if boxes_list:
                boxes_cat  = torch.cat(boxes_list)
                scores_cat = torch.cat(score_list)
                labels_cat = torch.cat(label_list)
                keep = batched_nms(boxes_cat, scores_cat, labels_cat,
                                   ROI_NMS_THRESH)[:ROI_DETECTIONS_PER_IMG]
                results.append({
                    "boxes":  boxes_cat[keep],
                    "labels": labels_cat[keep],
                    "scores": scores_cat[keep],
                })
            else:
                results.append({
                    "boxes":  props.new_zeros((0, 4)),
                    "labels": props.new_zeros(0, dtype=torch.long),
                    "scores": props.new_zeros(0),
                })
            offset += n

        return results, {}


# ===========================================================================
# 6. FasterRCNN
# ===========================================================================

class FasterRCNN(nn.Module):
    """
    FasterRCNN

    Forward:
      training → trả về dict losses
      inference → trả về list[dict{"boxes","labels","scores"}]
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = ResNet50Backbone(RESNET_LAYERS)
        self.fpn      = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=FPN_OUT_CHANNELS,
        )
        num_anchors   = len(RPN_ANCHOR_RATIOS)   # 3 per cell
        self.rpn      = RPN(FPN_OUT_CHANNELS, num_anchors)
        self.roi_head = ROIHead(FPN_OUT_CHANNELS, num_classes)
        self.anchor_gen = AnchorGenerator()

    def forward(
        self,
        images:  List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ):
        """
        Args:
            images:  List[Tensor[C,H,W]] – đã normalize, kích thước có thể khác nhau
            targets: List[{"boxes": Tensor[N,4], "labels": Tensor[N]}]
                     Chỉ cần trong lúc training.

        Returns (training):
            dict {"loss_objectness", "loss_rpn_box_reg",
                  "loss_classifier", "loss_box_reg"}
        Returns (inference):
            List[{"boxes": Tensor[K,4], "labels": Tensor[K], "scores": Tensor[K]}]
        """
        if self.training and targets is None:
            raise ValueError("targets phải được cung cấp khi training")

        device = images[0].device

        # ── Batch images → pad về cùng kích thước ────────────────────
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        max_h = max(s[0] for s in image_sizes)
        max_w = max(s[1] for s in image_sizes)
        batch = torch.zeros(len(images), 3, max_h, max_w, device=device)
        for i, img in enumerate(images):
            h, w = img.shape[-2:]
            batch[i, :, :h, :w] = img

        # ── Backbone + FPN ────────────────────────────────────────────
        backbone_feats = self.backbone(batch)
        fpn_feats      = self.fpn(backbone_feats)

        # ── Anchors ───────────────────────────────────────────────────
        anchors_all = self.anchor_gen(fpn_feats, (max_h, max_w))

        # ── RPN ───────────────────────────────────────────────────────
        proposals, rpn_losses = self.rpn(
            fpn_feats, anchors_all, targets, image_sizes
        )

        # ── ROI Head ─────────────────────────────────────────────────
        detections, roi_losses = self.roi_head(fpn_feats, proposals, targets)

        if self.training:
            return {**rpn_losses, **roi_losses}

        # Clip boxes vào biên ảnh gốc (trước padding)
        for i, (det, sz) in enumerate(zip(detections, image_sizes)):
            if det["boxes"].numel() > 0:
                det["boxes"] = clip_boxes_to_image(det["boxes"], sz)

        return detections


# ===========================================================================
# 7. Factory functions
# ===========================================================================

def build_model(num_classes: int = NUM_CLASSES) -> FasterRCNN:
    model = FasterRCNN(num_classes=num_classes)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] FasterRCNN")
    print(f"[Model] Tổng tham số  : {total:>12,}")
    print(f"[Model] Trainable     : {trainable:>12,}")
    return model


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def get_lr_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Model] Loaded checkpoint: {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')})")
    return ckpt


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    metrics:   dict,
    path:      str,
) -> None:
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
                           if hasattr(scheduler, "state_dict") else {},
        "metrics":         metrics,
    }, path)
    print(f"[Model] Checkpoint saved → {path}")