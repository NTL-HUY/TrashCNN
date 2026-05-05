"""
model.py – Xây dựng FasterRCNN với ResNet50-FPN (Transfer Learning)
=====================================================================
Chiến lược Transfer Learning được áp dụng: Feature Extraction
  ✔ Tải trọng số pretrained (ImageNet) cho toàn bộ backbone
  ✔ ĐÓNG BĂNG toàn bộ backbone.parameters() → requires_grad = False
  ✔ Chỉ train: RPN (Region Proposal Network) + ROI Head (Box Predictor)
  ✘ KHÔNG fine-tune backbone dù chỉ 1 layer

Tại sao dùng FPN (Feature Pyramid Network)?
  ResNet50 kết hợp FPN cho phép phát hiện vật thể ở nhiều tỉ lệ kích thước
  (rác nhỏ như đầu thuốc lá và rác lớn như túi rác) trong cùng 1 ảnh.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import NUM_CLASSES, FREEZE_BACKBONE, PRETRAINED_BACKBONE


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = FREEZE_BACKBONE,
    pretrained: bool = PRETRAINED_BACKBONE,
) -> nn.Module:
    """
    Tạo FasterRCNN ResNet50-FPN với head tùy chỉnh cho bài toán phân loại rác.

    Args:
        num_classes:     số class bao gồm background (mặc định: 6 = 5 + bg)
        freeze_backbone: True → Feature Extraction (không fine-tune backbone)
        pretrained:      True → dùng trọng số ImageNet pretrained

    Returns:
        model: nn.Module sẵn sàng để train/eval
    """
    # ── Bước 1: Load pretrained FasterRCNN ────────────────────────────────
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    # ── Bước 2: Đóng băng backbone (Feature Extraction mode) ──────────────
    if freeze_backbone:
        _freeze_backbone(model)

    # ── Bước 3: Thay Box Predictor head bằng head của mình ────────────
    # in_features của head gốc được lấy từ chính model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Head mới này mặc định requires_grad = True → sẽ được train

    # ── Bước 4: In thống kê để xác nhận ──────────────────────────────────
    _print_param_stats(model)

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module) -> None:
    """
    Đóng băng toàn bộ backbone (bao gồm FPN layers).
    Sau khi gọi hàm này, KHÔNG layer nào trong backbone được cập nhật.
    """
    frozen_count = 0
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        frozen_count += param.numel()

    print(f"[Model] Backbone đã được ĐÓNG BĂNG "
          f"({frozen_count:,} tham số không được train)")


def _print_param_stats(model: nn.Module) -> None:
    """In thống kê số tham số trainable vs frozen."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"[Model] Tổng tham số  : {total:>12,}")
    print(f"[Model] Trainable     : {trainable:>12,}  "
          f"({100 * trainable / total:.1f}%)")
    print(f"[Model] Frozen        : {frozen:>12,}  "
          f"({100 * frozen / total:.1f}%)")


def get_optimizer(model: nn.Module):
    """
    Tạo SGD optimizer CHỈ cho các tham số trainable (requires_grad=True).
    Backbone đã bị đóng băng sẽ không được truyền vào optimizer.
    """
    from src.config import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


def get_lr_scheduler(optimizer):
    """StepLR: giảm learning rate sau mỗi LR_STEP_SIZE epoch."""
    from src.config import LR_STEP_SIZE, LR_GAMMA
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Model] Loaded checkpoint: {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics":        metrics,
    }, path)
    print(f"[Model] Checkpoint saved → {path}")
