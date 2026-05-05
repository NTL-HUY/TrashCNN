"""
model.py – Xây dựng FasterRCNN với ResNet50-FPN (Transfer Learning)
=====================================================================
Chiến lược Transfer Learning được áp dụng: Feature Extraction
  ✔ Tải trọng số pretrained (ImageNet) cho toàn bộ backbone
  ✔ ĐÓNG BĂNG toàn bộ backbone.parameters() → requires_grad = False
  ✔ Chỉ train: RPN (Region Proposal Network) + ROI Head (Box Predictor)

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

from src.config import (
    NUM_CLASSES, FREEZE_BACKBONE, PRETRAINED_BACKBONE,
    LEARNING_RATE, BACKBONE_LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA,
)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(
    num_classes:     int  = NUM_CLASSES,
    freeze_backbone: bool = FREEZE_BACKBONE,
    pretrained:      bool = PRETRAINED_BACKBONE,
) -> nn.Module:
    """
    Tạo FasterRCNN ResNet50-FPN.

    Args:
        num_classes:     số class bao gồm background (mặc định: 6 = 5 + bg)
        freeze_backbone: True → Feature Extraction
        pretrained:      True → dùng trọng số ImageNet pretrained

    Returns:
        model: nn.Module sẵn sàng để train/eval
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    if freeze_backbone:
        _freeze_all_backbone(model)
    else:
        _freeze_partial_backbone(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    _print_param_stats(model)

    return model


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def _freeze_all_backbone(model: nn.Module) -> None:
    """Feature Extraction: đóng băng toàn bộ backbone + FPN."""
    frozen = 0
    for param in model.backbone.parameters():
        param.requires_grad = False
        frozen += param.numel()
    print(f"[Model] Chiến lược: FEATURE EXTRACTION")
    print(f"        Frozen toàn bộ backbone ({frozen:,} params)")


def _freeze_partial_backbone(model: nn.Module) -> None:
    """
    Đóng băng toàn bộ backbone (bao gồm FPN layers).
    Sau khi gọi hàm này, KHÔNG layer nào trong backbone được cập nhật.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    layers_to_unfreeze = ["layer3", "layer4"]
    unfrozen = 0
    for layer_name in layers_to_unfreeze:
        layer = getattr(model.backbone.body, layer_name, None)
        if layer is not None:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += param.numel()

    for param in model.backbone.fpn.parameters():
        param.requires_grad = True
        unfrozen += param.numel()


def _print_param_stats(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"[Model] Tổng tham số  : {total:>12,}")
    print(f"[Model] Trainable     : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"[Model] Frozen        : {frozen:>12,}  ({100*frozen/total:.1f}%)")


# ---------------------------------------------------------------------------
# Optimizer – discriminative learning rates
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Tạo SGD optimizer CHỈ cho các tham số trainable (requires_grad=True).
    Backbone đã bị đóng băng sẽ không được truyền vào optimizer.
    """
    backbone_param_ids = {
        id(p) for p in model.backbone.parameters() if p.requires_grad
    }

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in backbone_param_ids
    ]
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_param_ids
    ]

    print(f"[Model] Optimizer param groups:")
    print(f"        backbone (layer3+4+FPN) : {sum(p.numel() for p in backbone_params):,} params  "
          f"lr={BACKBONE_LEARNING_RATE}")
    print(f"        RPN + ROI head          : {sum(p.numel() for p in head_params):,} params  "
          f"lr={LEARNING_RATE}")

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_params,     "lr": LEARNING_RATE},
        ],
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer):
    """
    StepLR: giảm LR của TẤT CẢ param groups cùng tỉ lệ gamma.
    Ví dụ: step=8, gamma=0.5
      → epoch 8:  head LR 0.005→0.0025, backbone LR 0.0005→0.00025
      → epoch 16: head LR 0.0025→0.00125, ...
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Model] Loaded checkpoint: {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


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
        "scheduler_state": scheduler.state_dict(),
        "metrics":         metrics,
    }, path)
    print(f"[Model] Checkpoint saved → {path}")