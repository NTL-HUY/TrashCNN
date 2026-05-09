import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def load_backbone_weights(backbone, weights_path):
    print(f"--- Loading backbone weights from: {weights_path} ---")
    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("  [OK] Extracted 'model' key from checkpoint dict.")
    else:
        state_dict = checkpoint
        print("  [OK] Checkpoint is a raw state_dict.")

    backbone_state = backbone.body.state_dict()
    matched = {
        k: v for k, v in state_dict.items()
        if k in backbone_state and backbone_state[k].shape == v.shape
    }
    missing    = [k for k in backbone_state if k not in matched]
    unexpected = [k for k in state_dict if k not in backbone_state]

    backbone.body.load_state_dict(matched, strict=False)

    total = len(backbone_state)
    print(f"  [OK] Matched layers  : {len(matched)} / {total}")
    print(f"  [!!] Missing layers  : {len(missing)}")
    print(f"  [!!] Unexpected keys : {len(unexpected)}")

    if len(matched) == 0:
        print("\n  [ERROR] Không có layer nào được load!")
    elif len(matched) < total * 0.8:
        print(f"\n  [WARNING] Chỉ load được {len(matched)}/{total} layers.")
    else:
        print(f"  [OK] Backbone weights loaded successfully.")

    return backbone


def freeze_backbone_selective(backbone):
    for param in backbone.parameters():
        param.requires_grad = False

    for param in backbone.fpn.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  Backbone — Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    return backbone


def build_model(num_classes, my_weights_path):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    backbone = load_backbone_weights(backbone, my_weights_path)
    backbone = freeze_backbone_selective(backbone)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model