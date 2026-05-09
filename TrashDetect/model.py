import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def load_backbone_weights(backbone, weights_path):
    print(f"--- Loading backbone weights from: {weights_path} ---")
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Fix 1: Lấy state_dict từ checkpoint dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("  [OK] Extracted 'model' key from checkpoint dict.")
    else:
        state_dict = checkpoint
        print("  [OK] Checkpoint is a raw state_dict.")

    layer_remap = {
        "layer2.": "layer1.",
        "layer3.": "layer2.",
        "layer4.": "layer3.",
        "layer5.": "layer4.",
    }

    remapped_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in layer_remap.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped_state_dict[new_k] = v

    backbone_state = backbone.body.state_dict()
    matched = {k: v for k, v in remapped_state_dict.items() if k in backbone_state and backbone_state[k].shape == v.shape}
    missing = [k for k in backbone_state if k not in matched]
    unexpected = [k for k in remapped_state_dict if k not in backbone_state]

    backbone.body.load_state_dict(matched, strict=False)

    print(f"  [OK] Matched layers  : {len(matched)}")
    print(f"  [!!] Missing layers  : {len(missing)}")
    print(f"  [!!] Unexpected keys : {len(unexpected)}")

    if len(matched) == 0:
        print("\n  [WARNING] Không có layer nào được load! Kiểm tra lại file weights.")
    elif len(matched) < 10:
        print("\n  [WARNING] Ít layer được load, backbone có thể chưa đúng.")
    else:
        print("  [OK] Backbone weights loaded successfully.")

    return backbone


def build_model(num_classes, my_weights_path):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)

    backbone = load_backbone_weights(backbone, my_weights_path)

    for param in backbone.parameters():
        param.requires_grad = False

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model