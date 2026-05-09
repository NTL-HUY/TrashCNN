import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def freeze_backbone_selective(backbone):
    for param in backbone.parameters():
        param.requires_grad = False

    for name, param in backbone.body.named_parameters():
        if name.startswith("layer4."):
            param.requires_grad = True

    for param in backbone.fpn.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  Backbone — Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    return backbone


def build_model(num_classes, my_weights_path=None):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    backbone = freeze_backbone_selective(backbone)

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model