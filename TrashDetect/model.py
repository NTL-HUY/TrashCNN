import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def build_model(num_classes, my_weights_path):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)

    print(f"--- Loading backbone weights from: {my_weights_path} ---")
    custom_state_dict = torch.load(my_weights_path, map_location="cpu")
    backbone.body.load_state_dict(custom_state_dict, strict=False)

    for param in backbone.parameters():
        param.requires_grad = False

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model