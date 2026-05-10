import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import resnet18


# CUSTOM BACKBONE
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNNBackbone(nn.Module):
    def __init__(self, fpn_channels: int = 256):
        super().__init__()

        self.block1 = nn.Sequential(ConvBlock(3, 32),   nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(ConvBlock(32, 64),  nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(ConvBlock(64, 128), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(ConvBlock(128, 256), nn.MaxPool2d(2, 2))

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[32, 64, 128, 256],
            out_channels=fpn_channels,
        )
        self.out_channels = fpn_channels

    def forward(self, x: torch.Tensor):
        c2 = self.block1(x)
        c3 = self.block2(c2)
        c4 = self.block3(c3)
        c5 = self.block4(c4)
        return self.fpn({"0": c2, "1": c3, "2": c4, "3": c5})

# PRETRAINED BACKBONE
class ResNet18FPN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=256
        )

        self.out_channels = 256

    def forward(self, x):
        x = self.conv1(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = {
            "0": c2,
            "1": c3,
            "2": c4,
            "3": c5
        }

        fpn_feats = self.fpn(feats)
        return fpn_feats

# BUILD MODEL
def build_model(num_classes=6, backbone_type="resnet18"):
    if backbone_type == "resnet18fpn":
        backbone = ResNet18FPN()
    else:
        backbone = SimpleCNNBackbone()

    anchor_generator = AnchorGenerator(
        sizes=(
            (32,),
            (64,),
            (128,),
            (256,),
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        )
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=640,
        max_size=640,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=50,
    )
    return model

if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    backbone = ResNet18FPN()
    output = backbone(x)
    print(type(output))
    for k, v in output.items():
        print(f"[{k}] -> {v.shape}")
