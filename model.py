import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import resnet18


class ResNet18FPN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=None)

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


def build_model(num_classes=6):
    # Option A: ResNet-50 pretrained + FPN (mAP cao nhất)
    backbone = ResNet18FPN()

    anchor_generator = AnchorGenerator(
        sizes=(
            (16,),
            (32,),
            (64,),
            (128,)
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0)
        )
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=416,
        max_size=416,
        # Các tham số RPN — nếu không truyền thì dùng mặc định
        rpn_pre_nms_top_n_train=2000,  # lấy top 2000 trước NMS lúc train
        rpn_pre_nms_top_n_test=1000,  # lấy top 1000 trước NMS lúc inference
        rpn_post_nms_top_n_train=2000,  # giữ tối đa 2000 sau NMS lúc train
        rpn_post_nms_top_n_test=1000,  # giữ tối đa 1000 sau NMS lúc inference
        rpn_nms_thresh=0.7,  # IoU > 0.7 thì coi là trùng, bỏ
        rpn_score_thresh=0.0,  # score tối thiểu để giữ anchor

        # Box/detection params — bạn đang thiếu phần này
        box_score_thresh=0.05,  # loại box có score < 0.3
        box_nms_thresh=0.5,  # NMS lần 2 sau khi head phân loại
        box_detections_per_img=50,  # tối đa 50 box trên 1 ảnh
    )
    return model


if __name__ == '__main__':

    x = torch.randn(1, 3, 416, 416)

    backbone = ResNet18FPN()

    output = backbone(x)

    print(type(output))

    for k, v in output.items():
        print(f"[{k}] -> {v.shape}")
