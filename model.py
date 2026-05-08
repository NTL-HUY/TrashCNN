# import torch
# import torch.nn as nn
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models import ResNet50_Weights
#
# class SimpleBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.ReLU()
#         )
#         self.out_channels = 128
#
#     def forward(self, x):
#         return self.body(x)
#
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         return F.relu(out)
#
#
# class BetterBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, 64, 7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.layer1 = self._make_layer(64, 128, stride=2)
#         self.layer2 = self._make_layer(128, 256, stride=2)
#         self.layer3 = self._make_layer(256, 256, stride=1)
#
#         self.out_channels = 256
#
#     def _make_layer(self, in_c, out_c, stride):
#         return nn.Sequential(
#             ResidualBlock(in_c, out_c, stride),
#             ResidualBlock(out_c, out_c)
#         )
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x
#
#
# # def build_model(num_classes=7):
# #     anchor_generator = AnchorGenerator(
# #         sizes=((8, 16, 32, 64, 128, 256),),
# #         aspect_ratios=((0.3, 0.5, 1.0, 2.0, 3.0),)
# #     )
# #     model = FasterRCNN(
# #         backbone=BetterBackbone(),
# #         num_classes=num_classes,
# #         rpn_anchor_generator=anchor_generator,
# #         box_score_thresh=0.3,  # giảm threshold
# #         box_detections_per_img=50
# #     )
# #     return model
#
#
# def build_model(num_classes=6):
#     # Option A: ResNet-50 pretrained + FPN (mAP cao nhất)
#     backbone = resnet_fpn_backbone(
#         backbone_name='resnet50',
#         pretrained=True,  # ImageNet weights
#         trainable_layers=3  # Freeze 2 layer đầu
#     )
#
#     anchor_generator = AnchorGenerator(
#         sizes=((32,), (64,), (128,), (256,), (512,)),
#         aspect_ratios=((0.5, 1.0, 2.0),) * 5
#     )
#
#     model = FasterRCNN(
#         backbone=backbone,
#         num_classes=6,
#         rpn_anchor_generator=anchor_generator,
#
#         # Các tham số RPN — nếu không truyền thì dùng mặc định
#         rpn_pre_nms_top_n_train=2000,  # lấy top 2000 trước NMS lúc train
#         rpn_pre_nms_top_n_test=1000,  # lấy top 1000 trước NMS lúc inference
#         rpn_post_nms_top_n_train=2000,  # giữ tối đa 2000 sau NMS lúc train
#         rpn_post_nms_top_n_test=1000,  # giữ tối đa 1000 sau NMS lúc inference
#         rpn_nms_thresh=0.7,  # IoU > 0.7 thì coi là trùng, bỏ
#         rpn_score_thresh=0.0,  # score tối thiểu để giữ anchor
#
#         # Box/detection params — bạn đang thiếu phần này
#         box_score_thresh=0.05,  # loại box có score < 0.3
#         box_nms_thresh=0.5,  # NMS lần 2 sau khi head phân loại
#         box_detections_per_img=50,  # tối đa 50 box trên 1 ảnh
#     )
#     return model
#
#
# if __name__ == '__main__':
#     x = torch.randn(1, 3, 800, 800)  # 1 ảnh
#     backbone = resnet_fpn_backbone(
#         backbone_name='resnet50',
#         pretrained=True,  # ImageNet weights
#         trainable_layers=3  # Freeze 2 layer đầu
#     )
#
#     output = backbone(x)
#     print(type(output))
#     # OrderedDict
#
#     for k, v in output.items():
#         print(f"  [{k}]: {v.shape}")



import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights, resnet18


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.out_channels = 128

    def forward(self, x):
        return self.body(x)


import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class BetterBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 256, stride=1)

        self.out_channels = 256

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            ResidualBlock(out_c, out_c)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# def build_model(num_classes=7):
#     anchor_generator = AnchorGenerator(
#         sizes=((8, 16, 32, 64, 128, 256),),
#         aspect_ratios=((0.3, 0.5, 1.0, 2.0, 3.0),)
#     )
#     model = FasterRCNN(
#         backbone=BetterBackbone(),
#         num_classes=num_classes,
#         rpn_anchor_generator=anchor_generator,
#         box_score_thresh=0.3,  # giảm threshold
#         box_detections_per_img=50
#     )
#     return model


# def build_model(num_classes=6):
#     # Option A: ResNet-50 pretrained + FPN (mAP cao nhất)
#     backbone = resnet_fpn_backbone(
#         backbone_name='resnet50',
#         pretrained=True,  # ImageNet weights
#         trainable_layers=3  # Freeze 2 layer đầu
#     )

#     anchor_generator = AnchorGenerator(
#         sizes=((32,), (64,), (128,), (256,), (512,)),
#         aspect_ratios=((0.5, 1.0, 2.0),) * 5
#     )

#     model = FasterRCNN(
#         backbone=backbone,
#         num_classes=6,
#         rpn_anchor_generator=anchor_generator,

#         # Các tham số RPN — nếu không truyền thì dùng mặc định
#         rpn_pre_nms_top_n_train=2000,  # lấy top 2000 trước NMS lúc train
#         rpn_pre_nms_top_n_test=1000,  # lấy top 1000 trước NMS lúc inference
#         rpn_post_nms_top_n_train=2000,  # giữ tối đa 2000 sau NMS lúc train
#         rpn_post_nms_top_n_test=1000,  # giữ tối đa 1000 sau NMS lúc inference
#         rpn_nms_thresh=0.7,  # IoU > 0.7 thì coi là trùng, bỏ
#         rpn_score_thresh=0.0,  # score tối thiểu để giữ anchor

#         # Box/detection params — bạn đang thiếu phần này
#         box_score_thresh=0.05,  # loại box có score < 0.3
#         box_nms_thresh=0.5,  # NMS lần 2 sau khi head phân loại
#         box_detections_per_img=50,  # tối đa 50 box trên 1 ảnh
#     )
#     return model


# if __name__ == '__main__':
#     x = torch.randn(1, 3, 800, 800)  # 1 ảnh
#     backbone = resnet_fpn_backbone(
#         backbone_name='resnet50',
#         pretrained=True,  # ImageNet weights
#         trainable_layers=3  # Freeze 2 layer đầu
#     )

#     output = backbone(x)
#     print(type(output))
#     # OrderedDict

#     for k, v in output.items():
#         print(f"  [{k}]: {v.shape}")


def build_resnet18_backbone():
    resnet = resnet18(weights=None)  # KHÔNG pretrained

    # bỏ avgpool + fc
    modules = list(resnet.children())[:-2]
    backbone = nn.Sequential(*modules)

    # Faster R-CNN cần biết số channel output
    backbone.out_channels = 512

    return backbone


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class ResNet18FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_resnet18_backbone()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512],
            out_channels=256
        )
        self.out_channels = 256

    def forward(self, x):
        c5 = self.backbone(x)
        feats = {"0": c5}
        return self.fpn(feats)

def build_model(num_classes=6):

    backbone = ResNet18FPN()

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,

        rpn_anchor_generator=anchor_generator,

        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=50
    )

    return model
