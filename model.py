import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


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


def build_model(num_classes=7):
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = FasterRCNN(
        backbone=SimpleBackbone(),
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.5,
        box_detections_per_img=20
    )
    return model