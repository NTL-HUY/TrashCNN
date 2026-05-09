import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


# ── Building block ────────────────────────────────────────────────────────────
class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + LeakyReLU — unit cơ bản"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """
    Bottleneck residual block — squeeze channels xuống rồi mở lại.
    Tránh vanishing gradient khi train từ đầu.
    """
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.block = nn.Sequential(
            ConvBnAct(channels, mid, kernel=1, padding=0),   # 1×1 squeeze
            ConvBnAct(mid, channels, kernel=3, padding=1),   # 3×3 expand
        )

    def forward(self, x):
        return x + self.block(x)   # skip connection


class Stage(nn.Module):
    """
    Downsample → n residual blocks.
    Tương tự 1 stage của DarkNet nhưng gọn hơn.
    """
    def __init__(self, in_ch, out_ch, num_blocks):
        super().__init__()
        layers = [ConvBnAct(in_ch, out_ch, stride=2)]   # stride-2 downsample
        for _ in range(num_blocks):
            layers.append(ResBlock(out_ch))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


# ── Backbone ──────────────────────────────────────────────────────────────────
class TrashBackbone(nn.Module):
    """
    Input  : (B, 3, 416, 416)
    Output : dict 4 feature maps → FPN

    Stage | stride | channels | blocks  | lý do
    ------+--------+----------+---------+-----------------------------
    stem  |   2    |    32    |    —    | giảm ngay từ đầu
    s1    |   4    |    64    |    1    | low-level features
    s2    |   8    |   128    |    2    | mid-level
    s3    |  16    |   256    |    4    | semantic features (quan trọng)
    s4    |  32    |   512    |    2    | high-level, nhỏ nên ít block
    """
    def __init__(self):
        super().__init__()

        # Stem: 416 → 208
        self.stem = ConvBnAct(3, 32, kernel=3, stride=2, padding=1)

        # 4 stages — số block tăng ở giữa (nơi có nhiều thông tin nhất)
        self.s1 = Stage(32,  64,  num_blocks=1)   # 208 → 104
        self.s2 = Stage(64,  128, num_blocks=2)   # 104 → 52
        self.s3 = Stage(128, 256, num_blocks=4)   # 52  → 26   ← backbone chính
        self.s4 = Stage(256, 512, num_blocks=2)   # 26  → 13

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=256
        )
        self.out_channels = 256

        # ── Kaiming init — CỰC QUAN TRỌNG khi không pretrain ──────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.s1(x)    # 64ch,  104×104
        c2 = self.s2(c1)   # 128ch,  52×52
        c3 = self.s3(c2)   # 256ch,  26×26
        c4 = self.s4(c3)   # 512ch,  13×13

        return self.fpn({"0": c1, "1": c2, "2": c3, "3": c4})


# ── Full model ────────────────────────────────────────────────────────────────
def build_model(num_classes=6):
    backbone = TrashBackbone()

    anchor_generator = AnchorGenerator(
        sizes=(
            (32,),    # c1 — detect rác nhỏ
            (64,),    # c2
            (128,),   # c3 — detect rác vừa (phổ biến nhất)
            (256,),   # c4 — detect rác lớn
        ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=50,
    )
    return model


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(num_classes=6)
    x = torch.randn(2, 3, 416, 416)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print("boxes :", out[0]["boxes"].shape)
    print("scores:", out[0]["scores"].shape)
    print("labels:", out[0]["labels"].shape)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal params: {total:.2f}M")

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# def get_custom_faster_rcnn(num_classes, my_weights_path):
#     # 1. Tạo backbone FPN
#     backbone = resnet_fpn_backbone('resnet50', pretrained=False)
#
#     # 2. Load weights từ model ResNet50 bạn đã train (Classify)
#     # Lưu ý: Chỉ load phần body, bỏ phần head phân loại cũ
#     custom_state_dict = torch.load(my_weights_path)
#     backbone.body.load_state_dict(custom_state_dict, strict=False)
#
#     # 3. Đóng băng hoàn toàn (Feature Extractor only)
#     for param in backbone.parameters():
#         param.requires_grad = False
#
#     # 4. Gắn vào Faster R-CNN
#     model = FasterRCNN(backbone, num_classes=num_classes)
#     return model