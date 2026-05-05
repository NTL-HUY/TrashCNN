"""
transforms.py – Tiền xử lý & Data Augmentation
=================================================
Tất cả transform phải xử lý đồng thời image VÀ target (bounding box)
để đảm bảo bounding box luôn khớp với ảnh sau khi biến đổi.

Pipeline:
  Train : Resize → RandomHorizontalFlip → ToTensor → Normalize
  Val   : Resize → ToTensor → Normalize
  Test  : Resize → ToTensor → Normalize
"""

import random
from typing import Optional

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF

from src.config import IMAGE_MAX_SIZE, IMAGE_MIN_SIZE, MEAN, STD


# ---------------------------------------------------------------------------
# Utility: resize ảnh + scale bounding box tương ứng
# ---------------------------------------------------------------------------

def resize_image_and_boxes(
    image,          # PIL.Image
    target: dict,
    min_size: int = IMAGE_MIN_SIZE,
    max_size: int = IMAGE_MAX_SIZE,
):
    """
    Resize ảnh sao cho:
      - cạnh ngắn ≥ min_size
      - cạnh dài  ≤ max_size
    Bounding box được scale theo tỉ lệ tương ứng.
    """
    w, h    = image.size          # PIL: (width, height)
    scale   = min_size / min(w, h)

    # Kiểm tra cạnh dài có vượt max_size không
    if scale * max(w, h) > max_size:
        scale = max_size / max(w, h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    image = image.resize((new_w, new_h))

    # Scale + clean bounding boxes
    if target is not None and len(target["boxes"]) > 0:
        boxes = target["boxes"] * scale

        # Clip vào biên ảnh mới
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

        # ── Loại bỏ box suy biến sau khi clip ────────────────────────
        # Ví dụ: box nằm sát mép phải → sau clip x1 == x2 → width = 0
        keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & \
               (boxes[:, 3] - boxes[:, 1] >= 1.0)

        target["boxes"]   = boxes[keep]
        target["labels"]  = target["labels"][keep]
        target["area"]    = target["area"][keep]
        target["iscrowd"] = target["iscrowd"][keep]

    return image, target


# ---------------------------------------------------------------------------
# Các transform cơ bản (áp dụng đồng thời image + target)
# ---------------------------------------------------------------------------

class Resize:
    """Resize ảnh, scale bbox tương ứng."""

    def __init__(
        self,
        min_size: int = IMAGE_MIN_SIZE,
        max_size: int = IMAGE_MAX_SIZE,
    ):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        return resize_image_and_boxes(image, target, self.min_size, self.max_size)


class RandomHorizontalFlip:
    """Lật ngang ảnh + điều chỉnh bbox theo chiều ngang."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            w, _ = image.size
            image = F.hflip(image)

            if len(target["boxes"]) > 0:
                boxes = target["boxes"]
                # x1_new = w - x2_old, x2_new = w - x1_old
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return image, target


class RandomBrightness:
    """Ngẫu nhiên thay đổi độ sáng (augment nhẹ, không ảnh hưởng bbox)."""

    def __init__(self, factor: float = 0.2):
        self.factor = factor   # brightness ∈ [1-factor, 1+factor]

    def __call__(self, image, target):
        factor = 1.0 + random.uniform(-self.factor, self.factor)
        image  = F.adjust_brightness(image, factor)
        return image, target


class ToTensor:
    """Chuyển PIL.Image → torch.FloatTensor [C, H, W], scale về [0, 1]."""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """
    Chuẩn hóa theo mean/std của ImageNet.
    Áp dụng SAU ToTensor vì cần tensor đầu vào.
    """

    def __init__(self, mean=MEAN, std=STD):
        self.mean = mean
        self.std  = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# ---------------------------------------------------------------------------
# Compose: ghép nhiều transform lại
# ---------------------------------------------------------------------------

class Compose:
    """Chạy tuần tự danh sách transform, mỗi transform nhận (image, target)."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ---------------------------------------------------------------------------
# Pipeline xuất ra ngoài
# ---------------------------------------------------------------------------

def get_train_transforms() -> Compose:
    """
    Pipeline training:
      Resize → Lật ngang (p=0.5) → Đổi sáng nhẹ → ToTensor → Normalize
    """
    return Compose([
        Resize(),
        RandomHorizontalFlip(prob=0.5),
        RandomBrightness(factor=0.15),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms() -> Compose:
    """
    Pipeline validation / test:
      Resize → ToTensor → Normalize  (không augment)
    """
    return Compose([
        Resize(),
        ToTensor(),
        Normalize(),
    ])