"""
transforms.py – Tiền xử lý & Data Augmentation
=================================================
Pipeline:
  Train : Resize → HFlip → VFlip → ColorJitter → ToTensor → Normalize
  Val   : Resize → ToTensor → Normalize
"""

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as F

from src.config import (
    IMAGE_MAX_SIZE, IMAGE_MIN_SIZE, MEAN, STD,
    AUGMENT_HFLIP_PROB, AUGMENT_VFLIP_PROB,
    AUGMENT_BRIGHTNESS, AUGMENT_CONTRAST,
    AUGMENT_SATURATION, AUGMENT_HUE,
)


# ---------------------------------------------------------------------------
# Utility: resize ảnh + scale bounding box
# ---------------------------------------------------------------------------

def resize_image_and_boxes(image, target, min_size=IMAGE_MIN_SIZE,
                            max_size=IMAGE_MAX_SIZE):
    w, h  = image.size
    scale = min_size / min(w, h)
    if scale * max(w, h) > max_size:
        scale = max_size / max(w, h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h))

    if target is not None and len(target["boxes"]) > 0:
        boxes = target["boxes"] * scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & \
               ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
        target["boxes"]   = boxes[keep]
        target["labels"]  = target["labels"][keep]
        target["area"]    = target["area"][keep]
        target["iscrowd"] = target["iscrowd"][keep]

    return image, target


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class Resize:
    def __init__(self, min_size=IMAGE_MIN_SIZE, max_size=IMAGE_MAX_SIZE):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        return resize_image_and_boxes(image, target, self.min_size, self.max_size)


class RandomHorizontalFlip:
    def __init__(self, prob=AUGMENT_HFLIP_PROB):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            w, _ = image.size
            image = F.hflip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Lật dọc – rác có thể xuất hiện ở mọi góc độ."""

    def __init__(self, prob=AUGMENT_VFLIP_PROB):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h = image.size
            image = F.vflip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class ColorJitter:
    """
    Thay đổi ngẫu nhiên brightness, contrast, saturation, hue.
    """

    def __init__(
        self,
        brightness = AUGMENT_BRIGHTNESS,
        contrast   = AUGMENT_CONTRAST,
        saturation = AUGMENT_SATURATION,
        hue        = AUGMENT_HUE,
    ):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, image, target):
        # Áp dụng theo thứ tự ngẫu nhiên
        fns = []
        if self.brightness > 0:
            b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            fns.append(lambda img, _b=b: F.adjust_brightness(img, _b))
        if self.contrast > 0:
            c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            fns.append(lambda img, _c=c: F.adjust_contrast(img, _c))
        if self.saturation > 0:
            s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            fns.append(lambda img, _s=s: F.adjust_saturation(img, _s))
        if self.hue > 0:
            h = random.uniform(-self.hue, self.hue)
            fns.append(lambda img, _h=h: F.adjust_hue(img, _h))

        random.shuffle(fns)
        for fn in fns:
            image = fn(image)

        return image, target


class RandomGrayscale:
    def __init__(self, prob: float = 0.05):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.to_grayscale(image, num_output_channels=3)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean=MEAN, std=STD):
        self.mean = mean
        self.std  = std

    def __call__(self, image, target):
        return F.normalize(image, self.mean, self.std), target


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def get_train_transforms() -> Compose:
    """
    Pipeline training (mạnh hơn so với pretrained model):
      Resize → HFlip → VFlip → ColorJitter → Grayscale → ToTensor → Normalize
    """
    return Compose([
        Resize(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(),
        RandomGrayscale(prob=0.05),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms() -> Compose:
    """Pipeline val/test: không augment."""
    return Compose([
        Resize(),
        ToTensor(),
        Normalize(),
    ])