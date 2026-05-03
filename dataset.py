"""
  1. DetectionTransforms: transform ĐỒNG THỜI image + boxes (không phải chỉ image)
     → Bắt buộc để augmentation không làm lệch bounding box
  2. Thêm ImageNet normalization (mean/std) – thiếu cái này là nguyên nhân chính
     khiến loss cao và mAP ≈ 0 vì backbone expect input đã normalize
  3. Random horizontal flip với box transform (train only)
  4. Random brightness/contrast (train only) – giúp model robust hơn
  5. Lọc degenerate boxes (w≤0 hoặc h≤0) – gây NaN loss trong RPN
  6. image_id trong target – cần thiết cho torchmetrics MeanAveragePrecision
  7. get_class_weights() – dùng với WeightedRandomSampler để giải quyết imbalance
"""

import json
import os
import random
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────────── Transforms ─────────────────────────────────

class DetectionTransforms:
    """
    Custom transform xử lý cùng lúc image + target (boxes).

    Tại sao không dùng transforms.ToTensor() thông thường?
        → transforms.ToTensor chỉ biến đổi image, không đụng vào boxes.
        → Nếu flip/crop image mà không flip/crop boxes → annotation bị sai.

    Pipeline:
        1. PIL → Tensor (float32, [0,1])
        2. Normalize với ImageNet mean/std
        3. [train only] Random horizontal flip + flip boxes
        4. [train only] Random color jitter (brightness, contrast)
    """

    # ImageNet statistics – dùng khi train từ scratch trên ảnh RGB thông thường
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, is_train: bool = False):
        self.is_train = is_train

    def __call__(
            self,
            image: Image.Image,
            target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:

        # ── 1. PIL → Tensor [C, H, W] float32 in [0, 1] ──────────────
        img = TF.to_tensor(image)

        # ── 2. ImageNet Normalize ──────────────────────────────────────
        img = TF.normalize(img, mean=self.MEAN, std=self.STD)

        if self.is_train:
            _, H, W = img.shape
            boxes = target["boxes"]  # shape (N, 4) – xyxy

            # ── 3. Random Horizontal Flip ──────────────────────────────
            if random.random() > 0.5 and len(boxes) > 0:
                img = TF.hflip(img)
                flipped = boxes.clone()
                flipped[:, 0] = W - boxes[:, 2]  # new x1 = W - old x2
                flipped[:, 2] = W - boxes[:, 0]  # new x2 = W - old x1
                target["boxes"] = flipped

            # ── 4. Random Brightness / Contrast ───────────────────────
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)

        return img, target


# ─────────────────────────── Dataset ────────────────────────────────────

class TrashDataset(torch.utils.data.Dataset):
    """
    TACO dataset reader (COCO JSON format).

    Parameters
    ----------
    root       : thư mục chứa các split (train/, valid/, test/)
    split      : "train" | "valid" | "test"
    transforms : DetectionTransforms
    """

    def __init__(
            self,
            root: str,
            split: str = "train",
            transforms: Optional[DetectionTransforms] = None,
    ):
        self.root = os.path.join(root, split)
        self.transforms = transforms

        ann_path = os.path.join(self.root, "_annotations.coco.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        # category_id (COCO) → label index (1-based; 0 = background)
        self.cat_id_to_label: Dict[int, int] = {
            cat["id"]: i + 1
            for i, cat in enumerate(self.categories)
        }
        self.label_to_name: Dict[int, str] = {
            i + 1: cat["name"]
            for i, cat in enumerate(self.categories)
        }

        # image_id → list of annotations
        self.img_to_anns: Dict[int, list] = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

    # ── Helpers ───────────────────────────────────────────────────────

    def get_class_weights(self) -> List[float]:
        """
        Trả về weight cho mỗi sample để dùng với WeightedRandomSampler.
        Weight = nghịch đảo tần suất class hiếm nhất trong ảnh đó.
        → Giải quyết vấn đề mất cân bằng (plastic 1925 vs other 17).
        """
        # Đếm bbox theo class
        class_count: Dict[int, int] = defaultdict(int)
        for anns in self.img_to_anns.values():
            for ann in anns:
                label = self.cat_id_to_label[ann["category_id"]]
                class_count[label] += 1

        total = sum(class_count.values()) or 1
        # class weight = tổng / số bbox của class đó
        class_weight = {
            cls: total / (cnt + 1e-6)
            for cls, cnt in class_count.items()
        }

        weights: List[float] = []
        for img_info in self.images:
            img_id = img_info["id"]
            anns = self.img_to_anns.get(img_id, [])
            if not anns:
                weights.append(1.0)
            else:
                labels = [self.cat_id_to_label[a["category_id"]] for a in anns]
                # Ảnh chứa class hiếm → weight cao → được sample nhiều hơn
                w = max(class_weight.get(l, 1.0) for l in labels)
                weights.append(w)

        return weights

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.root, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        # ── Parse annotations ──────────────────────────────────────────
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # ── FIX: lọc degenerate box (w≤0 hoặc h≤0 gây NaN loss) ──
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        # ── Apply transforms ───────────────────────────────────────────
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = TF.to_tensor(image)
            image = TF.normalize(image,
                                 mean=DetectionTransforms.MEAN,
                                 std=DetectionTransforms.STD)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


# ─────────────────────────── Collate ────────────────────────────────────

def collate_fn(batch):
    """FasterRCNN cần list of images và list of targets (không stack tensor)."""
    return tuple(zip(*batch))
