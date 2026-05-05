"""
dataset.py – TACO Dataset Loader
==================================
Luồng dữ liệu:
  1. Đọc annotations.json (định dạng COCO)
  2. Remap 60 category TACO → 5 superclass (plastic/paper/metal/glass/other)
  3. Trả về (image_tensor, target_dict) đúng format mà FasterRCNN yêu cầu

Format target_dict:
  {
    "boxes":   FloatTensor[N, 4]  – [x1, y1, x2, y2]
    "labels":  Int64Tensor[N]     – superclass index (1–5, 0 = background)
    "image_id": IntTensor[1]
    "area":    FloatTensor[N]
    "iscrowd": UInt8Tensor[N]
  }
"""

import os
import json
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import (
    DATA_DIR,
    PROCESSED_DIR,
    ANNOTATION_FILE,
    TACO_TO_SUPERCLASS,
    SUPERCLASS_NAMES,
)


# ---------------------------------------------------------------------------
# Helper: remap TACO category id → superclass index
# ---------------------------------------------------------------------------

def build_category_remap(coco_categories: list[dict]) -> dict[int, int]:
    """
    Xây dựng dict: {taco_category_id → superclass_index}.

    Args:
        coco_categories: danh sách category từ annotations.json
                         mỗi phần tử có dạng {"id": int, "name": str, ...}

    Returns:
        remap: dict ánh xạ taco_id → superclass_index (1–5)
               category không tìm thấy → 5 (other)
    """
    remap: dict[int, int] = {}
    unmapped: list[str] = []

    for cat in coco_categories:
        cat_name = cat["name"].strip().lower()
        superclass_idx = TACO_TO_SUPERCLASS.get(cat_name, 5)  # default: other
        remap[cat["id"]] = superclass_idx

        if cat_name not in TACO_TO_SUPERCLASS:
            unmapped.append(cat["name"])

    if unmapped:
        print(f"[Dataset] {len(unmapped)} category không có trong mapping, "
              f"gán về 'other': {unmapped}")

    return remap


# ---------------------------------------------------------------------------
# TACODataset
# ---------------------------------------------------------------------------

class TACODataset(Dataset):
    """
    Dataset đọc TACO theo định dạng COCO.

    Args:
        annotation_file: đường dẫn tới annotations.json
        image_dir:        thư mục gốc chứa ảnh (data/processed hoặc data/)
        transforms:       transform áp dụng lên (image, target)
        indices:          danh sách index ảnh dùng cho split (train/val/test)
    """

    def __init__(
        self,
        annotation_file: str = ANNOTATION_FILE,
        image_dir: str = PROCESSED_DIR,
        transforms: Optional[Callable] = None,
        indices: Optional[list[int]] = None,
    ) -> None:
        self.image_dir  = image_dir
        self.transforms = transforms

        # ── Đọc COCO annotations ──────────────────────────────────────────
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # Remap category
        self.category_remap = build_category_remap(coco["categories"])

        # Build lookup: image_id → image_info
        self.images: dict[int, dict] = {img["id"]: img for img in coco["images"]}

        # Build lookup: image_id → list of annotations
        self.ann_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(ann)

        # Danh sách image_id còn annotation hợp lệ
        all_ids = [
            img_id for img_id in self.images
            if img_id in self.ann_by_image
        ]

        # Áp dụng split (train/val/test)
        if indices is not None:
            self.image_ids = [all_ids[i] for i in indices if i < len(all_ids)]
        else:
            self.image_ids = all_ids

        print(f"[Dataset] Loaded {len(self.image_ids)} images "
              f"from '{os.path.basename(image_dir)}'")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        # ── Đọc ảnh ───────────────────────────────────────────────────
        # TACO lưu ảnh trong data/batch_X/<filename>
        # Sau khi preprocess → data/processed/batch_X/<filename>
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            # Fallback: thư mục gốc DATA_DIR
            img_path = os.path.join(DATA_DIR, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        # ── Parse annotations ─────────────────────────────────────────
        boxes, labels, areas, iscrowd = [], [], [], []

        img_w = img_info.get("width",  float("inf"))
        img_h = img_info.get("height", float("inf"))

        for ann in self.ann_by_image.get(img_id, []):
            # COCO bbox: [x, y, width, height] → chuyển sang [x1,y1,x2,y2]
            x, y, w, h = ann["bbox"]

            # ── Lọc box lỗi từ TACO annotation ───────────────────────
            # Bỏ qua nếu width hoặc height <= 0 (box suy biến)
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(img_w), float(x + w))
            y2 = min(float(img_h), float(y + h))

            # Sau khi clip, kiểm tra lại: x2 > x1 và y2 > y1
            # (box có thể nằm hoàn toàn ngoài biên ảnh sau clip)
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            superclass = self.category_remap.get(ann["category_id"], 5)
            boxes.append([x1, y1, x2, y2])
            labels.append(superclass)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        # ── Đóng gói target ───────────────────────────────────────────
        target = {
            "boxes":    torch.as_tensor(boxes,   dtype=torch.float32)
                        if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.as_tensor(labels,  dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area":     torch.as_tensor(areas,   dtype=torch.float32),
            "iscrowd":  torch.as_tensor(iscrowd, dtype=torch.uint8),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    # ------------------------------------------------------------------
    def get_class_distribution(self) -> dict[str, int]:
        """Thống kê số annotation theo superclass (hữu ích để debug)."""
        dist: dict[int, int] = {i: 0 for i in range(6)}
        for img_id in self.image_ids:
            for ann in self.ann_by_image.get(img_id, []):
                sc = self.category_remap.get(ann["category_id"], 5)
                dist[sc] += 1
        return {SUPERCLASS_NAMES[k]: v for k, v in dist.items()}


# ---------------------------------------------------------------------------
# Collate function (dùng trong DataLoader)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    FasterRCNN yêu cầu batch là list of (image, target),
    không phải tensor được stack lại vì ảnh có kích thước khác nhau.

    Lọc bỏ các sample không còn box hợp lệ sau transforms để tránh
    AssertionError "All bounding boxes should have positive height and width".
    """
    filtered = []
    for img, tgt in batch:
        boxes = tgt["boxes"]
        # Giữ sample chỉ khi còn ít nhất 1 box có w>0 và h>0
        if (boxes.shape[0] > 0
                and (boxes[:, 2] - boxes[:, 0]).min() >= 1.0
                and (boxes[:, 3] - boxes[:, 1]).min() >= 1.0):
            filtered.append((img, tgt))

    # Fallback: nếu toàn batch bị lọc hết thì giữ batch gốc
    if not filtered:
        filtered = batch

    return tuple(zip(*filtered))