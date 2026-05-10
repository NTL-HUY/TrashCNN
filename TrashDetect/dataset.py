
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrashDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms=None,
        annotation_file: str = "_annotations.processed.coco.json",
    ):

        self.split_dir  = Path(root) / split
        self.transforms = transforms

        ann_path = self.split_dir / annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Chạy preprocess.py trước để tạo file này."
            )

        with open(ann_path, encoding="utf-8") as f:
            self.coco = json.load(f)

        self.categories = self.coco["categories"]

        # cat_id (0-based) → label cho model (1-based, 0 = background)
        self.cat_id_to_label = {
            c["id"]: i + 1
            for i, c in enumerate(self.categories)
        }

        self.img_to_anns: dict[int, list] = defaultdict(list)
        for ann in self.coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        self.images = [
            img for img in self.coco["images"]
            if self.img_to_anns.get(img["id"])
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]

        img_path = self.split_dir / img_info["file_name"]
        image    = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for ann in self.img_to_anns[img_info["id"]]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_info["id"]]),
        }

        if self.transforms:
            transformed = self.transforms(
                image   = np.array(image),
                bboxes  = boxes.tolist(),
                labels  = labels.tolist(),
            )
            image          = transformed["image"]
            target["boxes"]  = torch.as_tensor(transformed["bboxes"],  dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"],  dtype=torch.int64)
        else:
            image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, target

    def get_num_classes(self) -> int:
        return len(self.categories) + 1

    def get_label_map(self) -> dict:
        return {
            i + 1: c["name"]
            for i, c in enumerate(self.categories)
        }

    def get_class_names(self):
        return [cat["name"] for cat in self.categories]

def collate_fn(batch):
    return tuple(zip(*batch))