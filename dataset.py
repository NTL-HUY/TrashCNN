import torch
import os
import json
import random
from PIL import Image
from collections import defaultdict

import numpy as np
class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None, seed=42):
        self.root = os.path.join(root, split)
        self.transforms = transforms

        with open(os.path.join(self.root, "_annotations.coco.json")) as f:
            coco = json.load(f)

        # ── 1. Xóa trash và other ─────────────────────────
        REMOVE_IDS = {0, 4}
        coco["categories"] = [c for c in coco["categories"] if c["id"] not in REMOVE_IDS]
        coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] not in REMOVE_IDS]

        # ── 2. Reindex category ID ────────────────────────
        old_to_new = {}
        for i, cat in enumerate(coco["categories"]):
            old_to_new[cat["id"]] = i
            cat["id"] = i
        for ann in coco["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        # ── 3. Undersample plastic ────────────────────────
        if split == "train":
            plastic_id = next(c["id"] for c in coco["categories"] if c["name"] == "plastic")
            plastic = [a for a in coco["annotations"] if a["category_id"] == plastic_id]
            others = [a for a in coco["annotations"] if a["category_id"] != plastic_id]
            random.seed(seed)
            coco["annotations"] = others + random.sample(plastic, min(500, len(plastic)))

        # ── 4. Lọc ảnh còn annotation ────────────────────
        used_ids = {a["image_id"] for a in coco["annotations"]}
        coco["images"] = [img for img in coco["images"] if img["id"] in used_ids]

        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)
        # ── 5. Gán vào self ───────────────────────────────
        self.images = [
            img for img in coco["images"]
            if len(self.img_to_anns.get(img["id"], [])) > 0
        ]
        self.categories = coco["categories"]

        # Label bắt đầu từ 1 (0 là background cho Faster RCNN)
        self.cat_id_to_label = {cat["id"]: cat["id"] + 1 for cat in self.categories}

        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(os.path.join(self.root, img_info["file_name"])).convert("RGB")

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:  # bỏ bbox lỗi
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.zeros((0, 4), dtype=torch.float32) if not boxes \
            else torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        if self.transforms:
            transformed = self.transforms(
                image=np.array(image),
                bboxes=boxes.tolist(),
                labels=labels.tolist()
            )

            image = transformed["image"]

            boxes = torch.tensor(
                transformed["bboxes"],
                dtype=torch.float32
            )

            labels = torch.tensor(
                transformed["labels"],
                dtype=torch.int64
            )

            target["boxes"] = boxes
            target["labels"] = labels

        return image, target

    def __len__(self):
        return len(self.images)

    def get_num_classes(self):
        # +1 cho background
        return len(self.categories) + 1


def collate_fn(batch):
    return tuple(zip(*batch))

