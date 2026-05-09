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

        ann_path = os.path.join(self.root, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Không tìm thấy file annotation tại: {ann_path}")

        with open(ann_path) as f:
            coco = json.load(f)

        # ── 1. Xóa trash và other ─────────────────────────
        REMOVE_IDS = {0, 4}
        coco["categories"] = [c for c in coco["categories"] if c["id"] not in REMOVE_IDS]
        valid_cat_ids = {c["id"] for c in coco["categories"]}

        coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] in valid_cat_ids]

        # ── 2. Reindex category ID (Về 0, 1, 2...) ────────────────────────
        old_to_new = {}
        for i, cat in enumerate(coco["categories"]):
            old_to_new[cat["id"]] = i
            cat["id"] = i

        for ann in coco["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        # ── 3. Undersample plastic ────────────────────────
        if split == "train":
            try:
                plastic_id = next(c["id"] for c in coco["categories"] if "plastic" in c["name"].lower())
                plastic_anns = [a for a in coco["annotations"] if a["category_id"] == plastic_id]
                other_anns = [a for a in coco["annotations"] if a["category_id"] != plastic_id]

                random.seed(seed)
                # Giới hạn 500 mẫu plastic để cân bằng dữ liệu
                sampled_plastic = random.sample(plastic_anns, min(500, len(plastic_anns)))
                coco["annotations"] = other_anns + sampled_plastic
            except StopIteration:
                print("Warning: Không tìm thấy category 'plastic' để undersample.")

        # ── 4. Lọc và ánh xạ ảnh ──────────────────────────────────────────
        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        self.images = [img for img in coco["images"] if img["id"] in self.img_to_anns]
        self.categories = coco["categories"]

        # Label bắt đầu từ 1 (0 là background cho Faster RCNN)
        self.cat_id_to_label = {cat["id"]: cat["id"] + 1 for cat in self.categories}

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(os.path.join(self.root, img_info["file_name"])).convert("RGB")

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:  # bỏ bbox lỗi
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_label[ann["category_id"]])

        # Convert sang numpy để Albumentations xử lý
        image_np = np.array(image)

        if self.transforms:
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes,
                labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
        else:
            # Nếu không có transform, ít nhất phải chuyển sang tensor
            from torchvision.transforms import functional as F
            image = F.to_tensor(image)

        # Xử lý trường hợp không có box nào sau khi transform
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        return image, target

    def __len__(self):
        return len(self.images)

    def get_num_classes(self):
        return len(self.categories) + 1


def collate_fn(batch):
    return tuple(zip(*batch))