import torch
import os
import json
import random
from PIL import Image
from collections import defaultdict
import numpy as np


class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None, image_size=416, seed=42):
        self.root = os.path.join(root, split)
        self.transforms = transforms
        self.image_size = image_size

        ann_path = os.path.join(self.root, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Không tìm thấy file annotation tại: {ann_path}")

        with open(ann_path) as f:
            coco = json.load(f)

        # 1. Lọc Categories
        REMOVE_IDS = {0, 4}
        coco["categories"] = [c for c in coco["categories"] if c["id"] not in REMOVE_IDS]
        valid_cat_ids = {c["id"] for c in coco["categories"]}
        coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] in valid_cat_ids]

        # 2. Reindex (về 0, 1, 2...)
        old_to_new = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
        for cat in coco["categories"]:
            cat["id"] = old_to_new[cat["id"]]
        for ann in coco["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        # 3. Undersample plastic
        if split == "train":
            plastic_cats = [c["id"] for c in coco["categories"] if "plastic" in c["name"].lower()]
            if plastic_cats:
                p_id = plastic_cats[0]
                p_anns = [a for a in coco["annotations"] if a["category_id"] == p_id]
                o_anns = [a for a in coco["annotations"] if a["category_id"] != p_id]
                random.seed(seed)
                coco["annotations"] = o_anns + random.sample(p_anns, min(500, len(p_anns)))

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
        orig_w, orig_h = image.size

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Tọa độ gốc (chưa scale) theo ảnh gốc
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # Clamp trong biên ảnh gốc
            x1 = max(0.0, min(x1, orig_w))
            y1 = max(0.0, min(y1, orig_h))
            x2 = max(0.0, min(x2, orig_w))
            y2 = max(0.0, min(y2, orig_h))

            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_label[ann["category_id"]])

        if len(boxes) > 0:
            boxes_np = np.array(boxes, dtype=np.float32)
            labels_list = labels
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            labels_list = []

        if self.transforms:
            image_np = np.array(image)
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes_np.tolist() if len(boxes_np) > 0 else [],
                labels=labels_list
            )
            image = transformed["image"]           # Tensor [C, H, W]
            transformed_boxes = transformed["bboxes"]
            transformed_labels = transformed["labels"]

            if len(transformed_boxes) > 0:
                boxes_tensor = torch.as_tensor(transformed_boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(transformed_labels, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_scaled = []
            for x1, y1, x2, y2 in boxes_np:
                boxes_scaled.append([
                    x1 * self.image_size / orig_w,
                    y1 * self.image_size / orig_h,
                    x2 * self.image_size / orig_w,
                    y2 * self.image_size / orig_h,
                ])
            if len(boxes_scaled) > 0:
                boxes_tensor = torch.as_tensor(boxes_scaled, dtype=torch.float32)
                labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_info["id"]])
        }

        return image, target

    def __len__(self):
        return len(self.images)

    def get_num_classes(self):
        return len(self.categories) + 1


def collate_fn(batch):
    return tuple(zip(*batch))