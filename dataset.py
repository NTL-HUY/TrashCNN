import torch
import os
import json
import random
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class TrainAugmentation:
    def __init__(self, p_flip=0.5, p_color=0.5, p_crop=0.5):
        self.p_flip  = p_flip
        self.p_color = p_color
        self.p_crop  = p_crop

    def __call__(self, image, boxes):
        """
        Args:
            image : PIL.Image
            boxes : torch.Tensor shape (N, 4) — [x1, y1, x2, y2]
        Returns:
            image_tensor : torch.Tensor (3, H, W),
            boxes        : torch.Tensor (N, 4),
        """
        W, H = image.size  # PIL: (width, height)

        # ── 1. Random Horizontal Flip ──────────────────────────────
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            if len(boxes):
                boxes = boxes.clone()
                x1_new = W - boxes[:, 2]
                x2_new = W - boxes[:, 0]
                boxes[:, 0] = x1_new
                boxes[:, 2] = x2_new

        # ── 2. Color Jitter ────────────────────────────────────────
        if random.random() < self.p_color:
            brightness_factor = random.uniform(0.6, 1.4)
            image = TF.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.6, 1.4)
            image = TF.adjust_contrast(image, contrast_factor)

            saturation_factor = random.uniform(0.6, 1.4)
            image = TF.adjust_saturation(image, saturation_factor)

            hue_factor = random.uniform(-0.1, 0.1)
            image = TF.adjust_hue(image, hue_factor)

        # ── 3. Random Grayscale ────────────────────────────────────
        if random.random() < 0.15:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # ── 4. Random Crop ─────────────────────────────────────────
        if random.random() < self.p_crop and len(boxes):
            W2, H2 = image.size

            scale  = random.uniform(0.7, 1.0)
            crop_w = int(W2 * scale)
            crop_h = int(H2 * scale)

            left   = random.randint(0, W2 - crop_w)
            top    = random.randint(0, H2 - crop_h)
            right  = left + crop_w
            bottom = top  + crop_h

            new_boxes = boxes.clone()
            new_boxes[:, 0] = new_boxes[:, 0].clamp(min=left,   max=right)   # x1
            new_boxes[:, 1] = new_boxes[:, 1].clamp(min=top,    max=bottom)  # y1
            new_boxes[:, 2] = new_boxes[:, 2].clamp(min=left,   max=right)   # x2
            new_boxes[:, 3] = new_boxes[:, 3].clamp(min=top,    max=bottom)  # y2

            keep = (new_boxes[:, 2] - new_boxes[:, 0] > 1) & \
                   (new_boxes[:, 3] - new_boxes[:, 1] > 1)

            if keep.sum() > 0:
                image     = TF.crop(image, top, left, crop_h, crop_w)
                new_boxes = new_boxes[keep]
                new_boxes[:, 0] -= left
                new_boxes[:, 2] -= left
                new_boxes[:, 1] -= top
                new_boxes[:, 3] -= top
                boxes = new_boxes

        # ── 5. ToTensor + Normalize ────────────────────────────────
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        return image_tensor, boxes


class ValTransform:
    def __call__(self, image, boxes):
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, boxes


class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", seed=42):
        self.root  = os.path.join(root, split)
        self.split = split

        self.augmentation = TrainAugmentation() if split == "train" else ValTransform()

        with open(os.path.join(self.root, "_annotations.coco.json")) as f:
            coco = json.load(f)

        # ── 1. Xóa trash và other ─────────────────────────
        REMOVE_IDS = {0, 4}
        coco["categories"]  = [c for c in coco["categories"]  if c["id"] not in REMOVE_IDS]
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
            others  = [a for a in coco["annotations"] if a["category_id"] != plastic_id]
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

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(os.path.join(self.root, img_info["file_name"])).convert("RGB")

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.zeros((0, 4), dtype=torch.float32) if not boxes \
            else torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # ── Áp dụng augmentation (bbox-safe) ──────────────────────
        image_tensor, boxes = self.augmentation(image, boxes)

        if len(boxes) < len(labels):
            labels = labels[:len(boxes)]

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        return image_tensor, target

    def __len__(self):
        return len(self.images)

    def get_num_classes(self):
        # +1 cho background
        return len(self.categories) + 1


def collate_fn(batch):
    return tuple(zip(*batch))