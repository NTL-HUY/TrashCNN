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
  --- TACO official ---
  8. Parse annotations.json của TACO chính thức (flat, không chia sẵn split)
  9. Map tên category TACO → superclass label qua TACO_TO_SUPERCLASS
     → Bỏ qua annotation nếu tên category không có trong mapping
  10. Tự chia train/valid/test theo tỉ lệ 80/10/10 (seed cố định để reproducible)
  11. Resolve đường dẫn ảnh từ data/processed/batch_X/... (ảnh đã resize 600×800)
  --- Cải tiến mAP ---
  12. Random scale (train only): resize ảnh ngẫu nhiên trong khoảng [0.7, 1.3]
      rồi scale boxes tương ứng → model học được multi-scale object tốt hơn
  13. Random crop (train only): cắt ngẫu nhiên sau khi scale, chỉ giữ box
      có diện tích còn lại ≥ 30% so với ban đầu → tránh mất annotation
      Hai augmentation này kết hợp giúp tăng diversity dataset nhỏ (~1.5k ảnh)
      mà không cần thêm data mới
"""

import json
import os
import random
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────── Superclass mapping ─────────────────────────────

TACO_TO_SUPERCLASS: Dict[str, int] = {
    # 1: plastic
    "other plastic bottle": 1,
    "clear plastic bottle": 1,
    "plastic bottle cap": 1,
    "carded blister pack": 1,
    "disposable plastic cup": 1,
    "foam cup": 1,
    "other plastic cup": 1,
    "plastic lid": 1,
    "other plastic": 1,
    "plastic film": 1,
    "six pack rings": 1,
    "garbage bag": 1,
    "other plastic wrapper": 1,
    "single-use carrier bag": 1,
    "polypropylene bag": 1,
    "crisp packet": 1,
    "spread tub": 1,
    "tupperware": 1,
    "disposable food container": 1,
    "foam food container": 1,
    "other plastic container": 1,
    "plastic glooves": 1,
    "plastic utensils": 1,
    "squeezable tube": 1,
    "plastic straw": 1,
    "styrofoam piece": 1,

    # 2: paper
    "toilet tube": 2,
    "other carton": 2,
    "egg carton": 2,
    "drink carton": 2,
    "corrugated carton": 2,
    "meal carton": 2,
    "pizza box": 2,
    "paper cup": 2,
    "magazine paper": 2,
    "tissues": 2,
    "wrapping paper": 2,
    "normal paper": 2,
    "paper bag": 2,
    "plastified paper bag": 2,
    "paper straw": 2,

    # 3: metal
    "aluminium foil": 3,
    "aluminium blister pack": 3,
    "metal bottle cap": 3,
    "food can": 3,
    "aerosol": 3,
    "drink can": 3,
    "metal lid": 3,
    "pop tab": 3,
    "scrap metal": 3,

    # 4: glass
    "glass bottle": 4,
    "broken glass": 4,
    "glass cup": 4,
    "glass jar": 4,

    # 5: other
    "battery": 5,
    "food waste": 5,
    "rope & strings": 5,
    "shoe": 5,
    "unlabeled litter": 5,
    "cigarette": 5,
}

# Tên hiển thị cho từng superclass label
SUPERCLASS_NAMES: Dict[int, str] = {
    1: "plastic",
    2: "paper",
    3: "metal",
    4: "glass",
    5: "other",
}

# Số superclass (không tính background)
NUM_SUPERCLASSES = len(SUPERCLASS_NAMES)  # = 5


# ─────────────────────────── Transforms ─────────────────────────────────

class DetectionTransforms:
    """
    Custom transform xử lý cùng lúc image + target (boxes).

    Tại sao không dùng transforms.ToTensor() thông thường?
        → transforms.ToTensor chỉ biến đổi image, không đụng vào boxes.
        → Nếu flip/crop image mà không flip/crop boxes → annotation bị sai.

    Pipeline (train):
        1. PIL → Tensor (float32, [0,1])
        2. Normalize với ImageNet mean/std
        3. Random scale [0.7, 1.3] + scale boxes
        4. Random crop (giữ box có overlap ≥ 30%)
        5. Random horizontal flip + flip boxes
        6. Random color jitter (brightness, contrast)

    Pipeline (val/test):
        1. PIL → Tensor
        2. Normalize
    """

    # ImageNet statistics – dùng khi train từ scratch trên ảnh RGB thông thường
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Random crop: giữ box nếu diện tích còn lại ≥ ngưỡng này
    MIN_BOX_OVERLAP = 0.3

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

            # ── 3. Random Scale ────────────────────────────────────────
            # Resize ảnh về scale ngẫu nhiên, scale boxes tương ứng
            # → Model học được object ở nhiều kích thước khác nhau
            scale = random.uniform(0.7, 1.3)
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            img = TF.resize(img, [new_H, new_W])
            if len(boxes) > 0:
                boxes = boxes * scale  # scale tất cả tọa độ đồng đều
            H, W = new_H, new_W

            # ── 4. Random Crop ─────────────────────────────────────────
            # Chọn vùng crop ngẫu nhiên (tối thiểu 60% kích thước ảnh)
            # → Buộc model học nhận dạng object bị crop một phần
            crop_h = random.randint(int(H * 0.6), H)
            crop_w = random.randint(int(W * 0.6), W)
            top  = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)

            img = TF.crop(img, top, left, crop_h, crop_w)

            if len(boxes) > 0:
                # Clip boxes vào vùng crop
                clipped = boxes.clone()
                clipped[:, 0] = (boxes[:, 0] - left).clamp(min=0, max=crop_w)
                clipped[:, 1] = (boxes[:, 1] - top ).clamp(min=0, max=crop_h)
                clipped[:, 2] = (boxes[:, 2] - left).clamp(min=0, max=crop_w)
                clipped[:, 3] = (boxes[:, 3] - top ).clamp(min=0, max=crop_h)

                # Tính diện tích còn lại / diện tích ban đầu
                orig_area  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                clip_area  = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
                keep = (clip_area / (orig_area + 1e-6)) >= self.MIN_BOX_OVERLAP

                # Lọc degenerate box sau crop (w≤0 hoặc h≤0)
                keep &= (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])

                target["boxes"]  = clipped[keep]
                target["labels"] = target["labels"][keep]
                boxes = target["boxes"]

            H, W = crop_h, crop_w

            # ── 5. Random Horizontal Flip ──────────────────────────────
            if random.random() > 0.5 and len(boxes) > 0:
                img = TF.hflip(img)
                flipped = boxes.clone()
                flipped[:, 0] = W - boxes[:, 2]  # new x1 = W - old x2
                flipped[:, 2] = W - boxes[:, 0]  # new x2 = W - old x1
                target["boxes"] = flipped

            # ── 6. Random Brightness / Contrast ───────────────────────
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)

        return img, target


# ─────────────────────────── Dataset ────────────────────────────────────

class TrashDataset(torch.utils.data.Dataset):
    """
    TACO official dataset reader.

    Đọc annotations.json của TACO chính thức, map tên category → superclass
    qua TACO_TO_SUPERCLASS, tự chia train/valid/test (80/10/10).
    Ảnh được load từ data/processed/batch_X/... (đã resize 600×800).

    Parameters
    ----------
    root         : thư mục gốc chứa annotations.json và thư mục processed/
                   (mặc định là "data/" cùng cấp với train.py)
    split        : "train" | "valid" | "test"
    transforms   : DetectionTransforms
    train_ratio  : tỉ lệ chia train (default 0.8)
    val_ratio    : tỉ lệ chia valid (default 0.1) → phần còn lại là test
    seed         : seed cố định để split reproducible
    """

    # Expose để train.py / deploy.py lấy metadata
    categories = [
        {"id": label, "name": name}
        for label, name in SUPERCLASS_NAMES.items()
    ]

    def __init__(
            self,
            root: str = "data",
            split: str = "train",
            transforms: Optional[DetectionTransforms] = None,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            seed: int = 42,
    ):
        self.processed_root = os.path.join(root, "processed")
        self.transforms = transforms

        ann_path = os.path.join(root, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # ── Build category_id → superclass label ──────────────────────
        # Chỉ giữ lại category có trong TACO_TO_SUPERCLASS
        # category_id (TACO) → superclass label (1-based; 0 = background)
        self._cat_id_to_label: Dict[int, int] = {}
        for cat in coco["categories"]:
            name_lower = cat["name"].lower().strip()
            if name_lower in TACO_TO_SUPERCLASS:
                self._cat_id_to_label[cat["id"]] = TACO_TO_SUPERCLASS[name_lower]

        # ── Build image_id → list of valid annotations ─────────────────
        # "valid" = category có trong mapping và bbox không degenerate
        img_to_anns_all: Dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] not in self._cat_id_to_label:
                continue  # bỏ qua category không có trong mapping
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # FIX: lọc degenerate box gây NaN loss
            img_to_anns_all[ann["image_id"]].append(ann)

        # ── Chỉ giữ ảnh có ít nhất 1 annotation hợp lệ ────────────────
        valid_image_ids = set(img_to_anns_all.keys())
        all_images = [img for img in coco["images"] if img["id"] in valid_image_ids]

        # ── Chia split (seed cố định → reproducible) ──────────────────
        rng = random.Random(seed)
        shuffled = all_images[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            self.images = shuffled[:n_train]
        elif split == "valid":
            self.images = shuffled[n_train: n_train + n_val]
        else:  # test
            self.images = shuffled[n_train + n_val:]

        # ── Giữ lại img_to_anns chỉ cho split hiện tại ────────────────
        split_ids = {img["id"] for img in self.images}
        self.img_to_anns: Dict[int, list] = {
            img_id: anns
            for img_id, anns in img_to_anns_all.items()
            if img_id in split_ids
        }

        # Alias để deploy.py / train.py dùng như cũ
        self.cat_id_to_label = self._cat_id_to_label
        self.label_to_name: Dict[int, str] = SUPERCLASS_NAMES.copy()

    # ── Resolve đường dẫn ảnh ─────────────────────────────────────────

    def _img_path(self, file_name: str) -> str:
        """
        TACO file_name có dạng "batch_X/name.jpg".
        Ảnh processed nằm tại: <processed_root>/batch_X/name.jpg
        """
        return os.path.join(self.processed_root, file_name)

    # ── Helpers ───────────────────────────────────────────────────────

    def get_class_weights(self) -> List[float]:
        """
        Trả về weight cho mỗi sample để dùng với WeightedRandomSampler.
        Weight = nghịch đảo tần suất class hiếm nhất trong ảnh đó.
        → Giải quyết vấn đề mất cân bằng (plastic >> other, glass).
        """
        # Đếm bbox theo superclass label
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
        img_path = self._img_path(img_info["file_name"])

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