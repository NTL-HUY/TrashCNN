"""
dataset.py - TACO Dataset Loader for Waste Detection
Expects data already downloaded via TACO's official download.py.

Directory layout expected (cùng cấp với train.py):
  data/
    annotations.json          ← file annotation gốc từ TACO download.py
    batch_1/                  ← ảnh gốc (không dùng để train)
    batch_2/
    ...
    processed/
      batch_1/                ← ảnh đã tiền xử lí (600×800), dùng để train
      batch_2/
      ...

Usage:
  from dataset import build_dataloaders, NUM_CLASSES, TARGET_CLASSES
  train_loader, val_loader, test_loader = build_dataloaders(data_dir="data")
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TACO → 5-class mapping
# Maps TACO supercategory names → 5 classes
# ─────────────────────────────────────────────
TACO_SUPERCATEGORY_MAP = {
    # --- METAL ---
    "Aluminium foil": "metal",
    "Aluminium blister pack": "metal",
    "Metal bottle cap": "metal",
    "Food Can": "metal",
    "Aerosol": "metal",
    "Drink can": "metal",
    "Metal lid": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",

    # --- PAPER ---
    "Carded blister pack": "paper",
    "Toilet tube": "paper",
    "Other carton": "paper",
    "Egg carton": "paper",
    "Drink carton": "paper",
    "Corrugated carton": "paper",
    "Meal carton": "paper",
    "Pizza box": "paper",
    "Paper cup": "paper",
    "Magazine paper": "paper",
    "Tissues": "paper",
    "Wrapping paper": "paper",
    "Normal paper": "paper",
    "Paper bag": "paper",
    "Paper straw": "paper",

    # --- PLASTIC ---
    "Other plastic bottle": "plastic",
    "Clear plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Other plastic": "plastic",
    "Plastified paper bag": "plastic",
    "Plastic film": "plastic",
    "Six pack rings": "plastic",
    "Garbage bag": "plastic",
    "Other plastic wrapper": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Crisp packet": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Disposable food container": "plastic",
    "Foam food container": "plastic",
    "Other plastic container": "plastic",
    "Plastic glooves": "plastic",
    "Plastic utensils": "plastic",
    "Squeezable tube": "plastic",
    "Plastic straw": "plastic",
    "Styrofoam piece": "plastic",

    # --- OTHER ---
    "Battery": "other",
    "Food waste": "other",
    "Rope & strings": "other",
    "Shoe": "other",
    "Unlabeled litter": "other",
    "Cigarette": "other",
    "Glass bottle": "other",
    "Broken glass": "other",
    "Glass cup": "other",
    "Glass jar": "other",
}

TARGET_CLASSES = ["background", "plastic", "metal", "paper", "other"]
CLASS_TO_IDX   = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
NUM_CLASSES    = len(TARGET_CLASSES)


# ─────────────────────────────────────────────
# Parse annotations.json gốc của TACO
# ─────────────────────────────────────────────
def load_taco_annotations(data_dir: Path):
    """
    Đọc annotations.json gốc (do TACO download.py tạo ra),
    lọc chỉ lấy các ảnh/annotation thuộc 5 target class,
    và resolve đường dẫn ảnh sang thư mục processed/.

    Trả về:
        images        : list[dict]  — mỗi dict có thêm key "processed_path"
        img_ann_map   : dict        — image_id → list[annotation]
        cat_id_to_target : dict     — category_id (int) → tên class (str)
    """
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file annotation: {ann_path}\n"
            "Hãy chắc chắn bạn đã chạy download.py của TACO để tải dữ liệu."
        )

    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    # ── Build category_id → target class ──
    cat_id_to_target: dict[int, str] = {}
    for cat in coco_data["categories"]:
        supercategory = cat.get("supercategory", "")
        name          = cat.get("name", "")
        target = (
            TACO_SUPERCATEGORY_MAP.get(supercategory)
            or TACO_SUPERCATEGORY_MAP.get(name)
        )
        if target:
            cat_id_to_target[cat["id"]] = target

    logger.info(f"Mapped {len(cat_id_to_target)} TACO categories → {len(set(cat_id_to_target.values()))} target classes")

    # ── Lọc annotations ──
    valid_anns    = [a for a in coco_data["annotations"] if a["category_id"] in cat_id_to_target]
    valid_img_ids = set(a["image_id"] for a in valid_anns)
    valid_images  = [img for img in coco_data["images"] if img["id"] in valid_img_ids]

    # Class distribution
    class_counts = {cls: 0 for cls in TARGET_CLASSES[1:]}
    for ann in valid_anns:
        class_counts[cat_id_to_target[ann["category_id"]]] += 1
    logger.info(f"Filtered: {len(valid_images)} images | {len(valid_anns)} annotations")
    logger.info(f"Class distribution: {class_counts}")

    # ── Resolve đường dẫn sang processed/ ──
    # annotations.json lưu file_name dạng "batch_N/image.jpg"
    processed_dir = data_dir / "processed"
    missing = 0
    for img_info in valid_images:
        # file_name trong TACO thường có dạng "batch_1/000001.jpg"
        processed_path = processed_dir / img_info["file_name"]
        img_info["processed_path"] = str(processed_path)
        if not processed_path.exists():
            missing += 1

    if missing > 0:
        logger.warning(
            f"{missing}/{len(valid_images)} ảnh không tìm thấy trong {processed_dir}. "
            "Các ảnh này sẽ được thay bằng ảnh trắng khi load."
        )

    # ── Build image_id → annotations ──
    img_ann_map: dict[int, list] = {}
    for ann in valid_anns:
        img_ann_map.setdefault(ann["image_id"], []).append(ann)

    # Chỉ giữ ảnh có ít nhất 1 annotation hợp lệ
    valid_images = [img for img in valid_images if img["id"] in img_ann_map]

    return valid_images, img_ann_map, cat_id_to_target


# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────
class Augmentor:
    """
    Augmentation pipeline tương thích với bounding box.
    Chỉ áp dụng khi training. KHÔNG resize vì ảnh đã được
    tiền xử lí về 600×800 trong thư mục processed/.
    """

    def __call__(self, image: Image.Image, target: dict):
        # Random horizontal flip
        if random.random() > 0.5:
            image, target = self._hflip(image, target)

        # Random brightness / contrast / saturation / hue
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.3:
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        # Random grayscale (hiếm)
        if random.random() > 0.9:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
            image = TF.to_pil_image(np.array(image))

        return image, target

    @staticmethod
    def _hflip(image: Image.Image, target: dict):
        w, _ = image.size
        image = TF.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


# ─────────────────────────────────────────────
# Transform pipeline
# ─────────────────────────────────────────────
def get_transform() -> T.Compose:
    """PIL Image → normalized tensor [C, H, W]."""
    return T.Compose([T.ToTensor()])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class TACODataset(Dataset):
    """
    TACO Dataset cho Faster R-CNN.
    Đọc ảnh từ data/processed/batch_N/ (đã tiền xử lí 600×800).
    Annotation được parse trực tiếp từ data/annotations.json gốc.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transforms=None,
        augment: bool = False,
        seed: int = 42,
        min_area: float = 100.0,
    ):
        self.transforms = transforms
        self.augmentor  = Augmentor() if augment else None
        self.min_area   = min_area

        data_dir = Path(data_dir)
        all_images, self.img_ann_map, self.cat_id_to_target = load_taco_annotations(data_dir)

        # ── Train / Val / Test split (reproducible) ──
        rng = random.Random(seed)
        rng.shuffle(all_images)
        n       = len(all_images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            self.images = all_images[:n_train]
        elif split == "val":
            self.images = all_images[n_train : n_train + n_val]
        elif split == "test":
            self.images = all_images[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}. Phải là 'train', 'val', hoặc 'test'.")

        logger.info(f"[{split.upper()}] {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info   = self.images[idx]
        img_path   = img_info["processed_path"]

        # ── Load ảnh ──
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Không load được ảnh {img_path}: {e}. Dùng ảnh trắng thay thế.")
            image = Image.new("RGB", (800, 600), color=(128, 128, 128))

        w, h = image.size  # width=800, height=600 (đã tiền xử lí)

        # ── Parse annotations ──
        anns = self.img_ann_map.get(img_info["id"], [])
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            target_cls = self.cat_id_to_target.get(ann["category_id"])
            if target_cls is None:
                continue
            # COCO format: [x, y, width, height] → [x1, y1, x2, y2]
            x, y, bw, bh = ann["bbox"]

            # Scale bbox theo tỉ lệ resize đã thực hiện trong tiền xử lí
            # annotations.json vẫn giữ toạ độ gốc → cần scale về 600×800
            orig_w = img_info.get("width",  w)
            orig_h = img_info.get("height", h)
            scale_x = w / orig_w
            scale_y = h / orig_h

            x1 = max(0.0, x  * scale_x)
            y1 = max(0.0, y  * scale_y)
            x2 = min(w,  (x + bw) * scale_x)
            y2 = min(h,  (y + bh) * scale_y)

            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area or x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[target_cls])
            areas.append(area)
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            areas   = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes   = torch.tensor(boxes,   dtype=torch.float32)
            labels  = torch.tensor(labels,  dtype=torch.int64)
            areas   = torch.tensor(areas,   dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_info["id"]]),
            "area":     areas,
            "iscrowd":  iscrowd,
        }

        # ── Augmentation (train only) ──
        if self.augmentor is not None and len(target["boxes"]) > 0:
            image, target = self.augmentor(image, target)

        # ── To tensor ──
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────
def collate_fn(batch):
    """Custom collate: trả về list of images và list of targets."""
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────
def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 4,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Tạo train / val / test DataLoader từ thư mục data/.

    Args:
        data_dir   : thư mục chứa annotations.json và processed/
        batch_size : batch size cho training
        num_workers: số worker cho DataLoader
        train_ratio: tỉ lệ dữ liệu train
        val_ratio  : tỉ lệ dữ liệu validation
        seed       : random seed để reproducible split
    """
    transform = get_transform()

    train_ds = TACODataset(
        data_dir, split="train",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=transform, augment=True, seed=seed,
    )
    val_ds = TACODataset(
        data_dir, split="val",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=transform, augment=False, seed=seed,
    )
    test_ds = TACODataset(
        data_dir, split="test",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=transform, augment=False, seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader