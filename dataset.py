"""
dataset.py - TACO Dataset Download, Filter & DataLoader
Filters only 4 classes: plastic, metal, paper, glass
Auto-downloads from TACO GitHub and maps supercategories to 4 target classes
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

    # --- GLASS ---
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",

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
}

TARGET_CLASSES = ["background", "plastic", "metal", "paper", "glass", "other"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)  # 6 (including background)

# TACO annotation JSON URL (official)
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TACO_DOWNLOAD_SCRIPT = "https://raw.githubusercontent.com/pedropro/TACO/master/data/download.py"


# ─────────────────────────────────────────────
# Download Utilities
# ─────────────────────────────────────────────
def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return False


def download_taco_dataset(data_dir: str = "data/taco") -> str:
    """
    Downloads TACO annotations + images for only 5 target classes.
    Returns path to processed annotations JSON.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    ann_path = data_dir / "annotations.json"
    filtered_ann_path = data_dir / "annotations_filtered.json"

    # ── Step 1: Download annotations ──
    if not ann_path.exists():
        logger.info("Downloading TACO annotations...")
        ok = download_file(TACO_ANNOTATIONS_URL, ann_path, "annotations.json")
        if not ok:
            raise RuntimeError("Failed to download TACO annotations. Check internet connection.")
    else:
        logger.info(f"Annotations already exist at {ann_path}")

    # ── Step 2: Parse & filter annotations ──
    logger.info("Parsing and filtering annotations for 5 target classes...")
    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    # Build category_id → target class mapping
    cat_id_to_target = {}
    for cat in coco_data["categories"]:
        supercategory = cat.get("supercategory", "")
        name = cat.get("name", "")
        target = TACO_SUPERCATEGORY_MAP.get(supercategory) or TACO_SUPERCATEGORY_MAP.get(name)
        if target:
            cat_id_to_target[cat["id"]] = target

    logger.info(f"Mapped {len(cat_id_to_target)} TACO categories to 5 target classes")

    # Filter annotations
    valid_anns = [a for a in coco_data["annotations"] if a["category_id"] in cat_id_to_target]
    valid_img_ids = set(a["image_id"] for a in valid_anns)
    valid_images = [img for img in coco_data["images"] if img["id"] in valid_img_ids]

    logger.info(f"Filtered: {len(valid_images)} images | {len(valid_anns)} annotations")

    # Class distribution
    class_counts = {cls: 0 for cls in TARGET_CLASSES[1:]}
    for ann in valid_anns:
        cls = cat_id_to_target[ann["category_id"]]
        class_counts[cls] += 1
    logger.info(f"Class distribution: {class_counts}")

    # ── Step 3: Download images ──
    logger.info(f"Downloading {len(valid_images)} images...")
    failed = []
    for img_info in tqdm(valid_images, desc="Downloading images"):
        img_path = images_dir / img_info["file_name"].replace("/", "_")
        if img_path.exists():
            continue
        url = img_info.get("flickr_url") or img_info.get("coco_url", "")
        if not url:
            failed.append(img_info["id"])
            continue
        ok = download_file(url, img_path, desc="")
        if not ok:
            failed.append(img_info["id"])

    if failed:
        logger.warning(f"Failed to download {len(failed)} images. They will be skipped.")
        # Remove images and annotations for failed downloads
        failed_set = set(failed)
        valid_images = [img for img in valid_images if img["id"] not in failed_set]
        valid_anns = [a for a in valid_anns if a["image_id"] not in failed_set]

    # Update file_name to local paths
    for img_info in valid_images:
        img_info["local_path"] = str(images_dir / img_info["file_name"].replace("/", "_"))

    # ── Step 4: Save filtered annotations ──
    filtered_data = {
        "images": valid_images,
        "annotations": valid_anns,
        "categories": [
            {"id": CLASS_TO_IDX[cls], "name": cls, "supercategory": cls}
            for cls in TARGET_CLASSES[1:]
        ],
        "cat_id_to_target": {str(k): v for k, v in cat_id_to_target.items()},
        "original_categories": coco_data["categories"],
    }
    with open(filtered_ann_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    logger.info(f"Saved filtered annotations to {filtered_ann_path}")
    logger.info(f"Final: {len(valid_images)} images | {len(valid_anns)} annotations")
    return str(filtered_ann_path)


# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────
class Augmentor:
    """
    Custom augmentation pipeline compatible with bounding boxes.
    Applied only during training.
    """

    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # ── Resize (always applied, before any augmentation) ──
        image, target = self._resize(image, target)

        # Random horizontal flip
        if random.random() > 0.5:
            image, target = self._hflip(image, target)

        # Random brightness/contrast/saturation
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.3))
        if random.random() > 0.3:
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        # Random grayscale
        if random.random() > 0.9:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
            image = TF.to_pil_image(np.array(image))

        return image, target

    def _resize(self, image, target):
        w, h = image.size

        # Tính scale factor theo shorter side
        scale = self.min_size / min(w, h)

        # Nếu longer side vượt max_size → thu nhỏ lại
        if max(w, h) * scale > self.max_size:
            scale = self.max_size / max(w, h)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        image = TF.resize(image, [new_h, new_w])

        # Scale bounding boxes theo cùng tỉ lệ
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone().float()
            boxes[:, [0, 2]] *= (new_w / w)   # x1, x2
            boxes[:, [1, 3]] *= (new_h / h)   # y1, y2
            # Clamp để tránh out-of-bound sau khi làm tròn số
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes

        return image, target

    def _hflip(self, image, target):
        w, _ = image.size
        image = TF.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


def resize_image(image: "Image.Image", target: dict, min_size: int = 800, max_size: int = 1333):
    w, h = image.size
    scale = min_size / min(w, h)
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = TF.resize(image, [new_h, new_w])
    if len(target["boxes"]) > 0:
        boxes = target["boxes"].clone().float()
        boxes[:, [0, 2]] *= (new_w / w)
        boxes[:, [1, 3]] *= (new_h / h)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
        target["boxes"] = boxes
    return image, target


def get_transform(train: bool):
    """Returns transform pipeline."""
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────
class TACODataset(Dataset):
    """
    TACO Dataset for Faster R-CNN.
    Loads filtered annotations and returns images + targets.
    """

    def __init__(
            self,
            ann_file: str,
            split: str = "train",
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            transforms=None,
            augment: bool = False,
            seed: int = 42,
            min_area: float = 100.0,  # filter tiny boxes
    ):
        self.transforms = transforms
        self.augment = augment
        self.augmentor = Augmentor() if augment else None
        self.min_area = min_area

        with open(ann_file, "r") as f:
            data = json.load(f)

        self.cat_id_to_target = {int(k): v for k, v in data["cat_id_to_target"].items()}

        # Build image_id → annotations index
        self.img_ann_map = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self.img_ann_map.setdefault(iid, []).append(ann)

        # Filter images that have at least 1 valid annotation
        all_images = [img for img in data["images"] if img["id"] in self.img_ann_map]

        # Reproducible split
        rng = random.Random(seed)
        rng.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            self.images = all_images[:n_train]
        elif split == "val":
            self.images = all_images[n_train:n_train + n_val]
        elif split == "test":
            self.images = all_images[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        logger.info(f"[{split.upper()}] {len(self.images)} images loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = img_info.get("local_path", img_info.get("file_name", ""))

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot load image {img_path}: {e}. Using blank image.")
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        w, h = image.size
        anns = self.img_ann_map.get(img_info["id"], [])

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            target_cls = self.cat_id_to_target.get(ann["category_id"])
            if target_cls is None:
                continue
            x, y, bw, bh = ann["bbox"]
            # COCO format: [x, y, width, height] → [x1, y1, x2, y2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area or x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[target_cls])
            areas.append(area)
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            # Return a dummy sample (handled in collate)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # Resize + Augmentation
        if self.augmentor is not None and len(target["boxes"]) > 0:
            image, target = self.augmentor(image, target)
        else:
            image, target = resize_image(image, target, min_size=800, max_size=1333)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────
def collate_fn(batch):
    """Custom collate: returns list of images and list of targets."""
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────
def build_dataloaders(
        ann_file: str,
        batch_size: int = 4,
        num_workers: int = 2,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
):
    """Build train/val/test DataLoaders."""
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    train_ds = TACODataset(
        ann_file, split="train",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=train_transform, augment=True, seed=seed
    )
    val_ds = TACODataset(
        ann_file, split="val",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=val_transform, augment=False, seed=seed
    )
    test_ds = TACODataset(
        ann_file, split="test",
        train_ratio=train_ratio, val_ratio=val_ratio,
        transforms=val_transform, augment=False, seed=seed
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


# ─────────────────────────────────────────────
# CLI: Download dataset
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & prepare TACO dataset")
    parser.add_argument("--data_dir", type=str, default="data/taco", help="Directory to save dataset")
    args = parser.parse_args()

    ann_file = download_taco_dataset(data_dir=args.data_dir)
    print(f"\n✅ Dataset ready: {ann_file}")
    print(f"   Classes: {TARGET_CLASSES[1:]}")
    print(f"   Num classes (with background): {NUM_CLASSES}")