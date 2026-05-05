"""
utils.py – Tiện ích tổng hợp
==============================
Gồm:
  • setup_device()          – chọn GPU/CPU tự động
  • create_data_loaders()   – tạo train/val/test DataLoader với split ngẫu nhiên
  • plot_losses()           – vẽ đồ thị loss theo epoch
  • visualize_predictions() – vẽ bounding box lên ảnh (dùng OpenCV)
  • seed_everything()       – đảm bảo kết quả reproducible
"""

import os
import json
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
    SUPERCLASS_NAMES, SUPERCLASS_COLORS,
    CHECKPOINT_DIR, LOG_DIR,
)
from src.dataset import TACODataset, collate_fn
from src.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def setup_device() -> torch.device:
    """Tự động dùng CUDA nếu có, ngược lại dùng CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Utils] Device: {device}")
    if device.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")
        print(f"        VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Đặt seed để kết quả có thể reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_data_loaders(
    annotation_file: str,
    image_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo 3 DataLoader (train / val / test) với stratified split.

    Args:
        annotation_file: đường dẫn annotations.json
        image_dir:       thư mục chứa ảnh (sau preprocess)
        batch_size:      số ảnh mỗi batch
        num_workers:     số process đọc data song song
        seed:            random seed để split reproducible

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # ── Tạo full dataset để lấy tổng số ảnh ──────────────────────────────
    full_ds = TACODataset(annotation_file, image_dir)
    n       = len(full_ds)

    # ── Shuffle index ─────────────────────────────────────────────────────
    indices = list(range(n))
    rng     = random.Random(seed)
    rng.shuffle(indices)

    # ── Tính ngưỡng split ─────────────────────────────────────────────────
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[Utils] Split: train={len(train_idx)} | "
          f"val={len(val_idx)} | test={len(test_idx)}")

    # ── Tạo dataset với transform tương ứng ──────────────────────────────
    train_ds = TACODataset(annotation_file, image_dir,
                           transforms=get_train_transforms(),
                           indices=train_idx)
    val_ds   = TACODataset(annotation_file, image_dir,
                           transforms=get_val_transforms(),
                           indices=val_idx)
    test_ds  = TACODataset(annotation_file, image_dir,
                           transforms=get_val_transforms(),
                           indices=test_idx)

    # ── DataLoader ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Loss logging
# ---------------------------------------------------------------------------

class LossLogger:
    """Lưu và ghi log loss mỗi epoch ra file JSON."""

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.json")
        self.history: list[dict] = []

    def record(self, epoch: int, train_losses: dict, val_losses: dict) -> None:
        entry = {
            "epoch":       epoch,
            "train":       train_losses,
            "val":         val_losses,
        }
        self.history.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def best_epoch(self) -> int:
        """Trả về epoch có val total loss thấp nhất."""
        return min(self.history, key=lambda e: e["val"]["total"])["epoch"]


# ---------------------------------------------------------------------------
# Visualization (OpenCV)
# ---------------------------------------------------------------------------

def draw_predictions(
    image_bgr:   np.ndarray,
    boxes:        torch.Tensor,
    labels:       torch.Tensor,
    scores:       torch.Tensor,
    score_thresh: float = 0.3,
) -> np.ndarray:
    """
    Vẽ bounding box + label + score lên ảnh BGR (OpenCV format).

    Args:
        image_bgr:    ảnh BGR numpy array (H, W, 3)
        boxes:        Tensor [N, 4] – [x1, y1, x2, y2]
        labels:       Tensor [N]
        scores:       Tensor [N]
        score_thresh: chỉ vẽ box có score >= threshold

    Returns:
        ảnh với bounding box đã vẽ
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Cần cài opencv-python: pip install opencv-python")

    img = image_bgr.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        cls_idx  = label.item()
        cls_name = SUPERCLASS_NAMES.get(cls_idx, "unknown")
        color    = SUPERCLASS_COLORS.get(cls_idx, (255, 255, 255))

        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # Vẽ label + score
        text = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, text, (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), thickness=1,
        )

    return img


def plot_losses(log_path: str, save_path: Optional[str] = None) -> None:
    """
    Đọc training_log.json và vẽ đồ thị loss (train vs val).
    Yêu cầu matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cần cài matplotlib: pip install matplotlib")
        return

    with open(log_path) as f:
        history = json.load(f)

    epochs     = [e["epoch"] for e in history]
    train_loss = [e["train"]["total"] for e in history]
    val_loss   = [e["val"]["total"]   for e in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss,   label="Val Loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Utils] Đã lưu biểu đồ loss → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Tạo các thư mục cần thiết nếu chưa có."""
    for d in [CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
