"""
utils.py
==============================
Gồm:
  • setup_device()        – chọn GPU/CPU tự động
  • seed_everything()     – đảm bảo kết quả reproducible
  • create_data_loaders() – tạo train/val/test DataLoader
  • LossLogger            – ghi JSON + TensorBoard
  • draw_predictions()    – vẽ bounding box lên ảnh (OpenCV)
  • plot_losses()         – vẽ đồ thị loss chi tiết theo epoch
  • ensure_dirs()         – tạo thư mục cần thiết
"""

import os
import json
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_RATIO, VAL_RATIO,
    RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
    SUPERCLASS_NAMES, SUPERCLASS_COLORS,
    CHECKPOINT_DIR, LOG_DIR,
)
from src.dataset import TACODataset, collate_fn
from src.transforms import get_train_transforms, get_val_transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def setup_device() -> torch.device:
    """Tự động dùng CUDA nếu có, ngược lại dùng CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Utils] Device: {device}")
    if device.type == "cuda":
        print(f"        GPU : {torch.cuda.get_device_name(0)}")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_data_loaders(
    annotation_file: str,
    image_dir:       str,
    batch_size:      int = BATCH_SIZE,
    num_workers:     int = NUM_WORKERS,
    seed:            int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo 3 DataLoader (train / val / test) với random split cố định.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_ds = TACODataset(annotation_file, image_dir)
    n       = len(full_ds)

    indices = list(range(n))
    rng     = random.Random(seed)
    rng.shuffle(indices)

    n_train   = int(n * TRAIN_RATIO)
    n_val     = int(n * VAL_RATIO)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"[Utils] Split: train={len(train_idx)} | "
          f"val={len(val_idx)} | test={len(test_idx)}")

    train_ds = TACODataset(annotation_file, image_dir,
                           transforms=get_train_transforms(),
                           indices=train_idx)
    val_ds   = TACODataset(annotation_file, image_dir,
                           transforms=get_val_transforms(),
                           indices=val_idx)
    test_ds  = TACODataset(annotation_file, image_dir,
                           transforms=get_val_transforms(),
                           indices=test_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Loss logging
# ---------------------------------------------------------------------------

class LossLogger:
    """
    Ghi log loss mỗi epoch ra:
      • logs/training_log.json    (luôn ghi)
      • logs/tb/                  (nếu use_tensorboard=True)

    Dùng trong notebook:
      %load_ext tensorboard
      %tensorboard --logdir logs/tb
    """

    def __init__(
        self,
        log_dir:         str  = LOG_DIR,
        use_tensorboard: bool = True,
        tb_subdir:       str  = "tb",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.json")
        self.history:  list[dict] = []
        self.writer:   Optional["SummaryWriter"] = None

        if use_tensorboard:
            if not _TB_AVAILABLE:
                print("[LossLogger] ⚠  tensorboard chưa cài → pip install tensorboard")
            else:
                tb_dir = os.path.join(log_dir, tb_subdir)
                self.writer = SummaryWriter(log_dir=tb_dir)
                print(f"[LossLogger] TensorBoard → {tb_dir}")
                print(f"             %load_ext tensorboard")
                print(f"             %tensorboard --logdir {tb_dir}")

    def record(
        self,
        epoch:         int,
        train_losses:  dict,
        val_losses:    dict,
        extra_scalars: Optional[dict] = None,
    ) -> None:
        """
        Ghi một epoch.

        Args:
            extra_scalars: ghi thêm vào TB, vd. {"mAP/map_50": 0.45}
                           TB đã được ghi bên trong trainer nên
                           record() chỉ dùng extra_scalars để tránh ghi đôi.
        """
        self.history.append({
            "epoch": epoch,
            "train": train_losses,
            "val":   val_losses,
        })
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        if self.writer is not None and extra_scalars:
            for tag, value in extra_scalars.items():
                self.writer.add_scalar(tag, float(value), epoch)
            self.writer.flush()

    def best_epoch(self) -> int:
        """Epoch có val total loss thấp nhất."""
        return min(self.history, key=lambda e: e["val"]["total"])["epoch"]

    def close(self) -> None:
        """Flush và đóng TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            print("[LossLogger] TensorBoard writer đã đóng.")


# ---------------------------------------------------------------------------
# Visualization (OpenCV)
# ---------------------------------------------------------------------------

def draw_predictions(
    image_bgr:    np.ndarray,
    boxes:        torch.Tensor,
    labels:       torch.Tensor,
    scores:       torch.Tensor,
    score_thresh: float = 0.3,
) -> np.ndarray:
    """
    Vẽ bounding box + label + score lên ảnh BGR (OpenCV).

    Args:
        image_bgr:    numpy array (H, W, 3) BGR
        boxes:        Tensor [N, 4] x1y1x2y2
        labels:       Tensor [N]
        scores:       Tensor [N]
        score_thresh: chỉ vẽ box có score >= threshold
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    img = image_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        cls_idx  = label.item()
        cls_name = SUPERCLASS_NAMES.get(cls_idx, "unknown")
        color    = SUPERCLASS_COLORS.get(cls_idx, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


# ---------------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------------

def plot_losses(
    log_path:  str,
    save_path: Optional[str] = None,
) -> None:
    """
    Đọc training_log.json và vẽ 2 subplots:
      • Trên: Total loss (train vs val)
      • Dưới: Từng loss thành phần (objectness, rpn_box_reg, classifier, box_reg)

    Hữu ích để debug: nếu loss_objectness cao → RPN yếu;
    nếu loss_classifier cao → head chưa học được phân biệt class.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return

    with open(log_path) as f:
        history = json.load(f)

    epochs     = [e["epoch"]            for e in history]
    train_tot  = [e["train"]["total"]   for e in history]
    val_tot    = [e["val"]["total"]     for e in history]

    # Lấy tất cả loss keys (trừ "total")
    loss_keys = [k for k in history[0]["train"] if k != "total"]

    fig, axes = plt.subplots(
        1 + bool(loss_keys), 1,
        figsize=(11, 5 * (1 + bool(loss_keys))),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # ── Plot 1: Total loss ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_tot, "o-", label="Train total", linewidth=2)
    ax.plot(epochs, val_tot,   "s-", label="Val total",   linewidth=2)
    ax.set_ylabel("Total Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── Plot 2: Từng loss thành phần ──────────────────────────────────
    if loss_keys and len(axes) > 1:
        ax2 = axes[1]
        for key in loss_keys:
            train_vals = [e["train"].get(key, 0) for e in history]
            val_vals   = [e["val"].get(key, 0)   for e in history]
            line, = ax2.plot(epochs, train_vals, "o--",
                             label=f"Train {key}", linewidth=1.5)
            ax2.plot(epochs, val_vals, "s-",
                     color=line.get_color(),
                     label=f"Val {key}", linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss thành phần")
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Utils] Đồ thị loss → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Tạo các thư mục cần thiết nếu chưa có."""
    for d in [CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)