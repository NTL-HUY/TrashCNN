"""
trainer.py – Training & Validation Loops
==========================================
Chứa hai hàm chính:
  • train_one_epoch()  – 1 epoch training, trả về dict loss
  • validate()         – 1 epoch validation (chỉ tính loss, không tính mAP)

Note: FasterRCNN của torchvision khi ở mode train() TỰ tính loss
từ cặp (images, targets). Không cần tự viết loss function.

Loss trả về gồm:
  loss_classifier   – phân loại bounding box
  loss_box_reg      – hồi quy tọa độ box
  loss_objectness   – RPN phân biệt foreground/background
  loss_rpn_box_reg  – RPN hồi quy proposal
"""

import time
from typing import Optional

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Training – 1 epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    dataloader: DataLoader,
    device:     torch.device,
    epoch:      int,
    print_freq: int = 20,
) -> dict[str, float]:
    """
    Chạy 1 epoch training.

    Args:
        model:      FasterRCNN đang ở chế độ feature extraction
        optimizer:  SGD optimizer (chỉ update trainable params)
        dataloader: DataLoader training
        device:     cuda hoặc cpu
        epoch:      số epoch hiện tại (để log)
        print_freq: in log sau mỗi N batch

    Returns:
        avg_losses: dict {tên_loss: giá_trị_trung_bình} cho cả epoch
    """
    model.train()
    total_losses: dict[str, float] = {}
    n_batches = 0
    t_start = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        # ── Chuyển dữ liệu lên device ──────────────────────────────────
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ── Forward pass → FasterRCNN trả về dict loss ─────────────────
        loss_dict = model(images, targets)
        # loss_dict = {
        #   "loss_classifier": ...,
        #   "loss_box_reg": ...,
        #   "loss_objectness": ...,
        #   "loss_rpn_box_reg": ...
        # }

        losses = sum(loss_dict.values())   # tổng loss

        # ── Backward + update ──────────────────────────────────────────
        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping: tránh exploding gradient (đặc biệt khi train head)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=5.0
        )
        optimizer.step()

        # ── Tích lũy loss để tính trung bình ──────────────────────────
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        # ── Log định kỳ ───────────────────────────────────────────────
        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader):
            elapsed = time.time() - t_start
            loss_str = "  ".join(
                f"{k}: {v/n_batches:.4f}" for k, v in total_losses.items()
            )
            print(
                f"[Train] Epoch {epoch:02d}  "
                f"[{batch_idx+1:04d}/{len(dataloader):04d}]  "
                f"Total: {losses.item():.4f}  {loss_str}  "
                f"({elapsed:.1f}s)"
            )

    # Trả về loss trung bình cho cả epoch
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    avg_losses["total"] = sum(avg_losses.values())
    return avg_losses


# ---------------------------------------------------------------------------
# Validation – 1 epoch (tính loss, không tính mAP ở đây)
# ---------------------------------------------------------------------------

def validate(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    """
    Tính validation loss.
    Lưu ý: để tính loss, model vẫn phải ở mode .train()
    nhưng không gọi backward(). Đây là đặc điểm của FasterRCNN torchvision.

    Returns:
        avg_losses: dict loss trung bình trên validation set
    """
    # FasterRCNN torchvision chỉ trả loss khi model.training = True
    model.train()
    total_losses: dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():   # không tính gradient → tiết kiệm memory
        for images, targets in dataloader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    avg_losses["total"] = sum(avg_losses.values())

    loss_str = "  ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
    print(f"[Val]   Epoch {epoch:02d}  {loss_str}")

    return avg_losses


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Dừng training sớm nếu validation loss không cải thiện sau `patience` epoch.
    Lưu checkpoint tốt nhất (best_model_path).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Returns:
            True nếu nên dừng training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Không cải thiện "
                  f"{self.counter}/{self.patience} lần")
            if self.counter >= self.patience:
                self.should_stop = True
                print("[EarlyStopping] Dừng training sớm!")

        return self.should_stop
