"""
trainer.py – Training & Validation Loops
==========================================
"""

import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None

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
    writer:     Optional["SummaryWriter"] = None,
) -> dict[str, float]:
    model.train()
    total_losses: dict[str, float] = {}
    n_batches = 0
    t_start   = time.time()

    batches_per_epoch  = len(dataloader)
    global_batch_start = (epoch - 1) * batches_per_epoch

    for batch_idx, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # FasterRCNN from scratch trả về dict loss khi training
        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        # TensorBoard per-batch
        if writer is not None:
            gs = global_batch_start + batch_idx
            for k, v in loss_dict.items():
                writer.add_scalar(f"Train/batch_{k}", v.item(), gs)
            writer.add_scalar("Train/batch_total", losses.item(), gs)

        if (batch_idx + 1) % print_freq == 0 or \
                (batch_idx + 1) == len(dataloader):
            elapsed  = time.time() - t_start
            loss_str = "  ".join(
                f"{k}: {v/n_batches:.4f}" for k, v in total_losses.items()
            )
            print(
                f"[Train] Epoch {epoch:02d}  "
                f"[{batch_idx+1:04d}/{len(dataloader):04d}]  "
                f"Total: {losses.item():.4f}  {loss_str}  "
                f"({elapsed:.1f}s)"
            )

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    avg_losses["total"] = sum(avg_losses.values())

    if writer is not None:
        for k, v in avg_losses.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        writer.add_scalar(
            "Train/learning_rate",
            optimizer.param_groups[0]["lr"], epoch,
        )

    return avg_losses


# ---------------------------------------------------------------------------
# Validation – 1 epoch
# ---------------------------------------------------------------------------

def validate(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    epoch:      int,
    writer:     Optional["SummaryWriter"] = None,
) -> dict[str, float]:
    """
    Tính validation loss.
    model.train() + torch.no_grad() → FasterRCNN trả về loss (không inference).
    """
    model.train()
    total_losses: dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
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

    if writer is not None:
        for k, v in avg_losses.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

    return avg_losses


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
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