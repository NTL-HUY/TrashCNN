import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import TrashDataset, collate_fn
from model import build_model

# ─────────────────────────── Config ───────────────────────────
DATA_ROOT   = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"
NUM_EPOCHS  = 20
BATCH_SIZE  = 4
LR          = 1e-4
NUM_CLASSES = 7
SAVE_DIR    = "checkpoints"
LOG_DIR     = "runs/trash_detector"
# ──────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device, writer, global_step):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_val = losses.item()
        running_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Log từng loss thành phần lên TensorBoard
        for k, v in loss_dict.items():
            writer.add_scalar(f"Loss/train_{k}", v.item(), global_step)
        writer.add_scalar("Loss/train_total", loss_val, global_step)
        global_step += 1

    avg_loss = running_loss / len(loader)
    return avg_loss, global_step


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Dataset & DataLoader
    dataset = TrashDataset(
        root=DATA_ROOT,
        split="train",
        transforms=transforms.ToTensor()
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Model, optimizer, scheduler
    model     = build_model(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard logs → {LOG_DIR}")
    print(f"  Run: tensorboard --logdir {LOG_DIR}\n")

    best_loss    = float("inf")
    global_step  = 0

    # ── Main training loop ──────────────────────────────────
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        avg_loss, global_step = train_one_epoch(
            model, loader, optimizer, device, writer, global_step
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch-level metrics
        writer.add_scalar("Epoch/avg_loss", avg_loss, epoch)
        writer.add_scalar("Epoch/lr",       current_lr, epoch)

        epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

        # ── Lưu checkpoint mỗi epoch ──
        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch:02d}.pth")
        torch.save(
            {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "avg_loss":   avg_loss,
            },
            ckpt_path,
        )

        # ── Lưu model tốt nhất ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "avg_loss": best_loss,
                },
                best_path,
            )
            tqdm.write(f"  ✔ Best model saved  (epoch {epoch}, loss {best_loss:.4f})")

    writer.close()
    print(f"\nTraining done! Best loss: {best_loss:.4f}")
    print(f"Best model → {os.path.join(SAVE_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()