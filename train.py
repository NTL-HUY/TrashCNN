import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import TrashDataset, DetectionTransforms, collate_fn
from model import build_model


# ─────────────────────────── Args ────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="TrashCNN Training")
    p.add_argument("--data_path", type=str, default=r".\TACO dataset.v1i.coco")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Số epoch warmup LR từ lr/10 → lr")
    p.add_argument("--log_path", type=str, default="tensorboard/TrashCNN")
    p.add_argument("--save_path", type=str, default="trained_models")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint để tiếp tục train")
    p.add_argument("--early_stop", type=int, default=10,
                   help="Dừng nếu mAP không cải thiện sau N epoch")
    return p.parse_args()


# ─────────────────────────── LR Warmup ───────────────────────────────────

def get_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Linear warmup: epoch 0 → factor=0.1, epoch warmup_epochs → factor=1.0
    Sau warmup: CosineAnnealingLR tự điều chỉnh.
    Tại sao warmup?
        → Lúc đầu weights random, gradient lớn → LR cao → loss spike
        → Warmup tăng LR dần để model ổn định trước
    """
    if epoch >= warmup_epochs:
        return 1.0
    return 0.1 + 0.9 * (epoch / warmup_epochs)


# ─────────────────────────── Train one epoch ─────────────────────────────

def train_one_epoch(
        model, loader, optimizer, device, writer, global_step, epoch
):
    model.train()
    train_loss = []

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch + 1}", colour="cyan", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ── Forward ───────────────────────────────────────────────────
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # ── Backward ──────────────────────────────────────────────────
        optimizer.zero_grad()
        losses.backward()

        # ── FIX: Gradient clipping → tránh exploding gradient / loss spike
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        loss_val = losses.item()
        train_loss.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

        # ── TensorBoard: log từng thành phần loss ─────────────────────
        for k, v in loss_dict.items():
            writer.add_scalar(f"Loss/train_{k}", v.item(), global_step)
        writer.add_scalar("Loss/train_total", loss_val, global_step)
        global_step += 1

    avg_loss = float(np.mean(train_loss))
    return avg_loss, global_step


# ─────────────────────────── Evaluate ────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, writer, epoch):
    """
    Tính mAP trên validation set.
    Dùng torchmetrics.MeanAveragePrecision (COCO protocol).
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch + 1}", colour="yellow", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)

        # torchmetrics cần list of dict với keys: boxes, scores, labels
        metric.update(preds, targets)

    result = metric.compute()
    map_val = result["map"].item()
    map50_val = result["map_50"].item()

    writer.add_scalar("Val/mAP", map_val, epoch)
    writer.add_scalar("Val/mAP_50", map50_val, epoch)

    return map_val, map50_val


# ─────────────────────────── Main ────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────
    train_dataset = TrashDataset(
        root=args.data_path,
        split="train",
        transforms=DetectionTransforms(is_train=True),  # augment + normalize
    )
    val_dataset = TrashDataset(
        root=args.data_path,
        split="valid",
        transforms=DetectionTransforms(is_train=False),  # chỉ normalize
    )

    num_classes = len(train_dataset.categories) + 1  # +1 cho background
    print(f"Số class: {num_classes - 1} + background = {num_classes}")
    print("Classes:", [c['name'] for c in train_dataset.categories])

    # ── FIX: WeightedRandomSampler → giải quyết class imbalance ──────
    # Ảnh có class hiếm (other, glass) sẽ được sample nhiều hơn
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # thay shuffle=True
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ── FIX: SGD + momentum (ổn hơn Adam cho detection) ──────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * 0.1,  # bắt đầu thấp, warmup sẽ tăng lên
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # ── FIX: Warmup + CosineAnnealing ────────────────────────────────
    # Warmup 5 epoch đầu → CosineAnnealingLR phần còn lại
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.001,
    )

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=args.log_path)
    print(f"\nTensorBoard: tensorboard --logdir {args.log_path}\n")

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_map = -1.0
    no_improve = 0  # đếm epoch không cải thiện (early stopping)
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_map = ckpt.get("best_map", -1.0)
        global_step = ckpt.get("global_step", 0)
        print(f"  → Resumed from epoch {start_epoch}, best_mAP={best_map:.4f}\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):

        # ── Warmup LR ─────────────────────────────────────────────────
        warmup_factor = get_warmup_factor(epoch, args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * warmup_factor
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        # ── Train ─────────────────────────────────────────────────────
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, writer, global_step, epoch
        )
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)

        # ── Cosine LR step (sau warmup) ───────────────────────────────
        if epoch >= args.warmup_epochs:
            cosine_scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        map_val, map50_val = evaluate(
            model, val_loader, device, writer, epoch
        )

        print(
            f"Epoch [{epoch + 1:3d}/{args.epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"mAP={map_val:.4f}  mAP@50={map50_val:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ── Save checkpoint ───────────────────────────────────────────
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
            "global_step": global_step,
        }
        last_path = os.path.join(args.save_path, "last_model.pth")
        torch.save(ckpt, last_path)

        if map_val > best_map:
            best_map = map_val
            no_improve = 0
            best_path = os.path.join(args.save_path, "best_model.pth")
            torch.save(ckpt, best_path)
            print(f"  ✔ Best model saved  (mAP={best_map:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"\nEarly stopping: mAP không cải thiện sau {args.early_stop} epoch.")
                break

    writer.close()
    print(f"\nDone! Best mAP: {best_map:.4f}")
    print(f"Best model → {os.path.join(args.save_path, 'best_model.pth')}")


if __name__ == "__main__":
    main()
