import argparse
import os
import shutil
import time

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TrashDataset, collate_fn
from model import build_model
from tqdm.autonotebook import tqdm


# Args
def get_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN from scratch")
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--epochs",             type=int,   default=50)
    parser.add_argument("--lr",                 type=float, default=0.005)
    parser.add_argument("--momentum",           type=float, default=0.9)
    parser.add_argument("--weight_decay",       type=float, default=5e-4)
    parser.add_argument("--data_path",          type=str,   default=r"TACO dataset.v1i.coco")
    parser.add_argument("--log_path",           type=str,   default="tensorboard/TrashCNN")
    parser.add_argument("--save_path",          type=str,   default="trained_models")
    parser.add_argument("--resume_train_path",  type=str,   default=None,
                        help="Path to checkpoint .pth to resume (leave None to train from scratch)")
    # ── Scheduler ─────────────────────────────────────────────────────────
    parser.add_argument("--lr_step",            type=int,   default=5)
    parser.add_argument("--lr_gamma",           type=float, default=0.5)
    # ── Early stopping ────────────────────────────────────────────────────
    parser.add_argument("--early_stop_patience", type=int,  default=10,
                        help="Stop if mAP doesn't improve for this many epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum improvement considered as a gain")
    _args = parser.parse_args()
    return _args


LINE = "=" * 72

def print_epoch_header(epoch, total, lr):
    print(f"\n{LINE}")
    print(f"  EPOCH {epoch}/{total}   |   LR = {lr:.6f}")
    print(LINE)

def print_loss_summary(loss_dict, avg_total):
    print(f"\n  ▶ Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"      {k:<20s}: {v:.4f}")
    print(f"      {'Total (avg)':<20s}: {avg_total:.4f}")

def print_metric_summary(map_result, cat_names):
    print(f"\n  ▶ Validation metrics:")
    print(f"      {'mAP  (IoU 0.50:0.95)':<30s}: {map_result['map'].item():.4f}")
    print(f"      {'mAP50 (IoU 0.50)':<30s}: {map_result['map_50'].item():.4f}")
    print(f"      {'mAP75 (IoU 0.75)':<30s}: {map_result['map_75'].item():.4f}")
    per_class = map_result.get("map_per_class", None)
    if per_class is not None and per_class.ndim > 0 and len(per_class) == len(cat_names):
        print(f"\n  ▶ Per-class AP (IoU 0.50:0.95):")
        for name, ap in zip(cat_names, per_class):
            bar = "█" * int(ap.item() * 20)
            print(f"      {name:<18s}: {ap.item():.4f}  {bar}")
    else:
        print("      (per-class AP not available yet)")


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, metric):
        if metric - self.best > self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"\n  ⚠  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  ✖  Early stopping triggered (best mAP = {self.best:.4f})")


# Training loop
def train(_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{LINE}")
    print(f"  Device : {device}")
    print(f"  Epochs : {_args.epochs}")
    print(f"  LR     : {_args.lr}   |  Batch : {_args.batch_size}")
    print(f"  Early stop patience : {_args.early_stop_patience}")
    print(LINE)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = TrashDataset(root=_args.data_path, split="train")
    val_dataset   = TrashDataset(root=_args.data_path, split="test")
    cat_names     = [c["name"] for c in train_dataset.categories]

    train_loader = DataLoader(
        train_dataset, batch_size=_args.batch_size,
        num_workers=_args.num_workers, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=_args.batch_size,
        num_workers=_args.num_workers, collate_fn=collate_fn
    )

    print(f"\n  Train samples : {len(train_dataset)}")
    print(f"  Val   samples : {len(val_dataset)}")
    print(f"  Classes       : {cat_names}")
    print(f"  Num classes   : {train_dataset.get_num_classes()} (incl. background)")

    # ── TensorBoard ───────────────────────────────────────────────────────
    if os.path.isdir(_args.log_path):
        shutil.rmtree(_args.log_path)
    os.makedirs(_args.log_path)
    os.makedirs(_args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=_args.log_path)
    print(f"\n  TensorBoard  →  tensorboard --logdir {_args.log_path}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=train_dataset.get_num_classes()).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {total_params:,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=_args.lr,
        momentum=_args.momentum,
        weight_decay=_args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=_args.lr_step, gamma=_args.lr_gamma
    )

    best_map    = -1.0
    start_epoch = 0
    early_stop  = EarlyStopping(patience=_args.early_stop_patience,
                                min_delta=_args.early_stop_min_delta)

    # ── Resume ────────────────────────────────────────────────────────────
    if _args.resume_train_path and os.path.exists(_args.resume_train_path):
        print(f"\n  Resuming from: {_args.resume_train_path}")
        ckpt = torch.load(_args.resume_train_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_map    = ckpt.get("best_map", -1.0)
        early_stop.best = best_map
        print(f"  Resumed at epoch {start_epoch}, best mAP = {best_map:.4f}")

    # ── Main loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, _args.epochs):
        ep = epoch + 1
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else _args.lr
        print_epoch_header(ep, _args.epochs, current_lr)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        loss_accum = {"loss_rpn_cls": 0., "loss_rpn_reg": 0.,
                      "loss_roi_cls": 0., "loss_roi_reg": 0.}
        total_loss_accum = 0.
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"  Train", leave=False, colour="cyan")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in loss_dict.items():
                loss_accum[k] += v.item()
            total_loss_accum += total_loss.item()
            n_batches += 1

            # live tqdm description
            pbar.set_postfix({k.replace("loss_", ""): f"{v.item():.3f}"
                               for k, v in loss_dict.items()})

        scheduler.step()

        # average losses
        avg_losses = {k: v / n_batches for k, v in loss_accum.items()}
        avg_total  = total_loss_accum / n_batches
        elapsed    = time.time() - t0

        print_loss_summary(avg_losses, avg_total)
        print(f"\n      Epoch time : {elapsed:.1f}s")

        # ── TensorBoard: losses ────────────────────────────────────────────
        writer.add_scalar("Train/Loss_Total",   avg_total, ep)
        for k, v in avg_losses.items():
            writer.add_scalar(f"Train/{k}", v, ep)
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], ep)

        # ── Eval ──────────────────────────────────────────────────────────
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")
        pbar_val = tqdm(val_loader, desc="  Val  ", leave=False, colour="yellow")
        for images, targets in pbar_val:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                preds = model(images)
            metric.update(preds, targets)

        map_result = metric.compute()
        print_metric_summary(map_result, cat_names)

        # ── TensorBoard: metrics ───────────────────────────────────────────
        writer.add_scalar("Val/mAP",    map_result["map"].item(),    ep)
        writer.add_scalar("Val/mAP_50", map_result["map_50"].item(), ep)
        writer.add_scalar("Val/mAP_75", map_result["map_75"].item(), ep)
        per_class = map_result.get("map_per_class", None)
        if per_class is not None and per_class.ndim > 0 and len(per_class) == len(cat_names):
            for name, ap in zip(cat_names, per_class):
                writer.add_scalar(f"Val/AP_{name}", ap.item(), ep)

        current_map = map_result["map"].item()

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt = {
            "epoch":     ep,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map":  best_map,
        }
        torch.save(ckpt, os.path.join(_args.save_path, "last_model.pth"))

        if current_map > best_map:
            best_map = current_map
            ckpt["best_map"] = best_map
            torch.save(ckpt, os.path.join(_args.save_path, "best_model.pth"))
            print(f"\n  ✔  New best mAP = {best_map:.4f}  →  saved best_model.pth")

        # epoch footer
        print(f"\n{LINE}")
        print(f"  Epoch {ep} done  |  mAP={current_map:.4f}  best={best_map:.4f}")
        print(LINE)

        # ── Early stopping ─────────────────────────────────────────────────
        early_stop.step(current_map)
        if early_stop.should_stop:
            print(f"\n  Training stopped early at epoch {ep}.")
            break

    writer.close()
    print(f"\n{LINE}")
    print(f"  Training complete.  Best mAP = {best_map:.4f}")
    print(f"  Models saved to    : {_args.save_path}")
    print(f"  TensorBoard logs   : {_args.log_path}")
    print(LINE)


if __name__ == "__main__":
    _args = get_args()
    train(_args)