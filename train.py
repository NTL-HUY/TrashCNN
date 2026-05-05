"""
train.py - Training script for Faster R-CNN Waste Detector
Features:
  - Progress bar per epoch with loss, mAP, mAP@50, lr
  - TensorBoard logging
  - Best + Last model saving
  - Warmup + Cosine LR schedule
  - Mixed precision (AMP) for T4 GPU
  - Gradient clipping for stability
  - Early stopping
  - Gradient accumulation
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataloaders, download_taco_dataset, NUM_CLASSES, TARGET_CLASSES
from model import build_faster_rcnn, model_summary
from utils import (
    setup_logger,
    MetricLogger,
    LossMeters,
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    clip_gradients,
    format_time,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN for Waste Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--data_dir", type=str, default="data/taco", help="Root data directory")
    data.add_argument("--ann_file", type=str, default=None, help="Path to filtered annotations JSON")
    data.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training split")
    data.add_argument("--val_ratio", type=float, default=0.1, help="Fraction for validation split")
    data.add_argument("--skip_download", action="store_true", help="Skip dataset download")

    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=50, help="Total training epochs")
    train.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    train.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    train.add_argument("--seed", type=int, default=42, help="Random seed")
    train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    train.add_argument("--grad_clip", type=float, default=5.0, help="Max gradient norm for clipping")
    train.add_argument("--amp", action="store_true", help="Use automatic mixed precision (AMP)")
    train.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")

    optim = parser.add_argument_group("Optimizer")
    # Thay đổi mặc định optimizer và lr cho an toàn khi train from scratch
    optim.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer type")
    optim.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    optim.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    optim.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    optim.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs for LR schedule")
    optim.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR at end of cosine schedule")

    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--min_size", type=int, default=800, help="Minimum image size")
    model_args.add_argument("--max_size", type=int, default=1333, help="Maximum image size")
    model_args.add_argument("--rpn_nms_thresh", type=float, default=0.7, help="RPN NMS threshold")
    model_args.add_argument("--box_nms_thresh", type=float, default=0.5, help="Box NMS threshold")
    model_args.add_argument("--box_score_thresh", type=float, default=0.05, help="Box score threshold")
    model_args.add_argument("--box_detections", type=int, default=100, help="Max detections per image")

    es = parser.add_argument_group("Early Stopping")
    es.add_argument("--early_stop_patience", type=int, default=15, help="Patience epochs")
    es.add_argument("--early_stop_metric", type=str, default="mAP_50", help="Metric to monitor")

    out = parser.add_argument_group("Output")
    out.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    out.add_argument("--log_dir", type=str, default="runs/train", help="TensorBoard log directory")
    out.add_argument("--save_metric", type=str, default="mAP_50", help="Metric to track for best model")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if args.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs, scaler, args, writer, global_step, logger):
    model.train()
    loss_meters = LossMeters()
    optimizer.zero_grad()

    desc = f"Epoch [{epoch:3d}/{total_epochs}] Train"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    step_in_epoch = 0

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
        if not valid_pairs:
            continue
        images, targets = zip(*valid_pairs)
        images, targets = list(images), list(targets)

        with torch.autocast(device_type=device.type, enabled=args.amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses = losses / args.grad_accum_steps

        scaler.scale(losses).backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meters.update({k: v.detach() for k, v in loss_dict.items()})
        step_in_epoch += 1
        global_step += 1

        if global_step % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train_step/{k}", v.detach().item(), global_step)
            writer.add_scalar("train_step/total_loss", sum(loss_dict.values()).detach().item(), global_step)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train_step/lr", lr_now, global_step)

        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss_meters.meters['total'].avg:.4f}",
            "cls": f"{loss_meters.meters['loss_classifier'].avg:.4f}",
            "reg": f"{loss_meters.meters['loss_box_reg'].avg:.4f}",
            "lr": f"{lr_now:.2e}",
        }, refresh=False)

    return loss_meters.averages(), global_step


@torch.no_grad()
def evaluate(model, loader, device, epoch, total_epochs, logger) -> dict:
    model.eval()
    metric_logger = MetricLogger(num_classes=NUM_CLASSES, class_names=TARGET_CLASSES)

    desc = f"Epoch [{epoch:3d}/{total_epochs}]   Val"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets_cpu = [{k: v for k, v in t.items()} for t in targets]

        outputs = model(images)
        outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]

        metric_logger.update(outputs_cpu, targets_cpu)
        pbar.set_postfix({"computing": "mAP..."}, refresh=False)

    metrics = metric_logger.compute()
    pbar.set_postfix({
        "mAP": f"{metrics['mAP']:.4f}",
        "mAP@50": f"{metrics['mAP_50']:.4f}",
    }, refresh=True)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, name="train")
    writer = SummaryWriter(log_dir=args.log_dir)
    logger.info(f"Device  : {device}")
    if device.type == "cuda":
        logger.info(f"GPU     : {torch.cuda.get_device_name(0)}")

    ann_file = args.ann_file
    if ann_file is None:
        if not args.skip_download:
            logger.info("Downloading / verifying TACO dataset...")
            ann_file = download_taco_dataset(data_dir=args.data_dir)
        else:
            ann_file = str(Path(args.data_dir) / "annotations_filtered.json")

    logger.info(f"Annotations: {ann_file}")
    train_loader, val_loader, test_loader = build_dataloaders(
        ann_file=ann_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info("Building Faster R-CNN (ResNet-50 + FPN, from scratch with GroupNorm)...")
    model = build_faster_rcnn(
        num_classes=NUM_CLASSES,
        min_size=args.min_size,
        max_size=args.max_size,
        rpn_nms_thresh=args.rpn_nms_thresh,
        box_nms_thresh=args.box_nms_thresh,
        box_score_thresh=args.box_score_thresh,
        box_detections_per_img=args.box_detections,
    ).to(device)
    model_summary(model)

    optimizer = build_optimizer(model, args)
    logger.info(f"Optimizer: {args.optimizer.upper()} | lr={args.lr} | wd={args.weight_decay}")

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        min_lr=args.min_lr,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ckpt_manager = CheckpointManager(save_dir=args.weights_dir, metric=args.save_metric)
    early_stop = EarlyStopping(patience=args.early_stop_patience, metric=args.early_stop_metric)

    start_epoch = 1
    global_step = 0
    if args.resume:
        start_epoch, best_metrics = ckpt_manager.load(args.resume, model, optimizer, scheduler, scaler, device=device)
        start_epoch += 1

    hparam_dict = {
        "lr": args.lr, "batch_size": args.batch_size, "optimizer": args.optimizer,
        "weight_decay": args.weight_decay, "num_epochs": args.num_epochs,
    }
    writer.add_hparams(hparam_dict, {})

    logger.info("=" * 70)
    logger.info(f"  Starting Training: {args.num_epochs} epochs")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        train_losses, global_step = train_one_epoch(
            model=model, optimizer=optimizer, loader=train_loader, device=device,
            epoch=epoch, total_epochs=args.num_epochs, scaler=scaler, args=args,
            writer=writer, global_step=global_step, logger=logger,
        )

        scheduler.step()

        for k, v in train_losses.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

        val_metrics = {}
        if epoch % args.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, epoch, args.num_epochs, logger)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        is_best = ckpt_manager.save(model, optimizer, scheduler, epoch, val_metrics, scaler)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | Loss: {train_losses.get('total', 0):.4f} | "
            f"mAP: {val_metrics.get('mAP', 0):.4f} | LR: {lr_now:.2e} | Time: {format_time(epoch_time)}"
            + (" ← BEST" if is_best else "")
        )

        if val_metrics and early_stop.step(val_metrics):
            logger.info(f"\n⚠️  Early stopping triggered at epoch {epoch}.")
            break

    logger.info("  Final evaluation on TEST set...")
    best_path = Path(args.weights_dir) / "best_model.pth"
    if best_path.exists():
        ckpt_manager.load(str(best_path), model, device=device)

    test_metrics = evaluate(model, test_loader, device, args.num_epochs, args.num_epochs, logger)
    logger.info(f"TEST mAP@50  : {test_metrics.get('mAP_50', 0):.4f}")
    writer.close()

if __name__ == "__main__":
    main()