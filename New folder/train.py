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

Usage:
  python train.py --data_dir data/taco --num_epochs 50 --batch_size 4 --num_workers 4
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN for Waste Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    data = parser.add_argument_group("Data")
    data.add_argument("--data_dir", type=str, default="data/taco", help="Root data directory")
    data.add_argument("--ann_file", type=str, default=None,
                      help="Path to filtered annotations JSON (auto-set if not given)")
    data.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training split")
    data.add_argument("--val_ratio", type=float, default=0.1, help="Fraction for validation split")
    data.add_argument("--skip_download", action="store_true", help="Skip dataset download (use existing data)")

    # ── Training ──
    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=50, help="Total training epochs")
    train.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    train.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    train.add_argument("--seed", type=int, default=42, help="Random seed")
    train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    train.add_argument("--grad_clip", type=float, default=5.0, help="Max gradient norm for clipping")
    train.add_argument("--amp", action="store_true", help="Use automatic mixed precision (AMP)")
    train.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")

    # ── Optimizer ──
    optim = parser.add_argument_group("Optimizer")
    optim.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer type")
    optim.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    optim.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    optim.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    optim.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs for LR schedule")
    optim.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR at end of cosine schedule")

    # ── Model ──
    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--min_size", type=int, default=800, help="Minimum image size (resize shorter side)")
    model_args.add_argument("--max_size", type=int, default=1333, help="Maximum image size")
    model_args.add_argument("--rpn_nms_thresh", type=float, default=0.7, help="RPN NMS threshold")
    model_args.add_argument("--box_nms_thresh", type=float, default=0.5, help="Box NMS threshold")
    model_args.add_argument("--box_score_thresh", type=float, default=0.05, help="Box score threshold")
    model_args.add_argument("--box_detections", type=int, default=100, help="Max detections per image")

    # ── Early Stopping ──
    es = parser.add_argument_group("Early Stopping")
    es.add_argument("--early_stop_patience", type=int, default=15, help="Patience epochs for early stopping")
    es.add_argument("--early_stop_metric", type=str, default="mAP_50", help="Metric to monitor for early stopping")

    # ── Output ──
    out = parser.add_argument_group("Output")
    out.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    out.add_argument("--log_dir", type=str, default="runs/train", help="TensorBoard log directory")
    out.add_argument("--save_metric", type=str, default="mAP_50", help="Metric to track for best model")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Build Optimizer
# ─────────────────────────────────────────────
def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied only to non-bias/BN params."""
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
        return torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
        )
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


# ─────────────────────────────────────────────
# Train One Epoch
# ─────────────────────────────────────────────
def train_one_epoch(
        model,
        optimizer,
        loader,
        device,
        epoch: int,
        total_epochs: int,
        scaler: GradScaler,
        args,
        writer: SummaryWriter,
        global_step: int,
        logger: logging.Logger,
) -> tuple:
    model.train()
    loss_meters = LossMeters()
    optimizer.zero_grad()

    desc = f"Epoch [{epoch:3d}/{total_epochs}] Train"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    step_in_epoch = 0

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Filter out empty targets
        valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
        if not valid_pairs:
            continue
        images, targets = zip(*valid_pairs)
        images = list(images)
        targets = list(targets)

        with autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses = losses / args.grad_accum_steps

        scaler.scale(losses).backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            # Unscale before clip
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meters.update({k: v.detach() for k, v in loss_dict.items()})
        step_in_epoch += 1
        global_step += 1

        # TensorBoard step-level
        if global_step % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train_step/{k}", float(v), global_step)
            writer.add_scalar("train_step/total_loss", float(sum(loss_dict.values())), global_step)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train_step/lr", lr_now, global_step)

        # Update progress bar
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss_meters.meters['total'].avg:.4f}",
            "cls": f"{loss_meters.meters['loss_classifier'].avg:.4f}",
            "reg": f"{loss_meters.meters['loss_box_reg'].avg:.4f}",
            "rpn": f"{loss_meters.meters['loss_objectness'].avg:.4f}",
            "lr": f"{lr_now:.2e}",
        }, refresh=False)

    return loss_meters.averages(), global_step


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
        model,
        loader,
        device,
        epoch: int,
        total_epochs: int,
        logger: logging.Logger,
) -> dict:
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


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Logging ──
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, name="train")
    writer = SummaryWriter(log_dir=args.log_dir)
    logger.info(f"Device  : {device}")
    if device.type == "cuda":
        logger.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ──
    ann_file = args.ann_file
    if ann_file is None:
        if not args.skip_download:
            logger.info("Downloading / verifying TACO dataset...")
            ann_file = download_taco_dataset(data_dir=args.data_dir)
        else:
            ann_file = str(Path(args.data_dir) / "annotations_filtered.json")
            if not Path(ann_file).exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {ann_file}. "
                    "Run without --skip_download to download dataset."
                )

    logger.info(f"Annotations: {ann_file}")
    train_loader, val_loader, test_loader = build_dataloaders(
        ann_file=ann_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ── Model ──
    logger.info(
        'Building Faster R-CNN (ResNe"--ann_file",     type=str,   default=None,         help="Path to filtered annotations JSON (auto-set if not given)')
    data.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training split")
    data.add_argument("--val_ratio", type=float, default=0.1, help="Fraction for validation split")
    data.add_argument("--skip_download", action="store_true", help="Skip dataset download (use existing data)")

    # ── Training ──
    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=50, help="Total training epochs")
    train.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    train.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    train.add_argument("--seed", type=int, default=42, help="Random seed")
    train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    train.add_argument("--grad_clip", type=float, default=5.0, help="Max gradient norm for clipping")
    train.add_argument("--amp", action="store_true", help="Use automatic mixed precision (AMP)")
    train.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")

    # ── Optimizer ──
    optim = parser.add_argument_group("Optimizer")
    optim.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer type")
    optim.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    optim.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    optim.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    optim.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs for LR schedule")
    optim.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR at end of cosine schedule")

    # ── Model ──
    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--min_size", type=int, default=800, help="Minimum image size (resize shorter side)")
    model_args.add_argument("--max_size", type=int, default=1333, help="Maximum image size")
    model_args.add_argument("--rpn_nms_thresh", type=float, default=0.7, help="RPN NMS threshold")
    model_args.add_argument("--box_nms_thresh", type=float, default=0.5, help="Box NMS threshold")
    model_args.add_argument("--box_score_thresh", type=float, default=0.05, help="Box score threshold")
    model_args.add_argument("--box_detections", type=int, default=100, help="Max detections per image")

    # ── Early Stopping ──
    es = parser.add_argument_group("Early Stopping")
    es.add_argument("--early_stop_patience", type=int, default=15, help="Patience epochs for early stopping")
    es.add_argument("--early_stop_metric", type=str, default="mAP_50", help="Metric to monitor for early stopping")

    # ── Output ──
    out = parser.add_argument_group("Output")
    out.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    out.add_argument("--log_dir", type=str, default="runs/train", help="TensorBoard log directory")
    out.add_argument("--save_metric", type=str, default="mAP_50", help="Metric to track for best model")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Build Optimizer
# ─────────────────────────────────────────────
def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied only to non-bias/BN params."""
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
        return torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
        )
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


# ─────────────────────────────────────────────
# Train One Epoch
# ─────────────────────────────────────────────
def train_one_epoch(
        model,
        optimizer,
        loader,
        device,
        epoch: int,
        total_epochs: int,
        scaler: GradScaler,
        args,
        writer: SummaryWriter,
        global_step: int,
        logger: logging.Logger,
) -> tuple:
    model.train()
    loss_meters = LossMeters()
    optimizer.zero_grad()

    desc = f"Epoch [{epoch:3d}/{total_epochs}] Train"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    step_in_epoch = 0

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Filter out empty targets
        valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
        if not valid_pairs:
            continue
        images, targets = zip(*valid_pairs)
        images = list(images)
        targets = list(targets)

        with autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses = losses / args.grad_accum_steps

        scaler.scale(losses).backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            # Unscale before clip
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meters.update({k: v.detach() for k, v in loss_dict.items()})
        step_in_epoch += 1
        global_step += 1

        # TensorBoard step-level
        if global_step % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train_step/{k}", float(v), global_step)
            writer.add_scalar("train_step/total_loss", float(sum(loss_dict.values())), global_step)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train_step/lr", lr_now, global_step)

        # Update progress bar
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss_meters.meters['total'].avg:.4f}",
            "cls": f"{loss_meters.meters['loss_classifier'].avg:.4f}",
            "reg": f"{loss_meters.meters['loss_box_reg'].avg:.4f}",
            "rpn": f"{loss_meters.meters['loss_objectness'].avg:.4f}",
            "lr": f"{lr_now:.2e}",
        }, refresh=False)

    return loss_meters.averages(), global_step


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
        model,
        loader,
        device,
        epoch: int,
        total_epochs: int,
        logger: logging.Logger,
) -> dict:
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


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Logging ──
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=args.log_dir, name="train")
    writer = SummaryWriter(log_dir=args.log_dir)
    logger.info(f"Device  : {device}")
    if device.type == "cuda":
        logger.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ──
    ann_file = args.ann_file
    if ann_file is None:
        if not args.skip_download:
            logger.info("Downloading / verifying TACO dataset...")
            ann_file = download_taco_dataset(data_dir=args.data_dir)
        else:
            ann_file = str(Path(args.data_dir) / "annotations_filtered.json")
            if not Path(ann_file).exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {ann_file}. "
                    "Run without --skip_download to download dataset."
                )

    logger.info(f"Annotations: {ann_file}")
    train_loader, val_loader, test_loader = build_dataloaders(
        ann_file=ann_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ── Model ──
    logger.info("Building Faster R-CNN (ResNt-50 + FPN, from scratch)...")
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

    # ── Optimizer ──
    optimizer = build_optimizer(model, args)
    logger.info(f"Optimizer: {args.optimizer.upper()} | lr={args.lr} | wd={args.weight_decay}")

    # ── Scheduler ──
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        min_lr=args.min_lr,
    )

    # ── AMP Scaler ──
    scaler = GradScaler(enabled=args.amp)
    if args.amp:
        logger.info("Mixed Precision (AMP): ENABLED")

    # ── Checkpoint Manager ──
    ckpt_manager = CheckpointManager(save_dir=args.weights_dir, metric=args.save_metric)

    # ── Early Stopping ──
    early_stop = EarlyStopping(
        patience=args.early_stop_patience,
        metric=args.early_stop_metric,
    )

    # ── Resume ──
    start_epoch = 1
    global_step = 0
    best_metrics = {}
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        start_epoch, best_metrics = ckpt_manager.load(
            args.resume, model, optimizer, scheduler, scaler, device=device
        )
        start_epoch += 1
        logger.info(f"Resumed at epoch {start_epoch} | Previous metrics: {best_metrics}")

    # ── Add model graph to TensorBoard ──
    try:
        dummy_input = [torch.rand(3, 640, 640).to(device)]
        writer.add_graph(model, [dummy_input])
    except Exception:
        pass  # graph logging is optional

    # ── Log hyperparameters ──
    hparam_dict = {
        "lr": args.lr, "batch_size": args.batch_size, "optimizer": args.optimizer,
        "weight_decay": args.weight_decay, "num_epochs": args.num_epochs,
        "warmup_epochs": args.warmup_epochs, "grad_clip": args.grad_clip,
        "amp": args.amp,
    }
    writer.add_hparams(hparam_dict, {})

    # ── Training Loop ──
    logger.info("=" * 70)
    logger.info(f"  Starting Training: {args.num_epochs} epochs | Classes: {TARGET_CLASSES[1:]}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Epoch {epoch}/{args.num_epochs} | LR: {lr_now:.6f}")

        # ── Train ──
        train_losses, global_step = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            total_epochs=args.num_epochs,
            scaler=scaler,
            args=args,
            writer=writer,
            global_step=global_step,
            logger=logger,
        )

        scheduler.step()

        # ── TensorBoard: train epoch ──
        for k, v in train_losses.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)
        writer.add_scalar("train_epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        # ── Validation ──
        val_metrics = {}
        if epoch % args.val_every == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                total_epochs=args.num_epochs,
                logger=logger,
            )

            # TensorBoard: val metrics
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            # Scalars comparison chart
            writer.add_scalars("summary/loss", {
                "train_total": train_losses.get("total", 0),
            }, epoch)
            writer.add_scalars("summary/mAP", {
                "mAP": val_metrics.get("mAP", 0),
                "mAP_50": val_metrics.get("mAP_50", 0),
                "mAP_75": val_metrics.get("mAP_75", 0),
            }, epoch)

        # ── Save Checkpoint ──
        is_best = ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            scaler=scaler,
        )

        epoch_time = time.time() - epoch_start

        # ── Epoch Summary ──
        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Loss: {train_losses.get('total', 0):.4f} | "
            f"mAP: {val_metrics.get('mAP', 0):.4f} | "
            f"mAP@50: {val_metrics.get('mAP_50', 0):.4f} | "
            f"mAP@75: {val_metrics.get('mAP_75', 0):.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {format_time(epoch_time)}"
            + (" ← BEST" if is_best else "")
        )

        if val_metrics:
            per_class_str = " | ".join(
                f"{k}: {v:.3f}"
                for k, v in val_metrics.items()
                if k.startswith("AP_")
            )
            if per_class_str:
                logger.info(f"  Per-class: {per_class_str}")

        # ── Early Stopping ──
        if val_metrics and early_stop.step(val_metrics):
            logger.info(
                f"\n⚠️  Early stopping triggered at epoch {epoch}. "
                f"No improvement in {args.early_stop_metric} for {args.early_stop_patience} epochs."
            )
            break

    # ── Final Evaluation on Test Set ──
    logger.info("\n" + "=" * 70)
    logger.info("  Final evaluation on TEST set...")
    logger.info("=" * 70)

    best_path = Path(args.weights_dir) / "best_model.pth"
    if best_path.exists():
        ckpt_manager.load(str(best_path), model, device=device)
        logger.info(f"Loaded best model: {best_path}")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        epoch=args.num_epochs,
        total_epochs=args.num_epochs,
        logger=logger,
    )
    logger.info(f"TEST mAP     : {test_metrics.get('mAP', 0):.4f}")
    logger.info(f"TEST mAP@50  : {test_metrics.get('mAP_50', 0):.4f}")
    logger.info(f"TEST mAP@75  : {test_metrics.get('mAP_75', 0):.4f}")

    for k, v in test_metrics.items():
        if k.startswith("AP_"):
            logger.info(f"  {k}: {v:.4f}")
        writer.add_scalar(f"test/{k}", v, args.num_epochs)

    writer.close()
    logger.info(f"\n✅ Training complete. Weights saved in: {args.weights_dir}/")
    logger.info(f"   Best: {args.weights_dir}/best_model.pth")
    logger.info(f"   Last: {args.weights_dir}/last_model.pth")
    logger.info(f"   TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
