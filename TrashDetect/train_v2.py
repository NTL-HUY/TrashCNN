import argparse
import os
import random
import shutil

import albumentations as A
import time
from albumentations.pytorch import ToTensorV2
from torchinfo import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TrashDataset, collate_fn
from model import build_model
from tqdm.autonotebook import tqdm
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--data_path", type=str, default=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--save_path", type=str, default="trained_models")
    parser.add_argument("--resume_train_path", type=str, default="trained_models/last_model.pth")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    args = parser.parse_args()
    return args


def get_train_transform(args):
    return A.Compose([
        A.Resize(args.image_size, args.image_size),

        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.HueSaturationValue(p=0.3),

        A.GaussianBlur(p=0.2),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )


def get_val_transform(args):
    return A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )

def log_model_info(model, args, writer, device):
    """Log số params và model size bằng torchinfo"""
    # Dummy input cho Faster RCNN
    dummy = [torch.zeros(3, args.image_size, args.image_size).to(device)]
    try:
        model_info = summary(model, input_data=dummy, verbose=0)
        total_params = model_info.total_params
        trainable_params = model_info.trainable_params
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size (MB)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 ** 2
    model_size_mb = param_size_mb + buffer_size_mb

    print(f"\n{'=' * 60}")
    print(f"Model Info")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Model size      : {model_size_mb:.2f} MB")
    print(f"  GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'=' * 60}\n")

    # Ghi vào TensorBoard (dạng text để dễ xem)
    writer.add_text("Model/Info", f"Total params: {total_params:,} | Trainable: {trainable_params:,} | Size: {model_size_mb:.2f} MB")
    writer.add_text("Config", str(vars(args)))

    return total_params, model_size_mb


def measure_fps(model, device, image_size, num_runs=50):
    model.eval()
    dummy = [torch.zeros(3, image_size, image_size).to(device)]
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_e = torch.cuda.Event(enable_timing=True)
        end_e   = torch.cuda.Event(enable_timing=True)
        start_e.record()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy)
        end_e.record()
        torch.cuda.synchronize()
        elapsed_ms = start_e.elapsed_time(end_e)
    else:
        t0 = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy)
        elapsed_ms = (time.time() - t0) * 1000

    ms_per_image = elapsed_ms / num_runs
    return 1000 / ms_per_image, ms_per_image


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_dataset = TrashDataset(
        root=args.data_path,
        split='train',
        transforms=get_train_transform(args)
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_dataset = TrashDataset(
        root=args.data_path,
        split='valid',
        transforms=get_val_transform(args)
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    # tensor board
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(log_dir=args.log_path)
    print(f"TensorBoard logs → {args.log_path}")
    print(f"  Run: tensorboard --logdir {args.log_path}\n")
    model = build_model(num_classes=train_dataset.get_num_classes()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_map = -1

    # Log model info
    total_params, model_size_mb = log_model_info(model, args, writer, device)

    start_epoch = 0
    if args.resume_train_path is not None and os.path.exists(args.resume_train_path):
        print(f"[INFO] Load checkpoint from {args.resume_train_path}")
        checkpoint = torch.load(args.resume_train_path, map_location=device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_map"]
        print(f"[INFO] Continue from epoch {start_epoch}")

    class_names = train_dataset.get_class_names()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = []
        epoch_start = time.time()
        train_progress_bar = tqdm(train_data_loader, colour="cyan")
        for iter, (images, targets) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_components = model(images, targets)
            losses = sum(loss for loss in loss_components.values())

            optimizer.zero_grad()  # tat luu tru value cua gradient trong buffer ?
            losses.backward()  # tinh dao ham (gradient) (quy tac chain rule)
            optimizer.step()  # w = w-lr*grad , lay gia tri cua gradient de update weight

            train_loss.append(losses.item())

            # print
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
            train_progress_bar.set_description(
                f"Epoch {epoch + 1}/{args.epochs} | Total: {losses:.4f} | {loss_str}"
            )

            step = epoch * len(train_data_loader) + iter
            writer.add_scalar("Train/Loss_total", losses.item(), step)
            # Log từng thành phần loss
            for k, v in loss_components.items():
                writer.add_scalar(f"Train/Loss_{k}", v.item(), step)

        epoch_time = time.time() - epoch_start
        scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("Train/Epoch_time_sec", epoch_time, epoch)

        # EVAL
        # tinh MAP
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval", class_metrics=True)
        val_bar = tqdm(val_data_loader, colour="yellow")

        for images, targets in val_bar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():  # khong tinh backward()
                predictions = model(images)
                metric.update(predictions, targets)

        result = metric.compute()
        # ---- Log metrics ----

        map_val = float(result["map"])
        map50 = float(result["map_50"])
        map75 = float(result["map_75"])
        prec = float(result.get("map_per_class", torch.tensor(0)).mean()) if "map_per_class" in result else 0
        recall = float(result.get("mar_100", torch.tensor(0)))

        writer.add_scalar("Val/mAP",    map_val, epoch)
        writer.add_scalar("Val/mAP_50", map50,   epoch)
        writer.add_scalar("Val/mAP_75", map75,   epoch)
        writer.add_scalar("Val/Recall_AR100", recall, epoch)
        writer.add_scalar("Val/Precision", prec, epoch)

        if "map_per_class" in result:
            per_class_ap = result["map_per_class"]
            for i, ap in enumerate(per_class_ap):
                name = class_names[i] if class_names else f"class_{i}"
                writer.add_scalar(f"Val/AP_class/{name}", float(ap), epoch)

        # FPS (đo 1 lần mỗi 5 epoch để không chậm quá)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            fps, ms = measure_fps(model, device, args.image_size)
            writer.add_scalar("Perf/FPS", fps, epoch)
            writer.add_scalar("Perf/Latency_ms", ms, epoch)
            print(f"  FPS: {fps:.1f} | Latency: {ms:.1f} ms")

        # ---- Save ----
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": float(best_map),   # BUG FIX: ép về float
            "config": vars(args),
            "model_params": total_params,
            "model_size_mb": model_size_mb,
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pth"))

        if map_val > best_map:
            best_map = map_val
            checkpoint["best_map"] = best_map
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pth"))

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs} | Time: {epoch_time:.0f}s")
        print(f"  Loss     : {np.mean(train_loss):.4f}")
        print(f"  mAP      : {map_val:.4f} | mAP50: {map50:.4f} | mAP75: {map75:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  Best mAP : {best_map:.4f}")
        print(f"  LR       : {scheduler.get_last_lr()[0]:.6f}")

    writer.close()


if __name__ == "__main__":
    args = get_args()
    train(args)

