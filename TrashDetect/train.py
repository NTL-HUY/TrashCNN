import argparse
import os
import shutil
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import TrashDataset, collate_fn
from model import build_model


def get_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on TACO dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_head", type=float, default=0.0005,
                        help="LR cho detection head (RPN + ROI)")
    parser.add_argument("--lr_backbone", type=float, default=0.00005,
                        help="LR cho backbone layer3/layer4/FPN")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--data_path", type=str, default="TrashDetect/TACO dataset.v1i.coco")
    parser.add_argument("--backbone_weights", type=str, default="TrashClassify/trained_models/best_model.pth")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--log_path", type=str, default="TrashDetect/tensorboard/TrashCNN")
    parser.add_argument("--save_path", type=str, default="TrashDetect/trained_models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint để resume training")
    return parser.parse_args()


def get_transforms(image_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1.0,
            min_visibility=0.1,
        ))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1.0,
            min_visibility=0.1,
        ))


def build_optimizer(model, args):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params,     "lr": args.lr_head,     "name": "head"},
        {"params": backbone_params, "lr": args.lr_backbone, "name": "backbone"},
    ]

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    print(f"Optimizer — Head LR: {args.lr_head} | Backbone LR: {args.lr_backbone}")
    print(f"  Head params    : {sum(p.numel() for p in head_params):,}")
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")

    return optimizer


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    train_loss = []
    nan_count = 0

    pbar = tqdm(data_loader, colour="cyan", desc=f"Epoch {epoch + 1}")
    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Kiểm tra NaN loss trước khi backward
        if torch.isnan(losses) or torch.isinf(losses):
            nan_count += 1
            print(f"\n  [WARNING] NaN/Inf loss tại batch {i} (epoch {epoch + 1}). Bỏ qua batch này.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        train_loss.append(losses.item())
        avg = np.mean(train_loss)
        pbar.set_postfix(loss=f"{avg:.4f}", nan=nan_count)
        writer.add_scalar("Train/Batch_Loss", losses.item(), epoch * len(data_loader) + i)

    if nan_count > 0:
        print(f"  [WARNING] Epoch {epoch + 1}: {nan_count} batch bị NaN loss.")

    return np.mean(train_loss) if train_loss else float('nan')


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    for images, targets in tqdm(data_loader, colour="yellow", desc="Validation"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)
    return metric.compute()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_ds = TrashDataset(
        args.data_path, split='train',
        transforms=get_transforms(args.image_size, is_train=True),
        image_size=args.image_size
    )
    val_ds = TrashDataset(
        args.data_path, split='test',
        transforms=get_transforms(args.image_size, is_train=False),
        image_size=args.image_size
    )
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    model = build_model(
        num_classes=train_ds.get_num_classes(),
        my_weights_path=args.backbone_weights
    )
    model.to(device)

    optimizer = build_optimizer(model, args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    best_map = -1.0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint.get("best_map", -1.0)
        print(f"  Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")

    # TensorBoard
    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    writer = SummaryWriter(log_dir=args.log_path)
    os.makedirs(args.save_path, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer)
        scheduler.step()

        metrics = validate(model, val_loader, device)
        mAP = metrics['map'].item()
        mAP_50 = metrics.get('map_50', torch.tensor(0.0)).item()

        lr_head = optimizer.param_groups[0]['lr']
        lr_bb   = optimizer.param_groups[1]['lr']

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Loss      : {avg_loss:.4f}")
        print(f"  mAP       : {mAP:.4f}  |  mAP@50: {mAP_50:.4f}")
        print(f"  LR head   : {lr_head:.6f}  |  LR backbone: {lr_bb:.6f}")

        writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
        writer.add_scalar("Val/mAP", mAP, epoch)
        writer.add_scalar("Val/mAP_50", mAP_50, epoch)
        writer.add_scalar("LR/head", lr_head, epoch)
        writer.add_scalar("LR/backbone", lr_bb, epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pth"))

        if mAP > best_map:
            best_map = mAP
            checkpoint["best_map"] = best_map
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pth"))
            print(f"  *** Best model saved! mAP: {best_map:.4f} ***")

    writer.close()
    print(f"\nTraining complete. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()