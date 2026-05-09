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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--data_path", type=str, default="TrashDetect/TACO dataset.v1i.coco", help="Path to COCO dataset")
    parser.add_argument("--backbone_weights", type=str, default="TrashClassify/trained_models/best_model.pth", help="Path to resnet50 backbone weights")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--log_path", type=str, default="TrashDetect/tensorboard/TrashCNN")
    parser.add_argument("--save_path", type=str, default="TrashDetect/trained_models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def get_transforms(image_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, args):
    model.train()
    train_loss = []
    pbar = tqdm(data_loader, colour="cyan", desc=f"Epoch {epoch + 1}")

    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Logging
        train_loss.append(losses.item())
        curr_loss = np.mean(train_loss)
        pbar.set_postfix(loss=f"{curr_loss:.4f}")

        writer.add_scalar("Train/Batch_Loss", losses.item(), epoch * len(data_loader) + i)

    return curr_loss


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    pbar = tqdm(data_loader, colour="yellow", desc="Validation")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        # Chuyển target sang CPU cho torchmetrics nếu cần, hoặc để nguyên tùy version
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        metric.update(outputs, targets)

    return metric.compute()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Datasets & Dataloaders
    train_ds = TrashDataset(args.data_path, split='train', transforms=get_transforms(args.image_size, True))
    val_ds = TrashDataset(args.data_path, split='test', transforms=get_transforms(args.image_size, False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # 2. Model, Optimizer, Scheduler
    model = build_model(num_classes=train_ds.get_num_classes(), my_weights_path=args.backbone_weights)
    model.to(device)

    # Bỏ qua backbone đã bị đóng băng
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 3. Resume training
    start_epoch = 0
    best_map = -1.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint.get("best_map", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    # 4. Logger
    if os.path.exists(args.log_path): shutil.rmtree(args.log_path)
    writer = SummaryWriter(log_dir=args.log_path)
    os.makedirs(args.save_path, exist_ok=True)

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, args)
        scheduler.step()

        metrics = validate(model, val_loader, device)
        mAP = metrics['map'].item()

        # Log metrics
        writer.add_scalar("Train/Avg_Loss", avg_loss, epoch)
        writer.add_scalar("Val/mAP", mAP, epoch)
        writer.add_scalar("Val/mAP_50", metrics['map_50'].item(), epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch + 1} Results: Loss: {avg_loss:.4f} | mAP: {mAP:.4f} | Best mAP: {max(best_map, mAP):.4f}")

        # Save Checkpoints
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": max(best_map, mAP)
        }

        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pth"))
        if mAP > best_map:
            best_map = mAP
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pth"))
            print("--- Best model saved! ---")

        torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    main()