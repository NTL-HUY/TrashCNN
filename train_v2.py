import argparse
import os
import shutil
from pprint import pprint

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TrashDataset, collate_fn
from model import build_model
from tqdm.autonotebook import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--data_path", type=str, default=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default="tensorboard/TrashCNN")
    parser.add_argument("--save_path", type=str, default="trained_models")
    parser.add_argument("--resume_train_path", type=str, default="trained_models/last_model.pth")
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_dataset = TrashDataset(
        root=args.data_path,
        split='train',
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
        split='test',
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # TensorBoard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(log_dir=args.log_path)
    print(f"TensorBoard logs → {args.log_path}")
    print(f"  Run: tensorboard --logdir {args.log_path}\n")

    model = build_model(num_classes=train_dataset.get_num_classes()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_map = -1
    start_epoch = 0

    if args.resume_train_path is not None and os.path.exists(args.resume_train_path):
        print(f"Load checkpoint from {args.resume_train_path}")
        checkpoint = torch.load(args.resume_train_path, map_location=device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_map"]
        print(f"Tiep tuc tu epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_data_loader, colour="cyan")
        for iter, (images, targets) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_components = model(images, targets)
            losses = sum(loss for loss in loss_components.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
            train_progress_bar.set_description(
                f"Epoch {epoch + 1}/{args.epochs} | Total: {losses:.4f} | {loss_str}"
            )
            train_loss.append(losses.item())
            avg_loss = np.mean(train_loss)
            writer.add_scalar("Train/Loss", avg_loss, epoch * len(train_data_loader) + iter)

        scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)
        # EVAL
        model.eval()
        val_progress_bar = tqdm(val_data_loader, colour="yellow")
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")
        for iter, (images, targets) in enumerate(val_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                predictions = model(images)
                metric.update(predictions, targets)

        map_result = metric.compute()
        writer.add_scalar("Val/mAP",    map_result["map"].item(),    epoch)
        writer.add_scalar("Val/mAP_50", map_result["map_50"].item(), epoch)
        writer.add_scalar("Val/mAP_75", map_result["map_75"].item(), epoch)
        per_class = map_result["map_per_class"]
        if per_class.ndim > 0 and len(per_class) == len(train_dataset.categories):
            for i, ap in enumerate(per_class):
                writer.add_scalar(f"Val/AP_{train_dataset.categories[i]['name']}", ap.item(), epoch)
        else:
            print(f"   ⚠️ map_per_class chưa có dữ liệu (epoch {epoch + 1})")

        checkpoint = {
            "epoch":     epoch + 1,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map":  best_map
        }
        if map_result["map"] > best_map:
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pth"))
            best_map = map_result["map"]
        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pth"))

        avg_loss = np.mean(train_loss)
        print(f"\n Epoch {epoch + 1} summary:")
        print(f"   Avg total loss : {avg_loss:.4f}")
        print(f"   LR hiện tại    : {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch + 1} - mAP: {map_result['map']:.4f}")
    writer.close()


if __name__ == "__main__":
    args = get_args()
    train(args)