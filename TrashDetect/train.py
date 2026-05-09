import argparse
import os
import shutil
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from dataset import TrashDataset, collate_fn
from model import build_model


def get_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on TACO dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--data_path", type=str, default="TrashDetect/TACO dataset.v1i.coco")
    parser.add_argument("--backbone_weights", type=str, default="TrashClassify/trained_models/best_model.pth")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--log_path", type=str, default="TrashDetect/tensorboard/TrashCNN")
    parser.add_argument("--save_path", type=str, default="TrashDetect/trained_models")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def get_transforms(image_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


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
        train_loss.append(losses.item())
        pbar.set_postfix(loss=f"{np.mean(train_loss):.4f}")
        writer.add_scalar("Train/Batch_Loss", losses.item(), epoch * len(data_loader) + i)
    return np.mean(train_loss)


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

    # Load dataset với torchvision transforms
    train_ds = TrashDataset(args.data_path, split='train',
                            transforms=get_transforms(args.image_size, True),
                            image_size=args.image_size)  # Thêm image_size vào đây
    val_ds = TrashDataset(args.data_path, split='test',
                          transforms=get_transforms(args.image_size, False),
                          image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=train_ds.get_num_classes(), my_weights_path=args.backbone_weights)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if os.path.exists(args.log_path): shutil.rmtree(args.log_path)
    writer = SummaryWriter(log_dir=args.log_path)
    os.makedirs(args.save_path, exist_ok=True)

    best_map = -1.0
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, args)
        scheduler.step()
        metrics = validate(model, val_loader, device)
        mAP = metrics['map'].item()

        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, mAP: {mAP:.4f}")

        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pth"))
        if mAP > best_map:
            best_map = mAP
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pth"))

    writer.close()


if __name__ == "__main__":
    main()