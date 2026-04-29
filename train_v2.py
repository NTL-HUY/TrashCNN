import argparse
import os
import shutil
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from TrashCNN.dataset import TrashDataset, collate_fn
from TrashCNN.model import build_model
from tqdm.autonotebook import tqdm
import numpy as np

DATA_ROOT = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_CLASSES = 7


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--data_path", type=str, default=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default="tensorboard/TrashCNN")
    parser.add_argument("--save_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_dataset = TrashDataset(
        root=args.data_path,
        split='train',
        transforms=transforms.ToTensor()
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_dataset = TrashDataset(
        root=args.data_path,
        split='test',
        transforms=transforms.ToTensor()
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
    model = build_model(num_classes=NUM_CLASSES).to(device)
    opimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    best_map = -1
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_data_loader, colour="cyan")
        for iter, (images, targets) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_components = model(images, targets)
            losses = sum(loss for loss in loss_components.values())

            opimizer.zero_grad()  # tat luu tru value cua gradient trong buffer ?
            losses.backward()  # tinh dao ham (gradient) (quy tac chain rule)
            opimizer.step()  # w = w-lr*grad , lay gia tri cua gradient de update weight

            # print
            train_progress_bar.set_description(f"Epoch{epoch + 1}/{args.epochs}. Loss: {losses:.4f}")

            train_loss.append(losses.item())
            avg_loss = np.mean(train_loss)  # tensor.item(), lay value that cua tensor
            writer.add_scalar("Train/Losss", avg_loss, epoch * len(train_data_loader) + iter)

        # EVAL
        # tinh MAP
        model.eval()
        val_progress_bar = tqdm(val_data_loader, colour="yellow")
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")
        for iter, (images, targets) in enumerate(val_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():  # khong tinh backward()
                predictions = model(images)
                metric.update(predictions, targets)
                # post process
        map = metric.compute()
        writer.add_scalar("Val/mAP", map["map"].item(), epoch)
        # pprint(metric.compute())
        # save model
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opimizer.state_dict(),
        }
        if map["map"] > best_map:
            # triển khai
            save_path = os.path.join(args.save_path, "best_model.pth")
            torch.save(checkpoint, save_path)
            best_map = map["map"]
        # train tiếp
        save_path = os.path.join(args.save_path, "last_model.pth")
        torch.save(checkpoint, save_path)
        print(f"Epoch {epoch + 1} - mAP: {map['map']:.4f}")
    writer.close()


if __name__ == "__main__":
    args = get_args()
    train(args)
