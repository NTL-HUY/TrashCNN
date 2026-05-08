import os
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\Projects\Workspace\Coding\Dataset\TrashType_Image_Dataset"
    )

    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument(
        "--log_path",
        type=str,
        default="tensorboard/TrashClassifier"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="trained_models"
    )

    args = parser.parse_args(args=[])

    return args


def train(args):

    # =========================================================
    # DEVICE
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # =========================================================
    # TRANSFORM
    # =========================================================
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # =========================================================
    # DATASET
    # =========================================================
    full_dataset = ImageFolder(
        root=args.data_path,
        transform=train_transform
    )

    print("Classes:", full_dataset.classes)
    print("Class to idx:", full_dataset.class_to_idx)

    num_classes = len(full_dataset.classes)

    # =========================================================
    # SPLIT TRAIN / VAL
    # =========================================================
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    # validation transform
    val_dataset.dataset.transform = val_transform

    # =========================================================
    # DATALOADER
    # =========================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # =========================================================
    # MODEL
    # =========================================================
    model = models.resnet18(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model = model.to(device)

    # =========================================================
    # LOSS + OPTIMIZER
    # =========================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # =========================================================
    # LOGGING
    # =========================================================
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    writer = SummaryWriter(args.log_path)

    best_acc = 0

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(args.epochs):

        # =========================
        # TRAIN
        # =========================
        model.train()

        train_losses = []

        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            colour="cyan"
        )

        for images, labels in train_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

            _, preds = torch.max(outputs, 1)

            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

            train_acc = train_correct / train_total

            train_bar.set_description(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {train_acc:.4f}"
            )

        avg_train_loss = np.mean(train_losses)

        # =========================
        # VALIDATION
        # =========================
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()

                val_total += labels.size(0)

        val_acc = val_correct / val_total

        scheduler.step()

        # =====================================================
        # TENSORBOARD
        # =====================================================
        writer.add_scalar(
            "Train/Loss",
            avg_train_loss,
            epoch
        )

        writer.add_scalar(
            "Train/Accuracy",
            train_acc,
            epoch
        )

        writer.add_scalar(
            "Val/Accuracy",
            val_acc,
            epoch
        )

        # =====================================================
        # SAVE MODEL
        # =====================================================
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accuracy": val_acc,
            "classes": full_dataset.classes
        }

        # BEST MODEL
        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                checkpoint,
                os.path.join(
                    args.save_path,
                    "best_model.pth"
                )
            )

        # LAST MODEL
        torch.save(
            checkpoint,
            os.path.join(
                args.save_path,
                "last_model.pth"
            )
        )

        # =====================================================
        # PRINT
        # =====================================================
        print("\n" + "=" * 60)

        print(f"Epoch {epoch + 1}/{args.epochs}")

        print("-" * 60)

        print(f"Train Loss : {avg_train_loss:.4f}")

        print(f"Train Acc  : {train_acc:.4f}")

        print(f"Val Acc    : {val_acc:.4f}")

        print(f"Best Acc   : {best_acc:.4f}")

        print(f"Learning Rate : {scheduler.get_last_lr()[0]:.6f}")

        print("=" * 60)

    writer.close()


if __name__ == "__main__":

    args = get_args()

    train(args)