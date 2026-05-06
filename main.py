"""
main.py – Entry Point Chính
============================
CLI với 3 lệnh:
  python main.py train   – Train FasterRCNN from scratch
  python main.py eval    – Đánh giá mAP trên test set
  python main.py predict – Inference 1 ảnh

TensorBoard:
  %load_ext tensorboard
  %tensorboard --logdir logs/tb
"""

import os
import sys
import json
import argparse

import torch

from src.config import (
    ANNOTATION_FILE, PROCESSED_DIR,
    CHECKPOINT_DIR, LOG_DIR,
    NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES,
)
from src.model import (
    build_model, get_optimizer, get_lr_scheduler,
    load_checkpoint, save_checkpoint,
)
from src.utils import (
    setup_device, seed_everything, create_data_loaders,
    LossLogger, ensure_dirs, plot_losses,
)
from src.trainer import train_one_epoch, validate, EarlyStopping
from src.evaluator import evaluate


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def run_train(args) -> None:
    seed_everything()
    ensure_dirs()
    device = setup_device()

    if not os.path.isdir(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print("[Main] ⚠  Chưa tìm thấy data/processed/. Hãy chạy trước:")
        print("           python scripts/preprocess.py")
        sys.exit(1)

    train_loader, val_loader, _ = create_data_loaders(
        ANNOTATION_FILE, PROCESSED_DIR,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = build_model(num_classes=NUM_CLASSES)
    model.to(device)

    start_epoch = 1
    if args.resume:
        ckpt        = load_checkpoint(model, args.resume, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Main] Resume từ epoch {start_epoch}")

    optimizer = get_optimizer(model)
    # ReduceLROnPlateau – cần truyền val_loss vào scheduler.step()
    scheduler = get_lr_scheduler(optimizer)

    logger        = LossLogger(LOG_DIR, use_tensorboard=not args.no_tensorboard)
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")
    writer        = logger.writer

    print(f"\n{'='*60}")
    print(f"  FasterRCNN FROM SCRATCH")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Device   : {device}")
    print(f"  TensorBoard: {'BẬT → logs/tb' if writer else 'TẮT'}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} " + "─" * 30)

        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            print_freq=args.print_freq, writer=writer,
        )
        val_losses = validate(
            model, val_loader, device, epoch, writer=writer,
        )

        val_total = val_losses["total"]

        # ReduceLROnPlateau nhận val_loss (khác StepLR)
        scheduler.step(val_total)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Main] LR hiện tại: {current_lr:.7f}")

        logger.record(epoch, train_losses, val_losses)

        # Lưu best checkpoint
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_loss": val_total},
                os.path.join(CHECKPOINT_DIR, "best.pth"),
            )

        # Lưu checkpoint mỗi N epoch
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_loss": val_total},
                os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"),
            )

        if early_stopper.step(val_total):
            print(f"[Main] Early stopping tại epoch {epoch}")
            break

    logger.close()
    print(f"\n[Main] Training xong! Best val loss: {best_val_loss:.4f}")

    try:
        plot_losses(
            os.path.join(LOG_DIR, "training_log.json"),
            save_path=os.path.join(LOG_DIR, "loss_curve.png"),
        )
    except Exception as e:
        print(f"[Main] Không vẽ được đồ thị: {e}")


# ---------------------------------------------------------------------------
# EVAL
# ---------------------------------------------------------------------------

def run_eval(args) -> None:
    device = setup_device()
    _, _, test_loader = create_data_loaders(
        ANNOTATION_FILE, PROCESSED_DIR, batch_size=1
    )

    model = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    results = evaluate(
        model, test_loader, device,
        score_threshold=args.score_thresh,
    )

    def _to_json(v):
        if isinstance(v, torch.Tensor):
            return float(v.item()) if v.numel() == 1 else v.tolist()
        return v

    out_path = os.path.join(LOG_DIR, "eval_results.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: _to_json(v) for k, v in results.items()}, f, indent=2)
    print(f"[Eval] Kết quả đã lưu → {out_path}")


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------

def run_predict(args) -> None:
    from scripts.visualize import visualize_single_image
    device = setup_device()
    model  = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    visualize_single_image(
        model, device, args.image,
        score_thresh=args.score_thresh,
        output_path=args.output or "output/prediction.jpg",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Waste Detection – FasterRCNN"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p = sub.add_parser("train")
    p.add_argument("--epochs",         type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--workers",        type=int,   default=NUM_WORKERS)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--save-every",     type=int,   default=5)
    p.add_argument("--print-freq",     type=int,   default=20)
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--no-tensorboard", action="store_true")

    # eval
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--score-thresh", type=float, default=0.05)

    # predict
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--image",        required=True)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--output",       type=str,   default=None)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()