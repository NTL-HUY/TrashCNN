"""
main.py – Entry Point Chính
============================
CLI đơn giản với 3 lệnh:
  python main.py train   – Bắt đầu training
  python main.py eval    – Đánh giá model trên test set (tính mAP)
  python main.py predict – Chạy inference trên 1 ảnh

Luồng tổng quát:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. preprocess.py  → resize ảnh vào data/processed/     │
  │ 2. main.py train  → train model, lưu checkpoint/       │
  │ 3. main.py eval   → đánh giá trên test set             │
  │ 4. main.py predict → inference ảnh mới                 │
  └─────────────────────────────────────────────────────────┘

TensorBoard:
  %load_ext tensorboard
  %tensorboard --logdir logs/tb
"""

import os
import sys
import argparse

import torch

from src.config import (
    ANNOTATION_FILE, PROCESSED_DIR,
    CHECKPOINT_DIR, LOG_DIR,
    NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, FREEZE_BACKBONE,
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

    # ── Kiểm tra data đã được preprocess chưa ────────────────────────────
    if not os.path.isdir(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print("[Main] ⚠  Chưa tìm thấy data/processed/. Hãy chạy trước:")
        print("           python scripts/preprocess.py")
        sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, _ = create_data_loaders(
        ANNOTATION_FILE, PROCESSED_DIR,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE)
    model.to(device)

    # Resume từ checkpoint nếu được chỉ định
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[Main] Resume từ epoch {start_epoch}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = get_optimizer(model)
    scheduler = get_lr_scheduler(optimizer)

    # ── Logging & Early Stopping ──────────────────────────────────────────
    logger = LossLogger(
        log_dir=LOG_DIR,
        use_tensorboard=not args.no_tensorboard,
    )
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")

    writer = logger.writer

    print(f"\n{'='*55}")
    print(f"  BẮT ĐẦU TRAINING")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  Device  : {device}")
    print(f"  Backbone: {'FROZEN' if FREEZE_BACKBONE else 'TRAINABLE'}")
    print(f"{'='*55}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────")

        # Train
        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            print_freq=args.print_freq,
            writer=writer,
        )

        # Validation
        val_losses = validate(
            model, val_loader, device, epoch,
            writer=writer,
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Main] LR hiện tại: {current_lr:.6f}")

        # Logging
        logger.record(epoch, train_losses, val_losses)

        # Lưu checkpoint tốt nhất
        val_total = val_losses["total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = os.path.join(CHECKPOINT_DIR, "best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch,
                           {"val_loss": val_total}, best_path)

        # Lưu checkpoint theo epoch (mỗi N epoch)
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch,
                           {"val_loss": val_total}, ckpt_path)

        # Early stopping
        if early_stopper.step(val_total):
            print(f"[Main] Early stopping tại epoch {epoch}")
            break

    print(f"\n[Main] Training xong! Best val loss: {best_val_loss:.4f}")
    print(f"[Main] Checkpoint tốt nhất: {os.path.join(CHECKPOINT_DIR, 'best.pth')}")

    # ── Đóng TensorBoard writer ───────────────────────────────────────────
    logger.close()

    # Vẽ đồ thị loss
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
    """Đánh giá model trên test set, in mAP."""
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

    # Lưu kết quả
    import json
    out_path = os.path.join(LOG_DIR, "eval_results.json")
    os.makedirs(LOG_DIR, exist_ok=True)

    def _tensor_to_json(v):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.item())
            return [float(x) for x in v.tolist()]
        return v

    with open(out_path, "w") as f:
        json.dump({k: _tensor_to_json(v) for k, v in results.items()},
                  f, indent=2)
    print(f"[Eval] Kết quả đã lưu → {out_path}")


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------

def run_predict(args) -> None:
    """Inference 1 ảnh và in kết quả ra màn hình."""
    from scripts.visualize import visualize_single_image

    device = setup_device()
    model  = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    out_path = args.output or "output/prediction.jpg"
    visualize_single_image(
        model, device, args.image,
        score_thresh=args.score_thresh,
        output_path=out_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Waste Detection – FasterRCNN + ResNet50 Feature Extraction"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Bắt đầu training")
    p_train.add_argument("--epochs",          type=int,   default=NUM_EPOCHS)
    p_train.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    p_train.add_argument("--workers",         type=int,   default=NUM_WORKERS)
    p_train.add_argument("--patience",        type=int,   default=7,
                         help="Early stopping patience")
    p_train.add_argument("--save-every",      type=int,   default=5,
                         help="Lưu checkpoint mỗi N epoch")
    p_train.add_argument("--print-freq",      type=int,   default=20,
                         help="In log mỗi N batch")
    p_train.add_argument("--resume",          type=str,   default=None,
                         help="Đường dẫn checkpoint để resume training")
    p_train.add_argument("--no-tensorboard",  action="store_true",
                         help="Tắt TensorBoard logging")

    # ── eval ───────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Đánh giá model (mAP)")
    p_eval.add_argument("--checkpoint",   required=True)
    p_eval.add_argument("--score-thresh", type=float, default=0.3)

    # ── predict ────────────────────────────────────────────────────────
    p_pred = subparsers.add_parser("predict", help="Inference 1 ảnh")
    p_pred.add_argument("--checkpoint",   required=True)
    p_pred.add_argument("--image",        required=True,
                        help="Đường dẫn ảnh đầu vào")
    p_pred.add_argument("--score-thresh", type=float, default=0.3)
    p_pred.add_argument("--output",       type=str,   default=None,
                        help="Đường dẫn ảnh kết quả (default: output/prediction.jpg)")

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