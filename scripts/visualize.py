"""
scripts/visualize.py – Visualize kết quả dự đoán
==================================================
Dùng để kiểm tra model sau khi train, vẽ bounding box lên ảnh mẫu.

Cách dùng:
  # Visualize N ảnh từ test set (lưu ra thư mục output/)
  python scripts/visualize.py --checkpoint checkpoints/best.pth --n 10

  # Visualize 1 ảnh cụ thể
  python scripts/visualize.py --checkpoint checkpoints/best.pth \\
                               --image data/processed/batch_1/xxx.jpg
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ANNOTATION_FILE, PROCESSED_DIR,
    CHECKPOINT_DIR, SUPERCLASS_NAMES,
)
from src.model import build_model, load_checkpoint
from src.transforms import get_val_transforms
from src.utils import setup_device, draw_predictions, create_data_loaders
from src.evaluator import predict_single


# ---------------------------------------------------------------------------
def visualize_from_dataset(
    model:       torch.nn.Module,
    device:      torch.device,
    n_images:    int = 10,
    score_thresh: float = 0.3,
    output_dir:  str = "output/visualize",
):
    """Lấy n ảnh từ test set, chạy model, lưu kết quả."""
    try:
        import cv2
    except ImportError:
        print("Cần cài: pip install opencv-python")
        return

    os.makedirs(output_dir, exist_ok=True)

    _, _, test_loader = create_data_loaders(
        ANNOTATION_FILE, PROCESSED_DIR, batch_size=1
    )

    transform = get_val_transforms()

    for i, (images, targets) in enumerate(test_loader):
        if i >= n_images:
            break

        image_tensor = images[0]
        prediction   = predict_single(model, image_tensor, device, score_thresh)

        # Chuyển tensor ảnh về numpy BGR để OpenCV vẽ
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean).clip(0, 1)
        img_bgr = (img_np * 255).astype(np.uint8)[:, :, ::-1].copy()

        # Vẽ predictions
        img_result = draw_predictions(
            img_bgr,
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
            score_thresh,
        )

        # Vẽ ground truth (viền xanh lá nhạt)
        gt_boxes  = targets[0]["boxes"]
        gt_labels = targets[0]["labels"]
        for box, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 200, 0), 1)

        out_path = os.path.join(output_dir, f"result_{i:04d}.jpg")
        cv2.imwrite(out_path, img_result)

        n_pred = len(prediction["boxes"])
        print(f"[{i+1:03d}] {n_pred} detection(s) → {out_path}")

    print(f"\n[Visualize] Đã lưu {min(n_images, i+1)} ảnh vào '{output_dir}'")


def visualize_single_image(
    model:        torch.nn.Module,
    device:       torch.device,
    image_path:   str,
    score_thresh: float = 0.3,
    output_path:  str = "output/prediction.jpg",
):
    """Chạy model trên 1 ảnh bất kỳ và lưu kết quả."""
    try:
        import cv2
    except ImportError:
        print("Cần cài: pip install opencv-python")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    transform = get_val_transforms()
    image     = Image.open(image_path).convert("RGB")
    tensor, _ = transform(image, {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)})

    prediction = predict_single(model, tensor, device, score_thresh)

    # Convert sang BGR
    img_np  = tensor.permute(1, 2, 0).cpu().numpy()
    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    img_np  = (img_np * std + mean).clip(0, 1)
    img_bgr = (img_np * 255).astype(np.uint8)[:, :, ::-1].copy()

    result = draw_predictions(
        img_bgr,
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"],
        score_thresh,
    )

    cv2.imwrite(output_path, result)

    print(f"[Visualize] {len(prediction['boxes'])} detection(s)")
    for box, lbl, sc in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        cls = SUPERCLASS_NAMES.get(lbl.item(), "?")
        print(f"  {cls:<10} score={sc:.3f}  box={box.int().tolist()}")
    print(f"[Visualize] Đã lưu → {output_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize waste detection")
    parser.add_argument("--checkpoint", required=True,
                        help="Đường dẫn checkpoint .pth")
    parser.add_argument("--image", default=None,
                        help="Đường dẫn ảnh cụ thể (tùy chọn)")
    parser.add_argument("--n", type=int, default=10,
                        help="Số ảnh từ test set (nếu không dùng --image)")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Ngưỡng confidence")
    parser.add_argument("--output-dir", default="output/visualize",
                        help="Thư mục lưu kết quả")
    args = parser.parse_args()

    device = setup_device()
    model  = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    if args.image:
        out = os.path.join(args.output_dir, "prediction.jpg")
        visualize_single_image(model, device, args.image, args.score_thresh, out)
    else:
        visualize_from_dataset(model, device, args.n, args.score_thresh, args.output_dir)
