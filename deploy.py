"""
  1. Dùng DetectionTransforms thay transforms.ToTensor()
     → Inference PHẢI dùng cùng normalize như lúc train (mean/std)
     → Bản cũ thiếu normalize → model nhận input khác distribution → predict sai
  2. Dùng build_inference_model() → score_thresh cao hơn để lọc noise
  3. Tên class lấy từ dataset thay vì hardcode list
  4. Hỗ trợ deploy trên ảnh file tùy ý (--image)
  5. Hiển thị confidence score rõ ràng hơn
"""

import argparse

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TrashDataset, DetectionTransforms
from model import build_inference_model

# ─────────────────────────── Colors ──────────────────────────────────────
GT_COLOR = (255, 80, 80)  # Đỏ - Ground Truth
PRED_COLOR = (80, 220, 80)  # Xanh lá - Prediction

CLASS_PALETTE = [
    (255, 99, 132), (255, 159, 64), (255, 205, 86),
    (75, 192, 192), (54, 162, 235), (153, 102, 255),
    (201, 203, 207),
]


# ─────────────────────────── Helpers ─────────────────────────────────────

def tensor_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Undo normalization và chuyển tensor sang numpy uint8 để hiển thị.
    Dùng ImageNet mean/std (phải undo đúng cái đã apply lúc transform).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean  # undo normalize
    img = img.permute(1, 2, 0).numpy()  # CHW → HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img.copy()


def draw_box(img_np, x1, y1, x2, y2, color, text):
    """Vẽ bounding box + label lên ảnh numpy."""
    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Nền chữ
    cv2.rectangle(img_np, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img_np, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ─────────────────────────── Deploy with GT ──────────────────────────────

def deploy_with_gt(args):
    """So sánh Ground Truth vs Prediction trên ảnh từ dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────
    dataset = TrashDataset(
        root=args.data_path,
        split=args.split,
        transforms=DetectionTransforms(is_train=False),  # chỉ normalize
    )
    num_classes = len(dataset.categories) + 1
    class_names = {v: k for k, v in {
        i + 1: cat["name"] for i, cat in enumerate(dataset.categories)
    }.items()}
    label_to_name = {i + 1: cat["name"] for i, cat in enumerate(dataset.categories)}

    model = build_inference_model(
        num_classes=num_classes,
        score_thresh=args.score_thresh,
    ).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {args.model_path}")
    print(f"Best mAP: {ckpt.get('best_map', 'N/A')}")

    # ── Predict ───────────────────────────────────────────────────────
    img_tensor, target = dataset[args.index]

    with torch.no_grad():
        pred = model([img_tensor.to(device)])[0]

    pred_boxes = pred["boxes"].cpu().numpy()
    pred_labels = pred["labels"].cpu().numpy()
    pred_scores = pred["scores"].cpu().numpy()
    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()

    # ── Convert sang numpy để vẽ ──────────────────────────────────────
    img_np = tensor_to_numpy(img_tensor)

    # Vẽ GT (đỏ)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        name = label_to_name.get(int(label), f"cls{label}")
        draw_box(img_np, x1, y1, x2, y2, GT_COLOR, f"GT: {name}")

    # Vẽ Predictions (xanh)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = map(int, box)
        name = label_to_name.get(int(label), f"cls{label}")
        draw_box(img_np, x1, y1, x2, y2, PRED_COLOR, f"{name} {score:.2f}")

    # ── Hiển thị ──────────────────────────────────────────────────────
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(
        f"Index {args.index} | "
        f"GT: {len(gt_boxes)} boxes | "
        f"Pred: {(pred_scores >= args.score_thresh).sum()} boxes"
    )
    legend = [
        mpatches.Patch(color=(GT_COLOR[0] / 255, GT_COLOR[1] / 255, GT_COLOR[2] / 255), label="Ground Truth"),
        mpatches.Patch(color=(PRED_COLOR[0] / 255, PRED_COLOR[1] / 255, PRED_COLOR[2] / 255), label="Prediction"),
    ]
    plt.legend(handles=legend, loc="upper right", fontsize=10)
    plt.tight_layout()

    if args.save_img:
        out_path = f"result_idx{args.index}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.show()

    print(f"\nGT boxes  : {len(gt_boxes)}")
    print(f"Pred boxes: {len(pred_boxes)} total, "
          f"{(pred_scores >= args.score_thresh).sum()} above threshold {args.score_thresh}")


# ─────────────────────────── Deploy on custom image ──────────────────────

def deploy_image(args):
    """Predict trên ảnh file tùy ý (không cần dataset)."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Determine classes từ dataset (để lấy tên class) ──────────────
    dataset = TrashDataset(
        root=args.data_path,
        split="train",
    )
    num_classes = len(dataset.categories) + 1
    label_to_name = {i + 1: cat["name"] for i, cat in enumerate(dataset.categories)}

    # ── Load model ────────────────────────────────────────────────────
    model = build_inference_model(
        num_classes=num_classes,
        score_thresh=args.score_thresh,
    ).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Preprocess image ──────────────────────────────────────────────
    pil_img = Image.open(args.image).convert("RGB")
    img_tensor = TF.to_tensor(pil_img)
    img_tensor = TF.normalize(img_tensor,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    # ── Predict ───────────────────────────────────────────────────────
    with torch.no_grad():
        pred = model([img_tensor.to(device)])[0]

    img_np = np.array(pil_img)

    for box, label, score in zip(
            pred["boxes"].cpu().numpy(),
            pred["labels"].cpu().numpy(),
            pred["scores"].cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(int, box)
        name = label_to_name.get(int(label), f"cls{label}")
        draw_box(img_np, x1, y1, x2, y2, PRED_COLOR, f"{name} {score:.2f}")

    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(f"Detected {len(pred['boxes'])} objects  (thresh={args.score_thresh})")
    plt.tight_layout()
    if args.save_img:
        plt.savefig("result_custom.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────── CLI ─────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=r".\TACO dataset.v1i.coco")
    p.add_argument("--model_path", type=str, default="trained_models/best_model.pth")
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--score_thresh", type=float, default=0.4)
    p.add_argument("--image", type=str, default=None,
                   help="Path ảnh tùy ý để deploy (không cần dataset)")
    p.add_argument("--save_img", action="store_true",
                   help="Lưu kết quả ra file PNG")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.image:
        deploy_image(args)
    else:
        deploy_with_gt(args)
