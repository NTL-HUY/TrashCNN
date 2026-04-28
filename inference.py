"""
infer.py — Load model, chạy inference, vẽ bounding box lên ảnh
Usage:
    python infer.py                          # dùng valid set
    python infer.py --image path/to/img.jpg  # dùng ảnh tuỳ chọn
"""

import argparse
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from dataset import TrashDataset, collate_fn
from model import build_model

# ─── Config ────────────────────────────────────────────────────
DATA_ROOT   = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"
MODEL_PATH  = "checkpoints/best_model.pth"   # hoặc "trash_model.pth"
NUM_CLASSES = 7
SCORE_THRESH = 0.4   # chỉ vẽ box có confidence >= ngưỡng này
NUM_IMAGES   = 4     # số ảnh hiển thị cùng lúc (nếu dùng valid set)

CLASS_NAMES = [
    "background",       # 0 — bỏ qua
    "Bottle",           # 1
    "Can",              # 2
    "Carton",           # 3
    "Cup",              # 4
    "Lid",              # 5
    "Plastic bag",      # 6
]

PALETTE = [
    "#FF4757", "#2ED573", "#1E90FF",
    "#FFA502", "#ECCC68", "#A29BFE",
]
# ───────────────────────────────────────────────────────────────


def load_model(path, device):
    ckpt  = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt   # hỗ trợ cả 2 dạng save
    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("avg_loss", "?")
    print(f"✔ Loaded model  (epoch={epoch}, best_loss={loss})")
    return model


def predict(model, images, device):
    tensors = [transforms.ToTensor()(img).to(device) for img in images]
    with torch.no_grad():
        outputs = model(tensors)
    return outputs


def draw_results(images, outputs, score_thresh=SCORE_THRESH):
    n = len(images)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
    fig.patch.set_facecolor("#0D1117")
    axes = [axes] if n == 1 else axes.flatten()

    for ax, img, out in zip(axes, images, outputs):
        ax.imshow(img)
        ax.set_facecolor("#0D1117")
        ax.axis("off")

        boxes  = out["boxes"].cpu()
        labels = out["labels"].cpu()
        scores = out["scores"].cpu()

        drawn = 0
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1

            color      = PALETTE[(label.item() - 1) % len(PALETTE)]
            class_name = CLASS_NAMES[label.item()] if label < len(CLASS_NAMES) else f"cls{label}"

            # Vẽ bounding box
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2.5, edgecolor=color,
                facecolor=color + "22",   # fill mờ
            )
            ax.add_patch(rect)

            # Label badge
            label_text = f"{class_name}  {score:.0%}"
            ax.text(
                x1 + 4, y1 - 7,
                label_text,
                color="white",
                fontsize=10, fontweight="bold",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none", alpha=0.85),
            )
            drawn += 1

        title = f"{drawn} object{'s' if drawn != 1 else ''} detected"
        ax.set_title(title, color="white", fontsize=13,
                     fontfamily="monospace", pad=8)

    # Ẩn axes thừa
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("🗑  Trash Detector — Inference Results",
                 color="white", fontsize=16, fontweight="bold",
                 fontfamily="monospace", y=1.01)
    plt.tight_layout()
    plt.savefig("inference_result.png", dpi=150,
                bbox_inches="tight", facecolor="#0D1117")
    print("✔ Saved → inference_result.png")
    plt.show()


# ─── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a single image file")
    parser.add_argument("--model",  type=str, default=MODEL_PATH)
    parser.add_argument("--thresh", type=float, default=SCORE_THRESH)
    parser.add_argument("--n",      type=int,   default=NUM_IMAGES,
                        help="Number of images from valid set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, device)

    # ── Chế độ 1: ảnh tuỳ chọn ──
    if args.image:
        img = Image.open(args.image).convert("RGB")
        images  = [img]
        outputs = predict(model, images, device)

    # ── Chế độ 2: lấy ngẫu nhiên từ valid set ──
    else:
        dataset = TrashDataset(root=DATA_ROOT, split="valid",
                               transforms=None)   # giữ PIL để dễ vẽ

        indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))
        images  = []
        for idx in indices:
            img, _ = dataset[idx]
            if not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            images.append(img)

        outputs = predict(model, images, device)

    # In kết quả raw
    for i, out in enumerate(outputs):
        mask = out["scores"] >= args.thresh
        print(f"\n[Image {i+1}] {mask.sum().item()} detections (>= {args.thresh:.0%})")
        for box, lbl, sc in zip(out["boxes"][mask], out["labels"][mask], out["scores"][mask]):
            name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"cls{lbl}"
            print(f"  {name:15s}  conf={sc:.3f}  box={[round(v,1) for v in box.tolist()]}")

    draw_results(images, outputs, score_thresh=args.thresh)


if __name__ == "__main__":
    main()