"""
check_dataset.py  –  Kiểm tra dataset và visualize bbox
========================================================
- Tạo instance TrashDataset
- Vẽ bbox lên ảnh gốc (TRƯỚC transform) để kiểm tra alignment
- Lưu grid ảnh ra file PNG + hiện lên màn hình (nếu có display)

Usage:
    python check_dataset.py                    # 12 ảnh ngẫu nhiên, split train
    python check_dataset.py --split valid --n 6
    python check_dataset.py --idx 0 5 10 15    # chỉ định index cụ thể
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # đổi thành "Agg" nếu không có display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Config – chỉnh path ────────────────────────────────────────────────────
try:
    from FolderDemo.config import Config
    ROOT           = Config.DATA_TACO_PATH
    ANNOTATION_FILE = "_annotations.processed.coco.json"
except ImportError:
    ROOT            = r"D:\Projects\Workspace\Coding\Dataset\TACO dataset.v1i.coco"
    ANNOTATION_FILE = "_annotations.processed.coco.json"

OUTPUT_DIR = "./output_check"

# ── Palette (1-indexed, 0=background) ─────────────────────────────────────
COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#C77DFF",
    "#F4A261", "#2EC4B6", "#E76F51", "#A8DADC", "#457B9D",
]


# ──────────────────────────────────────────────────────────────────────────
# Raw loader (KHÔNG dùng transforms để thấy bbox thật)
# ──────────────────────────────────────────────────────────────────────────

class RawTrashDataset:
    """
    Load processed COCO và trả về (PIL.Image, boxes_xyxy, labels, img_info).
    Không apply transform — dùng để verify bbox alignment.
    """

    def __init__(self, root: str, split: str, annotation_file: str):
        self.split_dir = Path(root) / split

        ann_path = self.split_dir / annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Không tìm thấy: {ann_path}"
                f"\nChạy preprocess.py trước hoặc kiểm tra lại tên file annotation."
            )

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        self.categories    = coco["categories"]
        self.id_to_name    = {c["id"]: c["name"] for c in self.categories}
        self.cat_id_to_label = {c["id"]: c["id"] + 1 for c in self.categories}
        self.label_to_name = {c["id"] + 1: c["name"] for c in self.categories}

        from collections import defaultdict
        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        self.images     = [img for img in coco["images"] if img_to_anns.get(img["id"])]
        self.img_to_anns = img_to_anns

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        info  = self.images[idx]
        image = Image.open(self.split_dir / info["file_name"]).convert("RGB")

        boxes, labels = [], []
        for ann in self.img_to_anns[info["id"]]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        return image, boxes, labels, info

    def get_num_classes(self):
        return len(self.categories) + 1

    def sample_indices(self, n: int, seed: int = 42) -> list[int]:
        rng = random.Random(seed)
        return rng.sample(range(len(self)), min(n, len(self)))


# ──────────────────────────────────────────────────────────────────────────
# VISUALIZE
# ──────────────────────────────────────────────────────────────────────────

def _color_for_label(label: int) -> str:
    return COLORS[(label - 1) % len(COLORS)]


def draw_sample(ax, image: Image.Image, boxes: list, labels: list,
                label_map: dict, title: str = ""):
    """Vẽ 1 ảnh + tất cả bbox lên ax."""
    ax.imshow(np.array(image))
    ax.set_axis_off()

    img_w, img_h = image.size

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # ── kiểm tra bbox có nằm trong ảnh không ──────────────────────────
        in_bounds = (0 <= x1 < img_w) and (0 <= y1 < img_h) and \
                    (x2 <= img_w) and (y2 <= img_h)
        edge_color = _color_for_label(label)
        line_style = "-" if in_bounds else "--"   # nét đứt nếu lệch biên

        rect = patches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=2, edgecolor=edge_color,
            facecolor="none", linestyle=line_style,
        )
        ax.add_patch(rect)

        # Label text
        name = label_map.get(label, f"cls{label}")
        ax.text(
            x1, max(y1 - 4, 0),
            f"{name}",
            color="white", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor=edge_color, alpha=0.85, linewidth=0),
        )

    # Tiêu đề: tên file + kích thước ảnh
    short_title = title if len(title) <= 30 else "…" + title[-27:]
    ax.set_title(
        f"{short_title}\n{img_w}×{img_h}  |  {len(boxes)} obj",
        color="#ddddff", fontsize=7, pad=4,
    )


def build_grid(
    dataset: RawTrashDataset,
    indices: list[int],
    cols: int = 4,
    figsize_per_cell: tuple = (4, 4),
    title: str = "",
) -> plt.Figure:
    rows = (len(indices) + cols - 1) // cols
    fig_w = cols * figsize_per_cell[0]
    fig_h = rows * figsize_per_cell[1] + (0.6 if title else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                             facecolor="#0d0d1a")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_flat:
        ax.set_facecolor("#0d0d1a")
        ax.set_axis_off()

    for plot_i, ds_idx in enumerate(indices):
        image, boxes, labels, info = dataset[ds_idx]
        ax = axes_flat[plot_i]
        draw_sample(ax, image, boxes, labels,
                    dataset.label_to_name,
                    title=info.get("file_name", str(ds_idx)))

    if title:
        fig.suptitle(title, color="#e8e8ff", fontsize=13,
                     fontweight="bold", y=1.002)

    fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# STATS PRINT
# ──────────────────────────────────────────────────────────────────────────

def print_dataset_info(ds: RawTrashDataset, split: str):
    from collections import Counter
    label_counts = Counter()
    ann_per_img  = []
    for img_info in ds.images:
        anns = ds.img_to_anns[img_info["id"]]
        ann_per_img.append(len(anns))
        for ann in anns:
            name = ds.id_to_name.get(ann["category_id"], "?")
            label_counts[name] += 1

    total_anns = sum(label_counts.values())
    print(f"\n{'═'*55}")
    print(f"  Dataset  split={split.upper()}")
    print(f"{'═'*55}")
    print(f"  Images      : {len(ds):,}")
    print(f"  Annotations : {total_anns:,}")
    print(f"  Classes     : {ds.get_num_classes() - 1}  (+1 background = {ds.get_num_classes()})")
    print(f"  Ann/image   : mean={np.mean(ann_per_img):.2f}  "
          f"max={max(ann_per_img)}  min={min(ann_per_img)}")
    print(f"\n  {'Class':<16} {'Count':>6}  {'%':>6}")
    print(f"  {'─'*32}")
    for name, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(25 * cnt / total_anns)
        print(f"  {name:<16} {cnt:>6,}  {100*cnt/total_anns:>5.1f}%  {bar}")
    print()


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",   default=ROOT)
    p.add_argument("--split",  default="train", choices=["train", "valid", "test"])
    p.add_argument("--ann",    default=ANNOTATION_FILE,
                   help="Tên file annotation (mặc định: _annotations.processed.coco.json)")
    p.add_argument("--n",      type=int, default=12,
                   help="Số ảnh random để hiển thị")
    p.add_argument("--cols",   type=int, default=4)
    p.add_argument("--idx",    type=int, nargs="*",
                   help="Chỉ định index cụ thể, ví dụ --idx 0 5 10")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--no-show", action="store_true",
                   help="Chỉ lưu file, không plt.show()")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n  Loading dataset …")
    print(f"  root  = {args.root}")
    print(f"  split = {args.split}")
    print(f"  ann   = {args.ann}")

    ds = RawTrashDataset(args.root, args.split, args.ann)
    print_dataset_info(ds, args.split)

    # ── Chọn indices ────────────────────────────────────────────────────
    if args.idx:
        indices = args.idx
        print(f"  Visualizing indices: {indices}")
    else:
        indices = ds.sample_indices(args.n, seed=args.seed)
        print(f"  Visualizing {len(indices)} random samples (seed={args.seed})")

    # ── Build grid ──────────────────────────────────────────────────────
    fig = build_grid(
        ds, indices,
        cols=args.cols,
        title=f"BBox Check  –  {args.split.upper()}  "
              f"({ds.get_num_classes()-1} classes)",
    )

    out_path = os.path.join(OUTPUT_DIR, f"bbox_check_{args.split}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ✅ Saved → {out_path}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()