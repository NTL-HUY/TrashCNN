"""
dataset_stats.py — Thống kê toàn diện cho COCO-format dataset (TrashDataset)
Chạy: python dataset_stats.py --root /path/to/dataset
"""

import os
import json
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

warnings.filterwarnings("ignore")

# ─── màu cho từng split ────────────────────────────────────────────────────────
SPLIT_COLORS = {"train": "#4CAF50", "valid": "#2196F3", "test": "#FF5722"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load dữ liệu
# ══════════════════════════════════════════════════════════════════════════════

def load_split(root: str, split: str):
    """Load COCO json cho một split. Trả về None nếu không tồn tại."""
    split_dir = os.path.join(root, split)
    ann_file = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        return None, split_dir

    with open(ann_file) as f:
        coco = json.load(f)
    return coco, split_dir


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tính thống kê
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(coco: dict, split_dir: str):
    images      = coco["images"]
    annotations = coco["annotations"]
    categories  = coco["categories"]

    cat_map = {c["id"]: c["name"] for c in categories}
    img_map = {img["id"]: img for img in images}

    # --- gom annotations theo ảnh ---
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # --- per-image stats ---
    widths, heights, ann_counts = [], [], []
    aspect_ratios = []
    missing_files = []

    for img in images:
        w, h = img.get("width", 0), img.get("height", 0)
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        ann_counts.append(len(img_to_anns[img["id"]]))

        img_path = os.path.join(split_dir, img["file_name"])
        if not os.path.exists(img_path):
            missing_files.append(img["file_name"])

    # --- per-bbox stats ---
    bbox_widths, bbox_heights, bbox_areas = [], [], []
    cat_counts = defaultdict(int)
    invalid_bboxes = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cat_counts[cat_map[ann["category_id"]]] += 1

        if w <= 0 or h <= 0:
            invalid_bboxes.append(ann)
            continue

        img_info = img_map.get(ann["image_id"], {})
        iw, ih = img_info.get("width", 1), img_info.get("height", 1)

        bbox_widths.append(w / iw)   # normalized
        bbox_heights.append(h / ih)
        bbox_areas.append((w * h) / (iw * ih))

    # --- ảnh không có annotation ---
    no_ann_images = [img for img in images if len(img_to_anns[img["id"]]) == 0]

    return {
        "n_images":       len(images),
        "n_annotations":  len(annotations),
        "n_categories":   len(categories),
        "categories":     [c["name"] for c in categories],
        "cat_counts":     dict(cat_counts),
        "widths":         widths,
        "heights":        heights,
        "aspect_ratios":  aspect_ratios,
        "ann_counts":     ann_counts,
        "bbox_widths":    bbox_widths,
        "bbox_heights":   bbox_heights,
        "bbox_areas":     bbox_areas,
        "missing_files":  missing_files,
        "invalid_bboxes": invalid_bboxes,
        "no_ann_images":  no_ann_images,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. In báo cáo text
# ══════════════════════════════════════════════════════════════════════════════

def _stat(arr, label, fmt=".1f"):
    if len(arr) == 0:
        print(f"  {label}: N/A")
        return
    a = np.array(arr)
    print(f"  {label}: mean={a.mean():{fmt}}  std={a.std():{fmt}}"
          f"  min={a.min():{fmt}}  max={a.max():{fmt}}")


def print_report(split: str, stats: dict):
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  SPLIT: {split.upper()}")
    print(sep)

    print(f"\n📁 Tổng quan")
    print(f"  Số ảnh         : {stats['n_images']}")
    print(f"  Số annotation  : {stats['n_annotations']}")
    print(f"  Số class       : {stats['n_categories']}")
    print(f"  Classes        : {', '.join(stats['categories'])}")

    print(f"\n🖼️  Kích thước ảnh")
    _stat(stats["widths"],        "Width  (px)")
    _stat(stats["heights"],       "Height (px)")
    _stat(stats["aspect_ratios"], "Aspect ratio", fmt=".3f")

    print(f"\n📦 Annotation / ảnh")
    _stat(stats["ann_counts"], "Annotations/img", fmt=".2f")
    zero = len(stats["no_ann_images"])
    print(f"  Ảnh không có bbox : {zero} ({100*zero/max(stats['n_images'],1):.1f}%)")

    print(f"\n📐 Kích thước bbox (normalized 0–1)")
    _stat(stats["bbox_widths"],  "Width ")
    _stat(stats["bbox_heights"], "Height")
    _stat(stats["bbox_areas"],   "Area  ", fmt=".4f")

    # phân phối kích thước bbox
    areas = np.array(stats["bbox_areas"])
    if len(areas):
        small  = (areas < 0.01).sum()
        medium = ((areas >= 0.01) & (areas < 0.1)).sum()
        large  = (areas >= 0.1).sum()
        total  = len(areas)
        print(f"\n  Phân loại kích thước bbox (theo diện tích):")
        print(f"    Small  (<1% ảnh)   : {small:5d}  ({100*small/total:.1f}%)")
        print(f"    Medium (1%–10%)    : {medium:5d}  ({100*medium/total:.1f}%)")
        print(f"    Large  (>10% ảnh)  : {large:5d}  ({100*large/total:.1f}%)")

    print(f"\n🏷️  Số bbox theo class:")
    for cat, cnt in sorted(stats["cat_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * min(int(cnt / max(stats["n_annotations"], 1) * 50), 50)
        print(f"  {cat:30s}: {cnt:5d}  {bar}")

    print(f"\n⚠️  Kiểm tra chất lượng")
    print(f"  File ảnh bị thiếu   : {len(stats['missing_files'])}")
    print(f"  Bbox không hợp lệ   : {len(stats['invalid_bboxes'])} (w<=0 hoặc h<=0)")

    if stats["missing_files"]:
        for f in stats["missing_files"][:5]:
            print(f"    ✗ {f}")
        if len(stats["missing_files"]) > 5:
            print(f"    ... và {len(stats['missing_files'])-5} file khác")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Vẽ biểu đồ
# ══════════════════════════════════════════════════════════════════════════════

def plot_stats(all_stats: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    splits = list(all_stats.keys())
    colors = [SPLIT_COLORS.get(s, "#9C27B0") for s in splits]

    # ── Fig 1: Class distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]
    fig.suptitle("Class Distribution", fontsize=16, fontweight="bold")

    for ax, split, color in zip(axes, splits, colors):
        cc = all_stats[split]["cat_counts"]
        if not cc:
            continue
        cats = list(cc.keys())
        vals = [cc[c] for c in cats]
        bars = ax.barh(cats, vals, color=color, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title(split, fontsize=13)
        ax.set_xlabel("Số bbox")
        ax.invert_yaxis()

    plt.tight_layout()
    p1 = os.path.join(save_dir, "1_class_distribution.png")
    plt.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved: {p1}")

    # ── Fig 2: Annotations per image ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, color in zip(splits, colors):
        counts = all_stats[split]["ann_counts"]
        if counts:
            ax.hist(counts, bins=30, alpha=0.6, color=color, label=split, edgecolor="white")
    ax.set_title("Số annotation mỗi ảnh", fontsize=14)
    ax.set_xlabel("Số annotation")
    ax.set_ylabel("Số ảnh")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(save_dir, "2_annotations_per_image.png")
    plt.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {p2}")

    # ── Fig 3: Image size scatter ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, color in zip(splits, colors):
        ax.scatter(all_stats[split]["widths"], all_stats[split]["heights"],
                   alpha=0.3, s=15, color=color, label=split)
    ax.set_title("Kích thước ảnh (W × H)", fontsize=14)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend()
    plt.tight_layout()
    p3 = os.path.join(save_dir, "3_image_sizes.png")
    plt.savefig(p3, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {p3}")

    # ── Fig 4: Bbox size heatmap (w vs h normalized) ───────────────────────
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]
    fig.suptitle("Phân phối kích thước bbox (normalized)", fontsize=14, fontweight="bold")

    for ax, split in zip(axes, splits):
        bw = all_stats[split]["bbox_widths"]
        bh = all_stats[split]["bbox_heights"]
        if bw:
            h2d, xedge, yedge = np.histogram2d(bw, bh, bins=40,
                                                range=[[0, 1], [0, 1]])
            ax.imshow(h2d.T, origin="lower", aspect="auto",
                      extent=[0, 1, 0, 1], cmap="YlOrRd")
            ax.set_title(split, fontsize=13)
            ax.set_xlabel("Bbox Width (norm)")
            ax.set_ylabel("Bbox Height (norm)")

    plt.tight_layout()
    p4 = os.path.join(save_dir, "4_bbox_size_heatmap.png")
    plt.savefig(p4, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {p4}")

    # ── Fig 5: Bbox area distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, color in zip(splits, colors):
        areas = all_stats[split]["bbox_areas"]
        if areas:
            ax.hist(np.array(areas) * 100, bins=50, alpha=0.6,
                    color=color, label=split, edgecolor="white")
    ax.axvline(1,  color="gray", linestyle="--", linewidth=1, label="Small/Medium (1%)")
    ax.axvline(10, color="gray", linestyle=":",  linewidth=1, label="Medium/Large (10%)")
    ax.set_title("Phân phối diện tích bbox (% so với ảnh)", fontsize=14)
    ax.set_xlabel("Bbox Area (%)")
    ax.set_ylabel("Số bbox")
    ax.legend()
    plt.tight_layout()
    p5 = os.path.join(save_dir, "5_bbox_area_distribution.png")
    plt.savefig(p5, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {p5}")

    # ── Fig 6: Split comparison summary ────────────────────────────────────
    if len(splits) > 1:
        metrics = ["n_images", "n_annotations"]
        labels  = ["Số ảnh", "Số annotation"]
        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (split, color) in enumerate(zip(splits, colors)):
            vals = [all_stats[split][m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=split,
                          color=color, edgecolor="white")
            ax.bar_label(bars, padding=3, fontsize=9)

        ax.set_title("So sánh các splits", fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        p6 = os.path.join(save_dir, "6_split_comparison.png")
        plt.savefig(p6, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"💾 Saved: {p6}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Sample ảnh kèm bbox
# ══════════════════════════════════════════════════════════════════════════════

def visualize_samples(coco: dict, split_dir: str, split: str,
                      save_dir: str, n: int = 6):
    images      = coco["images"]
    annotations = coco["annotations"]
    categories  = coco["categories"]
    cat_map     = {c["id"]: c["name"] for c in categories}

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # ưu tiên ảnh có nhiều bbox
    sorted_imgs = sorted(images,
                         key=lambda x: len(img_to_anns[x["id"]]),
                         reverse=True)[:n]

    cols = 3
    rows = (len(sorted_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle(f"Sample ảnh — {split}", fontsize=14, fontweight="bold")

    rng = np.random.default_rng(42)
    cat_colors = {cat["name"]: rng.random(3).tolist() for cat in categories}

    for ax, img_info in zip(axes, sorted_imgs):
        img_path = os.path.join(split_dir, img_info["file_name"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            ax.axis("off")
            continue

        ax.imshow(img)
        for ann in img_to_anns[img_info["id"]]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cat_name = cat_map[ann["category_id"]]
            color    = cat_colors[cat_name]
            rect = patches.Rectangle((x, y), w, h,
                                      linewidth=1.5, edgecolor=color,
                                      facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y - 3, cat_name, color=color,
                    fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.4, pad=1, linewidth=0))

        ax.set_title(f"{img_info['file_name'][:25]}\n"
                     f"{len(img_to_anns[img_info['id']])} bbox",
                     fontsize=8)
        ax.axis("off")

    for ax in axes[len(sorted_imgs):]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"0_samples_{split}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Thống kê COCO dataset")
    parser.add_argument("--root",    default=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco",
                        help="Thư mục gốc chứa train/valid/test")
    parser.add_argument("--splits",  default="train,valid,test",
                        help="Danh sách splits, cách nhau bởi dấu phẩy")
    parser.add_argument("--out",     default="dataset_stats_output",
                        help="Thư mục lưu biểu đồ")
    parser.add_argument("--samples", type=int, default=6,
                        help="Số ảnh mẫu cần visualize mỗi split")
    args = parser.parse_args()

    splits_to_check = [s.strip() for s in args.splits.split(",")]
    all_stats = {}

    for split in splits_to_check:
        coco, split_dir = load_split(args.root, split)
        if coco is None:
            print(f"⚠️  Không tìm thấy split '{split}' tại {split_dir}")
            continue

        stats = compute_stats(coco, split_dir)
        all_stats[split] = stats
        print_report(split, stats)

        # sample visualization
        visualize_samples(coco, split_dir, split, args.out, n=args.samples)

    if all_stats:
        plot_stats(all_stats, args.out)

    print(f"\n✅ Xong! Biểu đồ được lưu tại: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()