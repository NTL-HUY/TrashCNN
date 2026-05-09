


"""
Dataset Statistics & Visualization
Thống kê và visualize TACO + Glass COCO datasets
Usage: python dataset_visualize.py
"""

import json
import os
import math
from collections import Counter

import matplotlib

from TrashDetect.config import Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────
# CONFIG – chỉnh đường dẫn ở đây
# ─────────────────────────────────────────────

Config = Config()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_coco(base_path: str, split: str) -> dict | None:
    path = os.path.join(base_path, split, "_annotations.processed.coco.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        print(f"\n  Loading: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"    images      : {len(data.get('images', [])):,}")
        print(f"    annotations : {len(data.get('annotations', [])):,}")
        print(f"    categories  : {len(data.get('categories', []))}")

        return data


def compute_stats(data: dict, name: str, split: str) -> dict:
    """Tính các thống kê cơ bản từ một file COCO."""
    images      = data.get("images", [])
    annotations = data.get("annotations", [])
    categories  = data.get("categories", [])

    cat_id2name = {c["id"]: c["name"] for c in categories}

    # Số annotation / ảnh
    ann_per_image = Counter(a["image_id"] for a in annotations)
    counts = list(ann_per_image.values()) if ann_per_image else [0]

    # Phân phối category
    cat_counts = Counter(cat_id2name.get(a["category_id"], "unknown")
                         for a in annotations)

    # Kích thước bbox
    bbox_areas = []
    bbox_ws, bbox_hs = [], []
    for a in annotations:
        bb = a.get("bbox")
        if bb and len(bb) == 4:
            _, _, w, h = bb
            bbox_areas.append(w * h)
            bbox_ws.append(w)
            bbox_hs.append(h)

    # Kích thước ảnh
    img_ws = [img.get("width",  0) for img in images]
    img_hs = [img.get("height", 0) for img in images]

    return {
        "name":          name,
        "split":         split,
        "n_images":      len(images),
        "n_annotations": len(annotations),
        "n_categories":  len(categories),
        "categories":    categories,
        "cat_counts":    cat_counts,
        "ann_per_image": counts,
        "ann_mean":      float(np.mean(counts))   if counts else 0,
        "ann_median":    float(np.median(counts)) if counts else 0,
        "ann_max":       int(np.max(counts))      if counts else 0,
        "bbox_areas":    bbox_areas,
        "bbox_ws":       bbox_ws,
        "bbox_hs":       bbox_hs,
        "img_ws":        img_ws,
        "img_hs":        img_hs,
    }


def print_stats(s: dict):

    print(f"\n{'═'*65}")
    print(f"  DATASET = {s['name']}   |   SPLIT = {s['split'].upper()}")
    print(f"{'═'*65}")

    print(f"  Images           : {s['n_images']:,}")
    print(f"  Annotations      : {s['n_annotations']:,}")
    print(f"  Categories       : {s['n_categories']}")

    print()

    print(f"  Annotation / Image")
    print(f"    Mean           : {s['ann_mean']:.2f}")
    print(f"    Median         : {s['ann_median']:.2f}")
    print(f"    Max            : {s['ann_max']}")

    print()

    if s["bbox_areas"]:
        print(f"  Bounding Boxes")
        print(f"    Mean Area      : {np.mean(s['bbox_areas']):.1f}")
        print(f"    Min Area       : {np.min(s['bbox_areas']):.1f}")
        print(f"    Max Area       : {np.max(s['bbox_areas']):.1f}")

        print(f"    Mean Width     : {np.mean(s['bbox_ws']):.1f}")
        print(f"    Mean Height    : {np.mean(s['bbox_hs']):.1f}")

    print()

    if s["img_ws"]:
        print(f"  Image Sizes")
        print(f"    Mean Width     : {np.mean(s['img_ws']):.1f}")
        print(f"    Mean Height    : {np.mean(s['img_hs']):.1f}")

        print(f"    Max Width      : {np.max(s['img_ws'])}")
        print(f"    Max Height     : {np.max(s['img_hs'])}")

    print()

    print(f"  Category Distribution")
    print(f"  {'─'*55}")

    total = s['n_annotations'] or 1

    for cat_name, cnt in sorted(
            s['cat_counts'].items(),
            key=lambda x: -x[1]
    ):

        pct = 100 * cnt / total

        bar = "█" * int(pct / 2)

        print(
            f"  {cat_name:<15} "
            f"{cnt:>6,}   "
            f"{pct:>5.1f}%   "
            f"{bar}"
        )

    print(f"{'═'*65}")


# ─────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────

PALETTE = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#C77DFF",
    "#F4A261", "#2EC4B6", "#E76F51", "#A8DADC", "#457B9D",
    "#E63946", "#06D6A0", "#118AB2", "#FFB703", "#FB8500",
]

def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#cccccc", labelsize=8)
    ax.spines[:].set_color("#333355")
    if title:   ax.set_title(title,   color="#e0e0ff", fontsize=10, pad=8, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, color="#aaaacc", fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color="#aaaacc", fontsize=8)


def plot_category_bar(ax, stat: dict, top_n=15):
    items = sorted(stat["cat_counts"].items(), key=lambda x: -x[1])[:top_n]
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white")
        return
    names, counts = zip(*items)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.barh(range(len(names)), counts, color=colors, height=0.7, alpha=0.92)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7, color="#ddddff")
    ax.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f"{cnt:,}", va="center", ha="left", color="#ffffff", fontsize=7)
    _ax_style(ax, f"{stat['name']} – Category Distribution (top {top_n})", "Count")


def plot_ann_hist(ax, stat: dict):
    data = stat["ann_per_image"]
    if not data:
        return
    n_bins = min(30, max(5, int(math.sqrt(len(data)))))
    ax.hist(data, bins=n_bins, color=PALETTE[3], alpha=0.85, edgecolor="#111122")
    ax.axvline(stat["ann_mean"],   color="#FFD93D", lw=1.5, ls="--", label=f"mean={stat['ann_mean']:.1f}")
    ax.axvline(stat["ann_median"], color="#6BCB77", lw=1.5, ls=":",  label=f"median={stat['ann_median']:.1f}")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#22224a")
    _ax_style(ax, f"{stat['name']} – Annotations per Image", "Count", "Frequency")


def plot_bbox_area(ax, stat: dict):
    areas = stat["bbox_areas"]
    if not areas:
        return
    log_areas = np.log1p(areas)
    ax.hist(log_areas, bins=40, color=PALETTE[4], alpha=0.85, edgecolor="#111122")
    _ax_style(ax, f"{stat['name']} – BBox Area Distribution (log scale)", "log(1 + area)", "Frequency")


def plot_img_size_scatter(ax, stat: dict):
    ws, hs = stat["img_ws"], stat["img_hs"]
    if not ws:
        return
    ax.scatter(ws, hs, alpha=0.3, s=12, color=PALETTE[1], edgecolors="none")
    _ax_style(ax, f"{stat['name']} – Image Sizes", "Width (px)", "Height (px)")


def plot_bbox_wh(ax, stat: dict):
    ws, hs = stat["bbox_ws"], stat["bbox_hs"]
    if not ws:
        return
    ax.scatter(ws, hs, alpha=0.15, s=8, color=PALETTE[0], edgecolors="none")
    _ax_style(ax, f"{stat['name']} – BBox W vs H", "Width", "Height")


def plot_cat_pie(ax, stat: dict, top_n=8):
    items = sorted(stat["cat_counts"].items(), key=lambda x: -x[1])
    if not items:
        return
    top    = items[:top_n]
    others = sum(v for _, v in items[top_n:])
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    if others:
        labels.append("others")
        values.append(others)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", pctdistance=0.82,
        textprops={"color": "#ddddff", "fontsize": 7},
        wedgeprops={"linewidth": 0.5, "edgecolor": "#1a1a2e"},
        startangle=140
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(6.5)
    ax.set_facecolor("#1a1a2e")
    _ax_style(ax, f"{stat['name']} – Category Share")


def plot_split_comparison(ax, all_stats: list[dict]):
    """Bar chart so sánh số ảnh + annotation giữa các split."""
    datasets = sorted({s["name"] for s in all_stats})
    splits   = sorted({s["split"] for s in all_stats})
    x        = np.arange(len(datasets))
    width    = 0.8 / len(splits)

    for i, split in enumerate(splits):
        vals = []
        for ds in datasets:
            match = next((s for s in all_stats if s["name"]==ds and s["split"]==split), None)
            vals.append(match["n_images"] if match else 0)
        bars = ax.bar(x + i*width - 0.4 + width/2, vals, width*0.9,
                      label=split, color=PALETTE[i*3 % len(PALETTE)], alpha=0.88)
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", color="#ffffff", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, color="#ddddff", fontsize=9)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#22224a")
    _ax_style(ax, "Images per Split", "", "Count")


# ─────────────────────────────────────────────
# MAIN FIGURE BUILDER
# ─────────────────────────────────────────────

def build_dashboard(all_stats: list[dict], output_path: str):
    n = len(all_stats)
    cols = 3
    rows_per = 4   # rows per dataset: bar, hist, bbox_area, pie
    extra_rows = 1 # split comparison
    total_rows = n * rows_per + extra_rows

    fig = plt.figure(figsize=(cols * 6, total_rows * 3.2), facecolor="#0d0d1a")
    fig.suptitle("🗑  Waste Dataset Statistics Dashboard",
                 color="#e8e8ff", fontsize=18, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(total_rows, cols, figure=fig, hspace=0.55, wspace=0.38)

    for idx, s in enumerate(all_stats):
        row_offset = idx * rows_per

        # Row 0: category bar (spans 2 cols) + pie (1 col)
        ax_bar = fig.add_subplot(gs[row_offset,     0:2])
        ax_pie = fig.add_subplot(gs[row_offset,     2])
        plot_category_bar(ax_bar, s)
        plot_cat_pie(ax_pie, s)

        # Row 1: ann hist + bbox area + bbox wh scatter
        ax_h1 = fig.add_subplot(gs[row_offset + 1, 0])
        ax_h2 = fig.add_subplot(gs[row_offset + 1, 1])
        ax_h3 = fig.add_subplot(gs[row_offset + 1, 2])
        plot_ann_hist(ax_h1, s)
        plot_bbox_area(ax_h2, s)
        plot_bbox_wh(ax_h3, s)

        # Row 2: image size scatter + summary text box
        ax_sc = fig.add_subplot(gs[row_offset + 2, 0])
        plot_img_size_scatter(ax_sc, s)

        ax_txt = fig.add_subplot(gs[row_offset + 2, 1:3])
        ax_txt.set_facecolor("#12122a")
        ax_txt.axis("off")
        summary = (
            f"Dataset : {s['name'].upper()}   |   Split : {s['split']}\n"
            f"Images  : {s['n_images']:,}\n"
            f"Annotations : {s['n_annotations']:,}\n"
            f"Categories  : {s['n_categories']}\n"
            f"Ann / Image : mean {s['ann_mean']:.2f} · median {s['ann_median']:.1f} · max {s['ann_max']}\n"
            f"BBox area   : mean {np.mean(s['bbox_areas']):.0f} px²"
            if s['bbox_areas'] else ""
        )
        ax_txt.text(0.05, 0.5, summary, transform=ax_txt.transAxes,
                    color="#c8c8ff", fontsize=9.5, va="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e1e40",
                              edgecolor="#4444aa", linewidth=1.2))

        # Row 3: empty / spacing row – reuse for dataset label banner
        ax_banner = fig.add_subplot(gs[row_offset + 3, :])
        ax_banner.set_facecolor("#10102a")
        ax_banner.axis("off")

    # Last row: split comparison
    ax_cmp = fig.add_subplot(gs[n * rows_per, :])
    plot_split_comparison(ax_cmp, all_stats)

    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✅ Dashboard saved → {output_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    all_stats = []
    print("\n" + "═" * 70)
    print("  DATASET STATISTICS DASHBOARD")
    print("═" * 70)
    for dataset_name, base_path in [("TACO", Config.DATA_TACO_PATH)]:
        for split in Config.SPLITS:
            print(f"\n[PROCESSING] dataset={dataset_name} split={split}")
            data = load_coco(base_path, split)
            if data is None:
                continue
            s = compute_stats(data, dataset_name, split)
            print_stats(s)
            all_stats.append(s)

    if not all_stats:
        print("\n[ERROR] Không tìm thấy file nào. Kiểm tra lại Config paths.")
        return

    out = os.path.join(Config.OUTPUT_DIR, "dashboard.png")
    print("\n  📊 Building dashboard …")
    build_dashboard(all_stats, out)
    print(f"\n  Saving dashboard to:")
    print(f"    {out}")

if __name__ == "__main__":
    main()