"""
visualize_processed.py
======================
Thống kê và visualize dataset SAU KHI tiền xử lý.
Đọc từ Config.OUTPUT_DIR (output của preprocess.py).

Usage:
    python visualize_processed.py
"""

import json
import math
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Import config của project ──────────────────────────────────────────────
# Nếu chạy độc lập thì sửa 3 dòng này
try:
    from TrashDetect.config import Config
    PROCESSED_DIR = Config.OUTPUT_DIR   # thư mục output của preprocess.py
    SPLITS        = Config.SPLITS       # ["train", "valid", "test"]
    REPORT_DIR    = getattr(Config, "REPORT_DIR", Config.OUTPUT_DIR)
except ImportError:
    PROCESSED_DIR = "./data/processed"
    SPLITS        = ["train", "valid", "test"]
    REPORT_DIR    = "./data/processed"

# ── Palette ────────────────────────────────────────────────────────────────
PAL = [
    "#FF6B6B","#FFD93D","#6BCB77","#4D96FF","#C77DFF",
    "#F4A261","#2EC4B6","#E76F51","#A8DADC","#457B9D",
]
SPLIT_COLORS = {"train": "#4D96FF", "valid": "#6BCB77", "test": "#FFD93D"}

# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────

def load_coco(base: str, split: str) -> dict | None:
    p = os.path.join(base, split, "_annotations.processed.coco.json")
    if not os.path.exists(p):
        print(f"  [SKIP] {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────────────
# COMPUTE STATS
# ──────────────────────────────────────────────────────────────────────────

def compute_stats(coco: dict, split: str) -> dict:
    images      = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories  = coco.get("categories", [])
    id2name     = {c["id"]: c["name"] for c in categories}

    ann_per_img = Counter(a["image_id"] for a in annotations)
    counts      = list(ann_per_img.values()) or [0]
    cat_counts  = Counter(id2name.get(a["category_id"], "?") for a in annotations)

    bbox_areas, bbox_ws, bbox_hs = [], [], []
    for a in annotations:
        bb = a.get("bbox")
        if bb and len(bb) == 4:
            _, _, w, h = bb
            bbox_ws.append(w); bbox_hs.append(h); bbox_areas.append(w * h)

    img_ws = [i.get("width",  0) for i in images]
    img_hs = [i.get("height", 0) for i in images]

    return dict(
        split        = split,
        categories   = categories,
        n_images     = len(images),
        n_anns       = len(annotations),
        n_cats       = len(categories),
        cat_counts   = cat_counts,
        ann_per_img  = counts,
        ann_mean     = float(np.mean(counts)),
        ann_median   = float(np.median(counts)),
        ann_max      = int(np.max(counts)),
        bbox_areas   = bbox_areas,
        bbox_ws      = bbox_ws,
        bbox_hs      = bbox_hs,
        img_ws       = img_ws,
        img_hs       = img_hs,
    )

# ──────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ──────────────────────────────────────────────────────────────────────────

def print_report(stats_list: list[dict]):
    print("\n" + "═"*70)
    print("  POST-PROCESSING DATASET REPORT")
    print("═"*70)
    for s in stats_list:
        total = s["n_anns"] or 1
        print(f"\n  split={s['split'].upper():<8}  "
              f"images={s['n_images']:,}  anns={s['n_anns']:,}  "
              f"categories={s['n_cats']}")
        print(f"  Ann/image: mean={s['ann_mean']:.2f}  "
              f"median={s['ann_median']:.1f}  max={s['ann_max']}")
        print(f"  {'Class':<16} {'Count':>6}  {'%':>6}  Bar")
        print(f"  {'─'*50}")
        for name, cnt in sorted(s["cat_counts"].items(), key=lambda x: -x[1]):
            bar = "█" * int(30 * cnt / total)
            print(f"  {name:<16} {cnt:>6,}  {100*cnt/total:>5.1f}%  {bar}")
    print()

# ──────────────────────────────────────────────────────────────────────────
# AXIS STYLE
# ──────────────────────────────────────────────────────────────────────────

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#13132b")
    ax.tick_params(colors="#bbbbdd", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a2a55")
    if title:  ax.set_title(title,  color="#e0e0ff", fontsize=9,  fontweight="bold", pad=7)
    if xlabel: ax.set_xlabel(xlabel, color="#9999cc", fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color="#9999cc", fontsize=8)

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1 – per-split class distribution (grouped bars)
# ──────────────────────────────────────────────────────────────────────────

def fig_class_distribution(stats_list: list[dict], out: str):
    all_cats = sorted({name for s in stats_list for name in s["cat_counts"]})
    splits   = [s["split"] for s in stats_list]
    n_cats   = len(all_cats)
    n_splits = len(splits)

    x     = np.arange(n_cats)
    width = 0.8 / n_splits

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.4), 5), facecolor="#0d0d1a")
    for i, s in enumerate(stats_list):
        vals   = [s["cat_counts"].get(c, 0) for c in all_cats]
        offset = (i - n_splits/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=s["split"],
                        color=SPLIT_COLORS.get(s["split"], PAL[i]),
                        alpha=0.88)
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.01,
                        f"{v:,}", ha="center", va="bottom",
                        color="#ffffff", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, color="#ddddff", fontsize=9)
    ax.legend(fontsize=9, labelcolor="white", facecolor="#1e1e40",
              edgecolor="#444488")
    _style(ax, "Class Distribution per Split  (after preprocessing)",
           "", "Annotation count")
    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 2 – class balance pie per split
# ──────────────────────────────────────────────────────────────────────────

def fig_class_pie(stats_list: list[dict], out: str):
    n = len(stats_list)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5), facecolor="#0d0d1a")
    if n == 1: axes = [axes]

    for ax, s in zip(axes, stats_list):
        items  = sorted(s["cat_counts"].items(), key=lambda x: -x[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = [PAL[i % len(PAL)] for i in range(len(labels))]

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct="%1.1f%%", pctdistance=0.80,
            textprops={"color": "#ddddff", "fontsize": 8},
            wedgeprops={"linewidth": 0.6, "edgecolor": "#0d0d1a"},
            startangle=140,
        )
        for at in autotexts:
            at.set_fontsize(7); at.set_color("#ffffff")
        ax.set_facecolor("#13132b")
        _style(ax, f"{s['split'].upper()}  ({s['n_anns']:,} anns)")

    fig.suptitle("Class Balance after Preprocessing",
                 color="#e8e8ff", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3 – annotations/image histogram (all splits overlay)
# ──────────────────────────────────────────────────────────────────────────

def fig_ann_per_image(stats_list: list[dict], out: str):
    fig, axes = plt.subplots(1, len(stats_list),
                             figsize=(5 * len(stats_list), 4),
                             facecolor="#0d0d1a")
    if len(stats_list) == 1: axes = [axes]

    for ax, s in zip(axes, stats_list):
        data  = s["ann_per_img"]
        n_bins = min(30, max(5, int(math.sqrt(len(data)))))
        color  = SPLIT_COLORS.get(s["split"], PAL[0])
        ax.hist(data, bins=n_bins, color=color, alpha=0.85, edgecolor="#111122")
        ax.axvline(s["ann_mean"],   color="#FFD93D", lw=1.5, ls="--",
                   label=f"mean {s['ann_mean']:.1f}")
        ax.axvline(s["ann_median"], color="#6BCB77", lw=1.5, ls=":",
                   label=f"median {s['ann_median']:.1f}")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#1e1e40")
        _style(ax, f"{s['split'].upper()}  – Ann per Image",
               "Annotations", "Images")

    fig.suptitle("Annotations per Image Distribution",
                 color="#e8e8ff", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 4 – bbox area distribution (log)
# ──────────────────────────────────────────────────────────────────────────

def fig_bbox_area(stats_list: list[dict], out: str):
    fig, axes = plt.subplots(1, len(stats_list),
                             figsize=(5 * len(stats_list), 4),
                             facecolor="#0d0d1a")
    if len(stats_list) == 1: axes = [axes]

    for ax, s in zip(axes, stats_list):
        areas = s["bbox_areas"]
        if not areas:
            ax.text(0.5, 0.5, "No data", ha="center", color="white",
                    transform=ax.transAxes); continue
        color = SPLIT_COLORS.get(s["split"], PAL[4])
        log_a = np.log1p(areas)
        ax.hist(log_a, bins=40, color=color, alpha=0.85, edgecolor="#111122")
        mean_log = np.mean(log_a)
        ax.axvline(mean_log, color="#FFD93D", lw=1.5, ls="--",
                   label=f"mean area ≈ {np.expm1(mean_log):.0f} px²")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#1e1e40")
        _style(ax, f"{s['split'].upper()}  – BBox Area (log)",
               "log(1 + area)", "Count")

    fig.suptitle("BBox Area Distribution (log scale)",
                 color="#e8e8ff", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 5 – summary table image
# ──────────────────────────────────────────────────────────────────────────

def fig_summary_table(stats_list: list[dict], out: str):
    all_cats = sorted({n for s in stats_list for n in s["cat_counts"]})
    col_hdrs = ["split", "images", "anns", "ann/img\nmean"] + all_cats
    rows = []
    for s in stats_list:
        row = [
            s["split"],
            f"{s['n_images']:,}",
            f"{s['n_anns']:,}",
            f"{s['ann_mean']:.2f}",
        ]
        for c in all_cats:
            cnt  = s["cat_counts"].get(c, 0)
            pct  = 100 * cnt / (s["n_anns"] or 1)
            row.append(f"{cnt}\n({pct:.1f}%)")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(max(10, len(col_hdrs) * 1.6), 2.2 + len(rows)*0.5),
                           facecolor="#0d0d1a")
    ax.axis("off")

    tbl = ax.table(
        cellText  = rows,
        colLabels = col_hdrs,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2a2a6a")
            cell.set_text_props(color="#ffffff", fontweight="bold")
        else:
            split_name = rows[r-1][0]
            base_color = SPLIT_COLORS.get(split_name, "#1e1e40")
            cell.set_facecolor(base_color + "33")   # semi-transparent
            cell.set_text_props(color="#e0e0ff")
        cell.set_edgecolor("#2a2a55")

    fig.suptitle("Dataset Summary  (after preprocessing)",
                 color="#e8e8ff", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")

# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  Reading processed data from: {PROCESSED_DIR}")
    os.makedirs(REPORT_DIR, exist_ok=True)

    stats_list = []
    for split in SPLITS:
        coco = load_coco(PROCESSED_DIR, split)
        if coco is None:
            continue
        stats_list.append(compute_stats(coco, split))

    if not stats_list:
        print("\n[ERROR] Không đọc được file nào từ PROCESSED_DIR.")
        print(f"        Kiểm tra lại: {PROCESSED_DIR}")
        print("        Cần chứa các thư mục train/valid/test với _annotations.coco.json")
        return

    # ── In báo cáo text ──────────────────────────────────────────────────
    print_report(stats_list)

    # ── Xuất ảnh ────────────────────────────────────────────────────────
    print("  Saving figures …")
    fig_class_distribution(stats_list, os.path.join(REPORT_DIR, "post_class_dist.png"))
    fig_class_pie         (stats_list, os.path.join(REPORT_DIR, "post_class_pie.png"))
    fig_ann_per_image     (stats_list, os.path.join(REPORT_DIR, "post_ann_per_img.png"))
    fig_bbox_area         (stats_list, os.path.join(REPORT_DIR, "post_bbox_area.png"))
    fig_summary_table     (stats_list, os.path.join(REPORT_DIR, "post_summary_table.png"))

    print(f"\n  ✅  All figures saved to → {REPORT_DIR}")


if __name__ == "__main__":
    main()