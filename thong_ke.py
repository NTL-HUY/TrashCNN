import json
from collections import Counter, defaultdict

# ── Cài thư viện nếu thiếu ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy"])
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

# ── Load dataset ─────────────────────────────────────────────────────────────
root = r"C:\Users\BAOHUY\Downloads\data\data\taco\annotations_filtered.json"
with open(root) as f:
    taco = json.load(f)

images       = taco["images"]
annotations  = taco["annotations"]
categories   = taco["categories"]
cat_id2target= taco["cat_id_to_target"]
orig_cats    = taco["original_categories"]

# ── 1. Tổng quan dataset ──────────────────────────────────────────────────────
print("=" * 60)
print("        TACO DATASET — TỔNG QUAN CẤU TRÚC")
print("=" * 60)
print(f"  📷  Số ảnh              : {len(images):>6,}")
print(f"  🏷️   Số annotations      : {len(annotations):>6,}")
print(f"  📦  Số categories        : {len(categories):>6,}")
print(f"  🗂️   Original categories  : {len(orig_cats):>6,}")
avg = len(annotations) / len(images) if images else 0
print(f"  📊  Annotation/ảnh (tb) : {avg:>6.2f}")
print("=" * 60)

# ── 2. Xây bảng id→name ───────────────────────────────────────────────────────
cat_id2name = {c["id"]: c["name"] for c in categories}

# ── 3. Thống kê annotations theo category ────────────────────────────────────
ann_per_cat  = Counter(a["category_id"] for a in annotations)
sorted_cats  = sorted(ann_per_cat.items(), key=lambda x: x[1], reverse=True)

print("\n📦 TOP 15 CATEGORIES (theo số annotations)")
print(f"  {'#':<4} {'Tên category':<35} {'Count':>7}  {'%':>6}")
print("  " + "-" * 57)
total_ann = len(annotations)
for rank, (cid, cnt) in enumerate(sorted_cats[:15], 1):
    name = cat_id2name.get(cid, f"id={cid}")
    print(f"  {rank:<4} {name:<35} {cnt:>7,}  {cnt/total_ann*100:>5.1f}%")

# ── 4. Annotations per image ─────────────────────────────────────────────────
ann_per_img = Counter(a["image_id"] for a in annotations)
counts = list(ann_per_img.values())
print(f"\n📷 ANNOTATIONS PER IMAGE")
print(f"  Min : {min(counts)}   Max : {max(counts)}   Mean : {np.mean(counts):.2f}   Median : {np.median(counts):.1f}")

# ── 5. Kiểm tra segmentation / bbox ──────────────────────────────────────────
has_seg   = sum(1 for a in annotations if a.get("segmentation"))
has_bbox  = sum(1 for a in annotations if a.get("bbox"))
print(f"\n🔍 LOẠI ANNOTATION")
print(f"  Có segmentation : {has_seg:,} / {total_ann:,}  ({has_seg/total_ann*100:.1f}%)")
print(f"  Có bbox         : {has_bbox:,} / {total_ann:,}  ({has_bbox/total_ann*100:.1f}%)")

# ── 6. Kích thước ảnh ─────────────────────────────────────────────────────────
widths  = [img["width"]  for img in images if "width"  in img]
heights = [img["height"] for img in images if "height" in img]
if widths:
    print(f"\n📐 KÍCH THƯỚC ẢNH")
    print(f"  Width  — min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}")
    print(f"  Height — min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")

# ── 7. Bbox area distribution ─────────────────────────────────────────────────
areas = []
for a in annotations:
    bb = a.get("bbox")
    if bb and len(bb) == 4:
        areas.append(bb[2] * bb[3])   # w * h

if areas:
    areas = np.array(areas)
    small  = (areas <  32**2).sum()
    medium = ((areas >= 32**2) & (areas < 96**2)).sum()
    large  = (areas >= 96**2).sum()
    print(f"\n📏 PHÂN BỐ KÍCH THƯỚC BBOX (COCO convention)")
    print(f"  Small  (<32²)   : {small:,}  ({small/len(areas)*100:.1f}%)")
    print(f"  Medium (32²–96²): {medium:,}  ({medium/len(areas)*100:.1f}%)")
    print(f"  Large  (>96²)   : {large:,}  ({large/len(areas)*100:.1f}%)")

print("\n" + "=" * 60)
print("  Đang vẽ biểu đồ… (đóng cửa sổ để tiếp tục)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN VẼ BIỂU ĐỒ
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12), facecolor="#0f0f0f")
fig.suptitle("TACO Dataset — Cấu trúc & Thống kê", fontsize=16,
             fontweight="bold", color="#f0f0f0", y=0.98)

# ── [1] Bar chart: top categories ────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
top_n  = 15
names  = [cat_id2name.get(cid, str(cid))[:22] for cid, _ in sorted_cats[:top_n]]
values = [cnt for _, cnt in sorted_cats[:top_n]]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, top_n))
bars   = ax1.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.7)
ax1.set_title("Top 15 Categories", color="#e0e0e0", fontsize=11, pad=8)
ax1.set_xlabel("Số annotations", color="#aaa", fontsize=9)
ax1.tick_params(colors="#ccc", labelsize=8)
ax1.spines[["top","right","left","bottom"]].set_visible(False)
ax1.xaxis.grid(True, color="#333", lw=0.5)
for bar, val in zip(bars, values[::-1]):
    ax1.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", color="#fff", fontsize=7)

# ── [2] Pie: annotation type ─────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
only_bbox = has_bbox - has_seg
pie_vals   = [has_seg, only_bbox, total_ann - has_bbox]
pie_labels = ["Seg + BBox", "BBox only", "Khác"]
pie_colors = ["#f97316", "#3b82f6", "#6b7280"]
wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "#ddd", "fontsize": 9},
    wedgeprops={"linewidth": 0.5, "edgecolor": "#0f0f0f"}
)
for at in autotexts:
    at.set_color("white"); at.set_fontsize(9)
ax2.set_title("Loại Annotation", color="#e0e0e0", fontsize=11, pad=8)

# ── [3] Histogram: annotations per image ─────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
ax3.hist(counts, bins=40, color="#06b6d4", edgecolor="#0f0f0f", linewidth=0.3)
ax3.axvline(np.mean(counts), color="#f97316", lw=1.5, linestyle="--", label=f"Mean={np.mean(counts):.1f}")
ax3.axvline(np.median(counts), color="#a3e635", lw=1.5, linestyle=":",  label=f"Median={np.median(counts):.1f}")
ax3.set_title("Phân bố Annotations / Ảnh", color="#e0e0e0", fontsize=11, pad=8)
ax3.set_xlabel("Số annotations", color="#aaa", fontsize=9)
ax3.set_ylabel("Số ảnh", color="#aaa", fontsize=9)
ax3.tick_params(colors="#ccc", labelsize=8)
ax3.legend(fontsize=8, labelcolor="#ddd", facecolor="#1a1a1a", edgecolor="#444")
ax3.spines[["top","right"]].set_visible(False)
ax3.yaxis.grid(True, color="#333", lw=0.5)

# ── [4] Histogram: bbox area ─────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
if len(areas):
    log_areas = np.log10(areas + 1)
    ax4.hist(log_areas, bins=50, color="#8b5cf6", edgecolor="#0f0f0f", linewidth=0.3)
    ax4.axvline(np.log10(32**2), color="#f97316", lw=1.5, ls="--", label="Small/Med (32²)")
    ax4.axvline(np.log10(96**2), color="#a3e635", lw=1.5, ls="--", label="Med/Large (96²)")
    ax4.set_title("Phân bố Bbox Area (log₁₀)", color="#e0e0e0", fontsize=11, pad=8)
    ax4.set_xlabel("log₁₀(area)", color="#aaa", fontsize=9)
    ax4.set_ylabel("Số annotations", color="#aaa", fontsize=9)
    ax4.tick_params(colors="#ccc", labelsize=8)
    ax4.legend(fontsize=8, labelcolor="#ddd", facecolor="#1a1a1a", edgecolor="#444")
    ax4.spines[["top","right"]].set_visible(False)
    ax4.yaxis.grid(True, color="#333", lw=0.5)

# ── [5] Stacked bar: small/medium/large per top-10 cat ───────────────────────
ax5 = fig.add_subplot(2, 3, 5)
top10_ids = [cid for cid, _ in sorted_cats[:10]]
size_data  = defaultdict(lambda: [0, 0, 0])
for a in annotations:
    cid = a.get("category_id")
    if cid not in top10_ids: continue
    bb = a.get("bbox")
    if bb and len(bb) == 4:
        area = bb[2] * bb[3]
        if   area <  32**2: size_data[cid][0] += 1
        elif area <  96**2: size_data[cid][1] += 1
        else:               size_data[cid][2] += 1

cat_labels = [cat_id2name.get(cid, str(cid))[:18] for cid in top10_ids]
s = np.array([size_data[cid][0] for cid in top10_ids])
m = np.array([size_data[cid][1] for cid in top10_ids])
l = np.array([size_data[cid][2] for cid in top10_ids])
x = np.arange(len(top10_ids))
w = 0.6
ax5.bar(x, s, w, label="Small",  color="#f97316")
ax5.bar(x, m, w, bottom=s,       label="Medium", color="#3b82f6")
ax5.bar(x, l, w, bottom=s+m,     label="Large",  color="#a3e635")
ax5.set_xticks(x)
ax5.set_xticklabels(cat_labels, rotation=35, ha="right", fontsize=7, color="#ccc")
ax5.set_title("Bbox Size × Top-10 Category", color="#e0e0e0", fontsize=11, pad=8)
ax5.set_ylabel("Annotations", color="#aaa", fontsize=9)
ax5.tick_params(colors="#ccc", labelsize=8)
ax5.legend(fontsize=8, labelcolor="#ddd", facecolor="#1a1a1a", edgecolor="#444")
ax5.spines[["top","right"]].set_visible(False)
ax5.yaxis.grid(True, color="#333", lw=0.5, zorder=0)

# ── [6] Tóm tắt dạng text ────────────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
summary = [
    ("Tổng số ảnh",          f"{len(images):,}"),
    ("Tổng annotations",     f"{total_ann:,}"),
    ("Categories",           f"{len(categories)}"),
    ("Original categories",  f"{len(orig_cats)}"),
    ("Ann/ảnh (mean)",       f"{avg:.2f}"),
    ("Ann/ảnh (max)",        f"{max(counts)}"),
    ("Has segmentation",     f"{has_seg:,}  ({has_seg/total_ann*100:.1f}%)"),
    ("Has bbox",             f"{has_bbox:,}  ({has_bbox/total_ann*100:.1f}%)"),
]
if len(areas):
    summary += [
        ("Bbox: Small",  f"{small:,}  ({small/len(areas)*100:.1f}%)"),
        ("Bbox: Medium", f"{medium:,}  ({medium/len(areas)*100:.1f}%)"),
        ("Bbox: Large",  f"{large:,}  ({large/len(areas)*100:.1f}%)"),
    ]
for i, (k, v) in enumerate(summary):
    y_pos = 0.95 - i * 0.085
    ax6.text(0.02, y_pos, f"{k}:", transform=ax6.transAxes,
             fontsize=9, color="#888", va="top")
    ax6.text(0.55, y_pos, v, transform=ax6.transAxes,
             fontsize=9, color="#f0f0f0", va="top", fontweight="bold")
ax6.set_title("📋 Tóm tắt", color="#e0e0e0", fontsize=11, pad=8)
rect = mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax6.transAxes,
    boxstyle="round,pad=0.02", linewidth=1, edgecolor="#333", facecolor="#1a1a1a", zorder=-1)
ax6.add_patch(rect)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()