import json
import os
from collections import Counter

# ── Config ────────────────────────────────────────────────
ROOT = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"
SPLITS = ["train", "valid", "test"]

SEP  = "=" * 60
SEP2 = "-" * 60

# ── Helper ────────────────────────────────────────────────
def load_split(split):
    path = os.path.join(ROOT, split, "_annotations.coco.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def print_split_stats(name, data):
    if data is None:
        print(f"\n[!] Không tìm thấy split: {name}")
        return

    images      = data.get("images", [])
    annotations = data.get("annotations", [])
    categories  = data.get("categories", [])

    id_to_name  = {c["id"]: c["name"] for c in categories}
    ann_per_img = Counter(a["image_id"] for a in annotations)
    ann_per_cls = Counter(a["category_id"] for a in annotations)

    imgs_with_ann    = len(ann_per_img)
    imgs_without_ann = len(images) - imgs_with_ann
    avg_ann          = len(annotations) / len(images) if images else 0
    max_ann          = max(ann_per_img.values(), default=0)
    min_ann          = min(ann_per_img.values(), default=0)

    # ── Header ──
    print(f"\n{SEP}")
    print(f"  SPLIT: {name.upper()}")
    print(SEP)

    # ── Tổng quan ──
    print(f"  {'Tổng ảnh':<30} {len(images):>6}")
    print(f"  {'Ảnh có annotation':<30} {imgs_with_ann:>6}")
    print(f"  {'Ảnh không có annotation':<30} {imgs_without_ann:>6}")
    print(f"  {'Tổng annotation':<30} {len(annotations):>6}")
    print(f"  {'Số class':<30} {len(categories):>6}")
    print(f"  {'TB annotation/ảnh':<30} {avg_ann:>6.2f}")
    print(f"  {'Max annotation/ảnh':<30} {max_ann:>6}")
    print(f"  {'Min annotation/ảnh (có ann)':<30} {min_ann:>6}")

    # ── Annotation theo class ──
    print(f"\n  {'Class distribution':}")
    print(f"  {SEP2}")
    print(f"  {'ID':<5} {'Tên class':<20} {'Số ann':>8}  {'%':>7}")
    print(f"  {SEP2}")
    total = len(annotations)
    for cid, cnt in sorted(ann_per_cls.items()):
        pct = cnt / total * 100 if total else 0
        print(f"  [{cid:<3}] {id_to_name.get(cid, '?'):<20} {cnt:>8}  {pct:>6.1f}%")
    print(f"  {SEP2}")
    print(f"  {'TỔNG':<25} {total:>8}  100.0%")


# ── Load all splits ───────────────────────────────────────
all_data = {s: load_split(s) for s in SPLITS}

# ── In từng split ─────────────────────────────────────────
for split in SPLITS:
    print_split_stats(split, all_data[split])

# ── So sánh tổng hợp ─────────────────────────────────────
print(f"\n{SEP}")
print(f"  TỔNG HỢP TẤT CẢ SPLITS")
print(SEP)
print(f"  {'Split':<10} {'Ảnh':>8} {'Annotation':>12} {'TB ann/ảnh':>12}")
print(f"  {SEP2}")

total_imgs = total_anns = 0
for split in SPLITS:
    d = all_data[split]
    if d is None:
        print(f"  {split:<10} {'N/A':>8} {'N/A':>12} {'N/A':>12}")
        continue
    n_img = len(d["images"])
    n_ann = len(d["annotations"])
    avg   = n_ann / n_img if n_img else 0
    total_imgs += n_img
    total_anns += n_ann
    print(f"  {split:<10} {n_img:>8} {n_ann:>12} {avg:>12.2f}")

print(f"  {SEP2}")
print(f"  {'TOTAL':<10} {total_imgs:>8} {total_anns:>12}")

# ── Tỷ lệ split ──────────────────────────────────────────
print(f"\n  Tỷ lệ split (theo ảnh):")
for split in SPLITS:
    d = all_data[split]
    if d is None:
        continue
    n_img = len(d["images"])
    pct   = n_img / total_imgs * 100 if total_imgs else 0
    print(f"    {split:<8}: {n_img:>5} ảnh  ({pct:.1f}%)")

print(f"\n{SEP}\n")