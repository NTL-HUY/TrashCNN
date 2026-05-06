import json
import os

root = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco\train"
with open(os.path.join(root, "_annotations.coco.json")) as f:
    taco = json.load(f)

# ── 1. Tổng quan ──────────────────────────────────────────
print("=" * 50)
print("TỔNG QUAN")
print("=" * 50)
for key in taco:
    val = taco[key]
    if isinstance(val, list):
        print(f"  {key}: {len(val)} items")
    else:
        print(f"  {key}: {val}")

# ── 2. Categories ──────────────────────────────────────────
print("\n" + "=" * 50)
print("CATEGORIES")
print("=" * 50)
for cat in taco["categories"]:
    print(f"  {cat}")

# ── 3. Một image mẫu ──────────────────────────────────────
print("\n" + "=" * 50)
print("IMAGE MẪU (index 0)")
print("=" * 50)
print(f"  {taco['images'][0]}")

# ── 4. Một annotation mẫu ─────────────────────────────────
print("\n" + "=" * 50)
print("ANNOTATION MẪU (index 0)")
print("=" * 50)
print(f"  {taco['annotations'][0]}")

# ── 5. Số annotation theo từng class ──────────────────────
print("\n" + "=" * 50)
print("SỐ LƯỢNG ANNOTATION THEO CLASS")
print("=" * 50)
id_to_name = {cat["id"]: cat["name"] for cat in taco["categories"]}
count = {}
for ann in taco["annotations"]:
    cid = ann["category_id"]
    count[cid] = count.get(cid, 0) + 1

for cid, cnt in sorted(count.items()):
    print(f"  [{cid}] {id_to_name.get(cid, '?'):<20} {cnt:>5} annotations")

import json
import os
from collections import Counter

root = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco\train"
with open(os.path.join(root, "_annotations.coco.json")) as f:
    taco = json.load(f)

# ── 1. Xóa trash và other ─────────────────────────────────
REMOVE_IDS = {0, 4}

taco["categories"] = [c for c in taco["categories"] if c["id"] not in REMOVE_IDS]
taco["annotations"] = [a for a in taco["annotations"] if a["category_id"] not in REMOVE_IDS]

# ── 2. Reindex category ID ────────────────────────────────
old_to_new = {}
for i, cat in enumerate(taco["categories"]):
    old_to_new[cat["id"]] = i
    cat["id"] = i

for ann in taco["annotations"]:
    ann["category_id"] = old_to_new[ann["category_id"]]

# ── 3. Kiểm tra kết quả ───────────────────────────────────
print("Categories còn lại:")
for cat in taco["categories"]:
    print(f"  [{cat['id']}] {cat['name']}")

count = Counter(a["category_id"] for a in taco["annotations"])
print("\nSố annotation theo class:")
id_to_name = {c["id"]: c["name"] for c in taco["categories"]}
for cid, cnt in sorted(count.items()):
    print(f"  [{cid}] {id_to_name[cid]:<15} {cnt:>5}")
print(f"\nTổng annotations: {len(taco['annotations'])}")

import random
PLASTIC_ID = next(c["id"] for c in taco["categories"] if c["name"] == "plastic")
TARGET_PLASTIC = 500

plastic_anns = [a for a in taco["annotations"] if a["category_id"] == PLASTIC_ID]
other_anns   = [a for a in taco["annotations"] if a["category_id"] != PLASTIC_ID]

random.seed(42)
plastic_anns_sampled = random.sample(plastic_anns, TARGET_PLASTIC)

taco["annotations"] = other_anns + plastic_anns_sampled

# Xóa ảnh không còn annotation nào
used_image_ids = {a["image_id"] for a in taco["annotations"]}
taco["images"] = [img for img in taco["images"] if img["id"] in used_image_ids]

# ── Kiểm tra ──────────────────────────────────────────────
count = Counter(a["category_id"] for a in taco["annotations"])
id_to_name = {c["id"]: c["name"] for c in taco["categories"]}
print("Sau undersample:")
for cid, cnt in sorted(count.items()):
    print(f"  [{cid}] {id_to_name[cid]:<15} {cnt:>5}")
print(f"Tổng ảnh còn lại: {len(taco['images'])}")
print(f"Tổng annotations: {len(taco['annotations'])}")