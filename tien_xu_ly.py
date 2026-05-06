import json
import os
import random
from collections import Counter

root = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco\train"
with open(os.path.join(root, "_annotations.coco.json")) as f:
    taco = json.load(f)

# 1. Xóa trash và other
REMOVE_IDS = {0, 4}
taco["categories"] = [c for c in taco["categories"] if c["id"] not in REMOVE_IDS]
taco["annotations"] = [a for a in taco["annotations"] if a["category_id"] not in REMOVE_IDS]

# 2. Reindex ID
old_to_new = {}
for i, cat in enumerate(taco["categories"]):
    old_to_new[cat["id"]] = i
    cat["id"] = i
for ann in taco["annotations"]:
    ann["category_id"] = old_to_new[ann["category_id"]]

# 3. Undersample plastic về 500
random.seed(42)
PLASTIC_ID = next(c["id"] for c in taco["categories"] if c["name"] == "plastic")
plastic = [a for a in taco["annotations"] if a["category_id"] == PLASTIC_ID]
others  = [a for a in taco["annotations"] if a["category_id"] != PLASTIC_ID]
taco["annotations"] = others + random.sample(plastic, 500)

# Kiểm tra
id_to_name = {c["id"]: c["name"] for c in taco["categories"]}
count = Counter(a["category_id"] for a in taco["annotations"])
for cid, cnt in sorted(count.items()):
    print(f"[{cid}] {id_to_name[cid]:<15} {cnt}")

import cv2
import matplotlib.pyplot as plt

# Gom annotation theo image_id
from collections import defaultdict
ann_by_image = defaultdict(list)
for ann in taco["annotations"]:
    ann_by_image[ann["image_id"]].append(ann)

id_to_img = {img["id"]: img for img in taco["images"]}

COLORS = {
    0: (255, 0, 0),    # cardboard - đỏ
    1: (0, 255, 0),    # glass - xanh lá
    2: (0, 0, 255),    # metal - xanh dương
    3: (255, 255, 0),  # paper - vàng
    4: (255, 0, 255),  # plastic - tím
}

# Mỗi class lấy 3 ảnh mẫu
SAMPLES_PER_CLASS = 3
fig, axes = plt.subplots(5, SAMPLES_PER_CLASS, figsize=(15, 25))

for class_idx, cat in enumerate(taco["categories"]):
    cid   = cat["id"]
    cname = cat["name"]

    # Lấy image_id có class này
    img_ids = list({a["image_id"] for a in taco["annotations"] if a["category_id"] == cid})
    sampled = random.sample(img_ids, min(SAMPLES_PER_CLASS, len(img_ids)))

    for col, img_id in enumerate(sampled):
        img_info = id_to_img[img_id]
        img_path = os.path.join(root, img_info["file_name"])
        image    = cv2.imread(img_path)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Vẽ bbox
        for ann in ann_by_image[img_id]:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            color = COLORS[ann["category_id"]]
            label = id_to_name[ann["category_id"]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        axes[class_idx][col].imshow(image)
        axes[class_idx][col].axis("off")
        if col == 0:
            axes[class_idx][col].set_title(cname, fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()