import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from torchvision import transforms

from TrashCNN.dataset import TrashDataset

root = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"

transform = transforms.ToTensor()
dataset = TrashDataset(root, split="train", transforms=transform)

# Info
id_to_name = {cat["id"] + 1: cat["name"] for cat in dataset.categories}
print(f"Tổng ảnh: {len(dataset)}")
print(f"Số class: {dataset.get_num_classes()} (gồm background)")
print("Classes:", {v: k for k, v in id_to_name.items()})

# Gom index theo class
class_to_indices = defaultdict(list)
for idx, img_info in enumerate(dataset.images):
    anns = dataset.img_to_anns.get(img_info["id"], [])
    for ann in anns:
        label = dataset.cat_id_to_label[ann["category_id"]]
        class_to_indices[label].append(idx)
        break  # mỗi ảnh chỉ tính 1 lần

# Vẽ 3 ảnh mẫu cho mỗi class
SAMPLES = 4
COLORS  = ["red", "green", "blue", "orange", "purple"]
classes = sorted(id_to_name.keys())

fig, axes = plt.subplots(len(classes), SAMPLES, figsize=(15, 5 * len(classes)))

for row, label in enumerate(classes):
    indices = class_to_indices[label]
    sampled = random.sample(indices, min(SAMPLES, len(indices)))

    for col, idx in enumerate(sampled):
        image, target = dataset[idx]

        # Tensor → numpy
        img_np = image.permute(1, 2, 0).numpy()

        ax = axes[row][col]
        ax.imshow(img_np)
        ax.axis("off")

        if col == 0:
            ax.set_title(id_to_name[label], fontsize=14, fontweight="bold")

        # Vẽ bbox
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            color = COLORS[(lbl.item() - 1) % len(COLORS)]
            rect  = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, id_to_name[lbl.item()],
                    color=color, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()