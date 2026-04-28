import torch
import os
import json
from PIL import Image


class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = os.path.join(root, split)
        self.transforms = transforms

        with open(os.path.join(self.root, "_annotations.coco.json")) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __getitem__(self, item):
        image_name = self.images[item]['file_name']
        image_id = self.images[item]['id']

        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path).convert('RGB')

        anns = self.img_to_anns.get(image_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.zeros((0, 4), dtype=torch.float32) if len(boxes) == 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))