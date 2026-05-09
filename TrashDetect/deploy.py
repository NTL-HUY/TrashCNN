import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import TrashDataset
from model import build_model

# Cấu hình
DATA_PATH = r"TACO dataset.v1i.coco"
WEIGHTS_PATH = "trained_models/best_model.pth"
BACKBONE_WEIGHTS = "../TrashClassify/trained_models/best_model.pth"
IMAGE_SIZE = 416
CLASS_NAMES = ["__background__", "trash", "cardboard", "glass", "metal", "other", "paper", "plastic"]


def get_inference_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_dataset_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0))


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def deploy_sample(model, dataset, index, device, threshold=0.5):
    image_tensor, target = dataset[index]

    # Predict
    model.eval()
    prediction = model([image_tensor.to(device)])[0]

    # Xử lý ảnh
    image_np = denormalize(image_tensor)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 1. Vẽ Ground Truth (Màu Đỏ)
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = map(int, box)
        label_idx = label.item()
        name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"ID:{label_idx}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image_bgr, f"GT:{name}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 2. Vẽ Prediction (Màu Xanh Lá)
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"ID:{label}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{name} {score:.2f}", (x1, min(y2 + 15, IMAGE_SIZE - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Show kết quả
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title(f"Sample Index: {index} | Red: GT | Green: Pred (threshold={threshold})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    dataset = TrashDataset(
        root=DATA_PATH,
        split="train",
        transforms=get_dataset_transform(IMAGE_SIZE),
        image_size=IMAGE_SIZE
    )
    print(f"Dataset size: {len(dataset)} images")

    # Build model và load checkpoint
    num_classes = dataset.get_num_classes()
    model = build_model(num_classes=num_classes, my_weights_path=BACKBONE_WEIGHTS)

    print(f"Loading detection checkpoint: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Chạy inference một vài mẫu
    indices = [0, 5, 10, 15, 20]
    for idx in indices:
        if idx >= len(dataset):
            print(f"Index {idx} vượt quá dataset size ({len(dataset)}), bỏ qua.")
            continue
        print(f"\nProcessing index {idx}...")
        deploy_sample(model, dataset, idx, device, threshold=0.4)


if __name__ == "__main__":
    main()