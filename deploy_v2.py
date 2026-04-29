import cv2
import torch
import numpy as np
from dataset import TrashDataset
from model import build_model
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = ["trash", "cardboard", "glass", "metal", "other", "paper", "plastic"]
COLORS = {
    "pred": (0, 255, 0),    # xanh lá - prediction
    "gt":   (255, 0, 0),    # đỏ - ground truth
}

def deploy_with_gt(index=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load model =====
    model = build_model(num_classes=len(CLASS_NAMES) + 1).to(device)
    checkpoint = torch.load("trained_models/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ===== Load dataset =====
    dataset = TrashDataset(
        root=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco",
        split="valid",
        transforms=transforms.ToTensor()
    )

    image_tensor, target = dataset[index]

    # ===== Predict =====
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    pred = prediction[0]
    pred_boxes  = pred["boxes"].cpu().numpy()
    pred_labels = pred["labels"].cpu().numpy()
    pred_scores = pred["scores"].cpu().numpy()

    gt_boxes  = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()

    # ===== Convert tensor → numpy image =====
    image_np = image_tensor.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
    image_np = (image_np * 255).astype(np.uint8).copy()

    # ===== Vẽ Ground Truth (đỏ) =====
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[label]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), COLORS["gt"], 2)
        cv2.putText(image_np, f"GT: {class_name}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS["gt"], 2)

    # ===== Vẽ Prediction (xanh lá) =====
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < 0.3:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[label]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), COLORS["pred"], 2)
        cv2.putText(image_np, f"Pred: {class_name} {score:.2f}",
                    (x1, y2 + 15),  # vẽ dưới box để không đè GT
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS["pred"], 2)

    # ===== Show =====
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(f"Index {index} | 🔴 Ground Truth  🟢 Prediction", fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="red",   linewidth=2, label="Ground Truth"),
        Line2D([0], [0], color="green", linewidth=2, label="Prediction"),
    ]
    plt.legend(handles=legend, loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig("result.jpg", bbox_inches="tight")
    plt.show()

    print(f"\n📊 Ground Truth:  {len(gt_boxes)} boxes")
    print(f"📊 Predictions:   {len(pred_boxes)} boxes (all), {(pred_scores >= 0.3).sum()} boxes (score ≥ 0.3)")


if __name__ == "__main__":
    for i in range(10):
        deploy_with_gt(i*2)
