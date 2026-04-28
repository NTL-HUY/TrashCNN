import torch
from torchvision import transforms

from TrashCNN.dataset import TrashDataset
from TrashCNN.inference import load_model
from TrashCNN.model import build_model

import matplotlib.pyplot as plt
import matplotlib.patches as patches
NUM_CLASSES = 7
MODEL_PATH  = "checkpoints/best_model.pth"
DATA_ROOT   = r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco"
SCORE_THRESH = 0.25
CLASS_NAMES = ["trash", "cardboard", "glass", "metal", "other", "paper", "plastic"]
COLORS      = ["#FF4757", "#2ED573", "#1E90FF", "#FFA502", "#ECCC68", "#A29BFE", "#FF6B81"]
def load_model(path,device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("avg_loss", "?")
    print(f"Loaded model  (epoch={epoch}, best_loss={loss})")
    return model

def predict(model,images,device):
    tensors = [transforms.ToTensor()(img).to(device) for img in images]
    with torch.no_grad():
        outputs = model(tensors)
    return outputs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH,device)

    dataset = TrashDataset(root=DATA_ROOT,split="valid")
    image, _ = dataset[1]
    output = predict(model,[image],device)
    ve_anh(image, output[0])
    print(f"Predicted: {output}")

def ve_anh(anh, output):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(anh)

    boxes  = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESH:
            continue

        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        mau = COLORS[label % len(COLORS)]
        ten = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"cls{label}"

        # Vẽ ô vuông
        rect = patches.Rectangle((x1, y1), w, h,
                                  linewidth=2, edgecolor=mau, facecolor="none")
        ax.add_patch(rect)

        # Vẽ nhãn
        ax.text(x1, y1 - 5, f"{ten} {score:.0%}",
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(facecolor=mau, edgecolor="none", pad=2))

    plt.axis("off")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()