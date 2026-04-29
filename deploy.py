import argparse

import cv2
import torch
import numpy as np

from dataset import TrashDataset
from model import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=r"test_images/test.jpg", help="path to test image")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--checkpoint", type=str, default="trained_models/best_model.pth", help="path to log file")
    args = parser.parse_args()
    return args

CLASS_NAMES = ["trash", "cardboard", "glass", "metal", "other", "paper", "plastic"]
def deploy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    origin_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image / 255.0

    # normalize

    image = [image.transpose((2, 0, 1))]
    image = [torch.tensor(image[0], dtype=torch.float32).to(device)]
    with torch.no_grad():
        prediction = model(image)
    print(prediction)
    pred = prediction[0]

    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    # ===== Scale bbox về ảnh gốc =====
    scale_x = w / args.image_size
    scale_y = h / args.image_size

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # ===== Vẽ bbox =====
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.3:   # threshold (có thể giảm xuống 0.1 để debug)
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASS_NAMES[label]

        # rectangle
        cv2.rectangle(origin_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # label
        text = f"{class_name}: {score:.2f}"
        cv2.putText(origin_image, text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    dataset = TrashDataset(
        root=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco",
        split="train"
    )

    # ===== Show =====
    # cv2.imshow("Result", origin_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__ == "__main__":
    dataset = TrashDataset(
        root=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco",
        split="valid"
    )

    image , target = dataset[9]
    print(image)
    print(target)
    # args = get_args()
    # deploy(args)

