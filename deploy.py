import argparse

import cv2
import torch
import numpy as np
from TrashCNN.model import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=r"test_images/test.jpg", help="path to test image")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--checkpoint", type=str, default="trained_models/last_model.pth", help="path to log file")
    args = parser.parse_args()
    return args

CLASS_NAMES = ["trash", "cardboard", "glass", "metal", "other", "paper", "plastic"]
def deploy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    orgin_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(orgin_image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image / 255.0

    # normalize

    image = [image.transpose((2, 0, 1))]
    image = [torch.tensor(image[0], dtype=torch.float32).to(device)]
    prediction = model(image)
    print(prediction)

if __name__ == "__main__":
    args = get_args()
    deploy(args)

