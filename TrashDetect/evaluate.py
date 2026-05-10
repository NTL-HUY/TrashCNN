# evaluate.py
import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.autonotebook import tqdm
from pprint import pprint

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import TrashDataset, collate_fn
from model import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"C:\Users\BAOHUY\Downloads\TACO dataset.v1i.coco")
    parser.add_argument("--model_path", type=str, default="trained_models/resnet18.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def get_test_transform():
    return A.Compose([
        A.Resize(416, 416),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load test dataset ──────────────────────────────────────
    test_dataset = TrashDataset(
        root=args.data_path,
        split='test',                   # ✅ Chỉ dùng test ở đây
        transforms=get_test_transform()
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False
    )

    # ── Load model ────────────────────────────────────────────
    model = build_model(num_classes=test_dataset.get_num_classes()).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded model from: {args.model_path}")
    print(f"Trained for {checkpoint['epoch']} epochs | Best mAP (valid): {checkpoint['best_map']:.4f}\n")

    # ── Inference ─────────────────────────────────────────────
    metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")
    per_class_metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval", class_metrics=True)

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating on test set"):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds   = model(images)
            metric.update(preds, targets)
            per_class_metric.update(preds, targets)

    # ── Kết quả ───────────────────────────────────────────────
    result      = metric.compute()
    result_cls  = per_class_metric.compute()
    id_to_name  = {cat["id"] + 1: cat["name"] for cat in test_dataset.categories}

    print("\n" + "=" * 55)
    print("         KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 55)
    print(f"  mAP@[0.5:0.95] : {result['map']:.4f}")
    print(f"  mAP@0.50       : {result['map_50']:.4f}")
    print(f"  mAP@0.75       : {result['map_75']:.4f}")
    print(f"  mAP small      : {result['map_small']:.4f}")
    print(f"  mAP medium     : {result['map_medium']:.4f}")
    print(f"  mAP large      : {result['map_large']:.4f}")
    print("-" * 55)
    print("  AP từng class:")
    for idx, ap in enumerate(result_cls["map_per_class"].tolist()):
        class_id   = idx + 1
        class_name = id_to_name.get(class_id, f"class_{class_id}")
        print(f"    [{class_id:2d}] {class_name:<25} : {ap:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    args = get_args()
    evaluate(args)