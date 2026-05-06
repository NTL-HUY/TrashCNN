from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")

# Ground truth — 2 ảnh
targets = [
    {
        # Ảnh 0: có 1 con chó (class 1) ở box [10,20,50,80]
        "boxes":  torch.tensor([[10, 20, 50, 80]], dtype=torch.float32),
        "labels": torch.tensor([1]),
    },
    {
        # Ảnh 1: có 1 con mèo (class 2) ở box [5, 5, 40, 40]
        "boxes":  torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
        "labels": torch.tensor([2]),
    },
]

# Prediction — model đoán gì
preds = [
    {
        # Ảnh 0: đoán đúng box gần đúng, score 0.9
        "boxes":  torch.tensor([[12, 22, 48, 78]], dtype=torch.float32),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    },
    {
        # Ảnh 1: đoán sai class (đoán 1 thay vì 2), score 0.8
        "boxes":  torch.tensor([[5, 5, 40, 40]], dtype=torch.float32),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.8]),
    },
]

metric.update(preds, targets)
result = metric.compute()

print(f"mAP     : {result['map']:.4f}")       # trung bình tất cả class
print(f"mAP@50  : {result['map_50']:.4f}")    # IoU threshold 0.5
print(f"mAP@75  : {result['map_75']:.4f}")    # IoU threshold 0.75