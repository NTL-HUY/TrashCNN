"""
evaluator.py – Tính mAP (mean Average Precision)
==================================================
Sử dụng thư viện torchmetrics.detection.MeanAveragePrecision
theo chuẩn COCO IoU (AP@[0.5:0.95]).

Luồng:
  1. Chạy model.eval() trên tập test
  2. Thu thập tất cả predictions và ground truths
  3. Gọi metric.compute() để lấy mAP tổng thể + mAP theo từng class
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import SUPERCLASS_NAMES


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    iou_type:   str = "bbox",
    score_threshold: float = 0.3,
) -> dict:
    """
    Đánh giá model trên một DataLoader, trả về kết quả mAP.

    Args:
        model:           FasterRCNN đã train
        dataloader:      DataLoader (val hoặc test set)
        device:          cuda/cpu
        iou_type:        "bbox" (detection) hoặc "segm" (segmentation)
        score_threshold: lọc bỏ prediction có confidence < threshold

    Returns:
        results: dict gồm map, map_50, map_75, map_per_class, mar_*
    """
    model.eval()

    # ── Khởi tạo metric ───────────────────────────────────────────────────
    metric = MeanAveragePrecision(
        iou_type=iou_type,
        class_metrics=True,   # mAP từng class
    )
    metric.to(device)

    print("[Eval] Đang thu thập predictions ...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]

            # Forward pass → list of prediction dicts
            outputs = model(images)
            # outputs[i] = {"boxes": Tensor, "labels": Tensor, "scores": Tensor}

            # ── Lọc theo score threshold ──────────────────────────────
            preds = []
            for out in outputs:
                keep = out["scores"] >= score_threshold
                preds.append({
                    "boxes":  out["boxes"][keep].cpu(),
                    "scores": out["scores"][keep].cpu(),
                    "labels": out["labels"][keep].cpu(),
                })

            # ── Ground truths ─────────────────────────────────────────
            gts = []
            for tgt in targets:
                gts.append({
                    "boxes":  tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                })

            metric.update(preds, gts)

            if (batch_idx + 1) % 50 == 0:
                print(f"[Eval] Đã xử lý {batch_idx + 1}/{len(dataloader)} batch")

    # ── Tính mAP ─────────────────────────────────────────────────────────
    print("[Eval] Đang tính mAP ...")
    results = metric.compute()

    # ── In kết quả ───────────────────────────────────────────────────────
    _print_results(results)

    return results


# ---------------------------------------------------------------------------
# Helper: in kết quả đẹp
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    """In mAP tổng thể và per-class ra console."""
    print("\n" + "=" * 55)
    print("  KẾT QUẢ ĐÁNH GIÁ (COCO mAP)")
    print("=" * 55)
    print(f"  mAP  @[0.50:0.95] : {results['map']:.4f}")
    print(f"  mAP  @[0.50]      : {results['map_50']:.4f}")
    print(f"  mAP  @[0.75]      : {results['map_75']:.4f}")
    print(f"  mAR  (max 100)    : {results['mar_100']:.4f}")
    print("-" * 55)

    # mAP theo từng superclass
    map_per_class = results.get("map_per_class", [])
    classes       = results.get("classes", [])

    if len(map_per_class) > 0:
        print("  mAP per class:")
        for cls_idx, ap in zip(classes.tolist(), map_per_class.tolist()):
            cls_name = SUPERCLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
            print(f"    {cls_name:<12}: {ap:.4f}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Quick predict – dùng để inference 1 ảnh
# ---------------------------------------------------------------------------

def predict_single(
    model:          torch.nn.Module,
    image_tensor:   torch.Tensor,
    device:         torch.device,
    score_threshold: float = 0.3,
) -> dict:
    """
    Chạy model trên 1 ảnh đã được transform.

    Args:
        image_tensor: FloatTensor [C, H, W] đã normalize
        score_threshold: ngưỡng confidence

    Returns:
        dict {"boxes", "labels", "scores"}
    """
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])

    out  = outputs[0]
    keep = out["scores"] >= score_threshold
    return {
        "boxes":  out["boxes"][keep].cpu(),
        "labels": out["labels"][keep].cpu(),
        "scores": out["scores"][keep].cpu(),
    }
