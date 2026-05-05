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
# Score diagnostic – kiểm tra phân phối confidence trước khi eval
# ---------------------------------------------------------------------------

def score_diagnostic(
    model:  torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 30,
) -> None:
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            images  = [img.to(device) for img in images]
            outputs = model(images)
            for out in outputs:
                all_scores.extend(out["scores"].cpu().tolist())

    total = len(all_scores)
    if total == 0:
        print("[Diagnostic] Model không tạo ra prediction nào cả!")
        print("             Kiểm tra lại checkpoint hoặc xem model có load đúng không.")
        return

    print(f"\n[Diagnostic] Score distribution ({n_batches} batches, "
          f"{total} raw predictions):")
    for thresh in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        count = sum(1 for s in all_scores if s >= thresh)
        pct   = 100 * count / total
        flag  = "  ← threshold hiện tại" if thresh == 0.30 else ""
        print(f"    score >= {thresh:.2f} : {count:>5} predictions  "
              f"({pct:5.1f}%){flag}")
    print()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model:           torch.nn.Module,
    dataloader:      DataLoader,
    device:          torch.device,
    iou_type:        str   = "bbox",
    score_threshold: float = 0.05,
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

    total_preds = 0
    total_gts   = 0

    print(f"[Eval] Đang thu thập predictions (score_threshold={score_threshold}) ...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]

            # Forward pass → list of prediction dicts
            outputs = model(images)

            # ── Lọc theo score threshold ──────────────────────────────
            preds = []
            for out in outputs:
                keep = out["scores"] >= score_threshold
                preds.append({
                    "boxes":  out["boxes"][keep].cpu(),
                    "scores": out["scores"][keep].cpu(),
                    "labels": out["labels"][keep].cpu(),
                })
                total_preds += int(keep.sum())

            # ── Ground truths ─────────────────────────────────────────
            gts = []
            for tgt in targets:
                gts.append({
                    "boxes":  tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                })
                total_gts += len(tgt["boxes"])

            metric.update(preds, gts)

            if (batch_idx + 1) % 50 == 0:
                print(f"[Eval] Đã xử lý {batch_idx + 1}/{len(dataloader)} batch  "
                      f"| preds so far: {total_preds}")

    # ── Cảnh báo khi không có prediction nào ─────────────────────────────
    print(f"\n[Eval] Tổng ground truth boxes : {total_gts}")
    print(f"[Eval] Tổng prediction boxes   : {total_preds} "
          f"(threshold={score_threshold})")

    if total_preds == 0:
        print(
            "\n[Eval] ⚠  KHÔNG CÓ PREDICTION NÀO QUA THRESHOLD!\n"
            f"          score_threshold={score_threshold} quá cao.\n"
            "          Chạy score_diagnostic() để xem phân phối confidence,\n"
            "          rồi hạ threshold xuống (ví dụ: --score-thresh 0.01)."
        )
    elif total_preds < total_gts * 0.1:
        print(
            f"[Eval] ⚠  Số prediction ({total_preds}) << ground truth ({total_gts}).\n"
            "          Có thể threshold vẫn còn cao. Thử --score-thresh 0.01."
        )

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