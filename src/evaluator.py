"""
evaluator.py – Tính mAP (mean Average Precision)
==================================================
Sử dụng torchmetrics.detection.MeanAveragePrecision (COCO IoU AP@[0.5:0.95]).
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import SUPERCLASS_NAMES


# ---------------------------------------------------------------------------
# Score diagnostic
# ---------------------------------------------------------------------------

def score_diagnostic(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    n_batches:  int = 30,
) -> None:
    """
    Chạy model trên n_batches đầu, in phân phối confidence score.
    Gọi khi nghi ngờ mAP = 0.0 do threshold quá cao.

    Ví dụ output:
      [Diagnostic] Score distribution (30 batches, 842 predictions):
        score >= 0.01 :   831  (98.7%)
        score >= 0.05 :   612  (72.7%)
        score >= 0.10 :   341  (40.5%)
        score >= 0.20 :    87  (10.3%)
        score >= 0.30 :    12  ( 1.4%)  ← threshold hiện tại
        score >= 0.50 :     0  ( 0.0%)
    """
    # eval mode: FasterRCNN from scratch trả về predictions khi không có targets
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            outputs = model([img.to(device) for img in images])
            for out in outputs:
                all_scores.extend(out["scores"].cpu().tolist())

    total = len(all_scores)
    if total == 0:
        print("[Diagnostic] ⚠  Model không tạo ra prediction nào!")
        print("             Kiểm tra checkpoint hoặc quá trình training.")
        return

    print(f"\n[Diagnostic] Score distribution "
          f"({n_batches} batches, {total} predictions):")
    for thresh in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        count = sum(1 for s in all_scores if s >= thresh)
        pct   = 100 * count / total
        flag  = "  ← threshold hiện tại" if abs(thresh - 0.30) < 1e-6 else ""
        print(f"    score >= {thresh:.2f} : {count:>5}  ({pct:5.1f}%){flag}")
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
    Đánh giá model trên DataLoader, trả về kết quả mAP.

    Args:
        model:           FasterRCNN
        dataloader:      DataLoader (val hoặc test, batch_size=1 khuyến nghị)
        device:          cuda / cpu
        iou_type:        "bbox"
        score_threshold: lọc prediction có confidence < threshold.
                         Mặc định 0.05

    Returns:
        dict: map, map_50, map_75, mar_100, map_per_class, classes
    """
    model.eval()

    metric = MeanAveragePrecision(iou_type=iou_type, class_metrics=True)
    metric.to(device)

    total_preds = 0
    total_gts   = 0

    print(f"[Eval] Đang thu thập predictions (score_threshold={score_threshold}) ...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            outputs = model([img.to(device) for img in images])

            # Lọc theo threshold
            preds = []
            for out in outputs:
                keep = out["scores"] >= score_threshold
                preds.append({
                    "boxes":  out["boxes"][keep].cpu(),
                    "scores": out["scores"][keep].cpu(),
                    "labels": out["labels"][keep].cpu(),
                })
                total_preds += int(keep.sum())

            gts = []
            for tgt in targets:
                gts.append({
                    "boxes":  tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                })
                total_gts += len(tgt["boxes"])

            metric.update(preds, gts)

            if (batch_idx + 1) % 50 == 0:
                print(f"[Eval] {batch_idx+1}/{len(dataloader)} batch  "
                      f"| preds: {total_preds}")

    # ── Cảnh báo ─────────────────────────────────────────────────────────
    print(f"\n[Eval] Ground truth boxes : {total_gts}")
    print(f"[Eval] Prediction boxes   : {total_preds}  "
          f"(threshold={score_threshold})")

    if total_preds == 0:
        print(
            "\n[Eval] ⚠  KHÔNG CÓ PREDICTION NÀO QUA THRESHOLD!\n"
            f"          Chạy score_diagnostic() để xem phân phối confidence.\n"
            f"          Thử: --score-thresh 0.01"
        )
    elif total_preds < total_gts * 0.1:
        print(f"[Eval] ⚠  Predictions ({total_preds}) << GT ({total_gts}). "
              f"Thử hạ --score-thresh.")

    # ── Tính mAP ─────────────────────────────────────────────────────────
    print("[Eval] Đang tính mAP ...")
    results = metric.compute()
    _print_results(results)
    return results


# ---------------------------------------------------------------------------
# Helper: in kết quả
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    print("\n" + "=" * 55)
    print("  KẾT QUẢ ĐÁNH GIÁ (COCO mAP)")
    print("=" * 55)
    print(f"  mAP  @[0.50:0.95] : {results['map']:.4f}")
    print(f"  mAP  @[0.50]      : {results['map_50']:.4f}")
    print(f"  mAP  @[0.75]      : {results['map_75']:.4f}")
    print(f"  mAR  (max 100)    : {results['mar_100']:.4f}")
    print("-" * 55)

    map_per_class = results.get("map_per_class", [])
    classes       = results.get("classes",       [])

    if len(map_per_class) > 0:
        print("  mAP per class:")
        for cls_idx, ap in zip(classes.tolist(), map_per_class.tolist()):
            cls_name = SUPERCLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
            print(f"    {cls_name:<12}: {ap:.4f}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Helper: serialize tensor → JSON-safe
# ---------------------------------------------------------------------------

def tensor_to_json(v):
    """
    Chuyển kết quả torchmetrics sang kiểu JSON-serializable.
    torchmetrics trả về nhiều kiểu:
      • scalar tensor (0-dim) → float
      • 1-D tensor            → list[float]   (map_per_class, classes)
      • Python scalar         → giữ nguyên
    """
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.item())
        return [float(x) for x in v.tolist()]
    return v


# ---------------------------------------------------------------------------
# Quick predict – inference 1 ảnh
# ---------------------------------------------------------------------------

def predict_single(
    model:           torch.nn.Module,
    image_tensor:    torch.Tensor,
    device:          torch.device,
    score_threshold: float = 0.3,
) -> dict:
    """
    Chạy model trên 1 ảnh đã được transform.

    Args:
        image_tensor: FloatTensor [C, H, W] đã normalize
        score_threshold: ngưỡng confidence

    Returns:
        {"boxes": Tensor[K,4], "labels": Tensor[K], "scores": Tensor[K]}
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