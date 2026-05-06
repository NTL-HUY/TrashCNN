"""
utils.py - Utilities: mAP computation, checkpoint management, logger, EarlyStopping
"""

import os
import json
import math
import time
import shutil
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from torchvision.ops import box_iou


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
def setup_logger(log_dir: str = "runs", name: str = "train") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────
# mAP Computation (COCO-style)
# ─────────────────────────────────────────────
class MetricLogger:
    """
    Accumulates predictions and ground truths for mAP calculation.
    Supports: mAP@[0.5:0.95] and mAP@0.5
    """

    def __init__(self, num_classes: int, class_names: List[str], iou_thresholds: Optional[List[float]] = None):
        self.num_classes = num_classes
        self.class_names = class_names
        if iou_thresholds is None:
            # COCO: 0.50:0.05:0.95
            self.iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
        else:
            self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        """Clear all stored predictions and targets."""
        self.all_predictions = []  # list of {boxes, scores, labels}
        self.all_targets     = []  # list of {boxes, labels}

    def update(self, predictions: List[Dict], targets: List[Dict]):
        """
        Add batch predictions and targets.
        predictions: list of dicts with 'boxes', 'scores', 'labels'
        targets    : list of dicts with 'boxes', 'labels'
        """
        for pred, tgt in zip(predictions, targets):
            self.all_predictions.append({
                "boxes":  pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            self.all_targets.append({
                "boxes":  tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })

    def compute(self) -> Dict[str, float]:
        """Compute mAP across all IoU thresholds and per class."""
        if not self.all_predictions:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        all_ap = []
        ap_50_list = []
        per_class_ap = {}

        for cls_idx in range(1, self.num_classes):  # skip background (0)
            aps_for_class = []
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_ap_for_class(cls_idx, iou_thresh)
                aps_for_class.append(ap)
            mean_ap_cls = float(np.mean(aps_for_class)) if aps_for_class else 0.0
            per_class_ap[self.class_names[cls_idx]] = round(mean_ap_cls, 4)
            all_ap.append(aps_for_class)
            ap_50_idx = self.iou_thresholds.index(0.5) if 0.5 in self.iou_thresholds else 0
            ap_50_list.append(aps_for_class[ap_50_idx])

        # mAP averaged over classes and IoU thresholds
        all_ap_flat = [ap for cls_aps in all_ap for ap in cls_aps]
        mAP    = float(np.mean(all_ap_flat)) if all_ap_flat else 0.0
        mAP_50 = float(np.mean(ap_50_list)) if ap_50_list else 0.0

        # mAP@75
        ap_75_list = []
        if 0.75 in self.iou_thresholds:
            ap_75_idx = self.iou_thresholds.index(0.75)
            for cls_aps in all_ap:
                ap_75_list.append(cls_aps[ap_75_idx])
        mAP_75 = float(np.mean(ap_75_list)) if ap_75_list else 0.0

        result = {
            "mAP":    round(mAP,    4),
            "mAP_50": round(mAP_50, 4),
            "mAP_75": round(mAP_75, 4),
        }
        result.update({f"AP_{k}": v for k, v in per_class_ap.items()})
        return result

    def _compute_ap_for_class(self, cls_idx: int, iou_thresh: float) -> float:
        """Compute AP for a single class at a given IoU threshold."""
        # Collect all predictions and ground truths for this class
        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0
        img_gt_used = {}

        for img_idx, (pred, tgt) in enumerate(zip(self.all_predictions, self.all_targets)):
            gt_mask  = tgt["labels"] == cls_idx
            gt_boxes = tgt["boxes"][gt_mask]
            n_gt += len(gt_boxes)
            img_gt_used[img_idx] = torch.zeros(len(gt_boxes), dtype=torch.bool)

            pred_mask   = pred["labels"] == cls_idx
            pred_boxes  = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            if len(pred_boxes) == 0:
                continue

            # Sort by score descending
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order]

            for pb, ps in zip(pred_boxes, pred_scores):
                scores_list.append(float(ps))
                if len(gt_boxes) == 0:
                    tp_list.append(0)
                    fp_list.append(1)
                    continue

                iou = box_iou(pb.unsqueeze(0), gt_boxes)  # (1, n_gt)
                max_iou, max_idx = iou[0].max(0) if iou.numel() > 0 else (torch.tensor(0.0), torch.tensor(0))

                if len(iou[0]) > 0:
                    max_iou_val, max_idx_val = float(iou[0].max()), int(iou[0].argmax())
                else:
                    max_iou_val, max_idx_val = 0.0, 0

                if max_iou_val >= iou_thresh and not img_gt_used[img_idx][max_idx_val]:
                    tp_list.append(1)
                    fp_list.append(0)
                    img_gt_used[img_idx][max_idx_val] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if n_gt == 0:
            return 0.0

        if not scores_list:
            return 0.0

        # Sort all by score
        order = np.argsort(scores_list)[::-1]
        tp_arr = np.array(tp_list)[order]
        fp_arr = np.array(fp_list)[order]

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)

        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        recalls    = tp_cum / (n_gt + 1e-8)

        # Interpolate AP (11-point or area under curve)
        ap = self._voc_ap(recalls, precisions)
        return ap

    @staticmethod
    def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute AP using PASCAL VOC 11-point interpolation."""
        recalls    = np.concatenate([[0.0], recalls, [1.0]])
        precisions = np.concatenate([[1.0], precisions, [0.0]])
        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        idx = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap  = np.sum((recalls[idx] - recalls[idx - 1]) * precisions[idx])
        return float(ap)


# ─────────────────────────────────────────────
# Running Average Meter
# ─────────────────────────────────────────────
class AverageMeter:
    """Tracks and computes running average of a value."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class LossMeters:
    """Tracks all Faster R-CNN loss components."""
    KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "total"]

    def __init__(self):
        self.meters = {k: AverageMeter(k) for k in self.KEYS}

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def update(self, loss_dict: Dict[str, torch.Tensor]):
        total = sum(loss_dict.values())
        for k, v in loss_dict.items():
            if k in self.meters:
                self.meters[k].update(float(v))
        self.meters["total"].update(float(total))

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def __str__(self):
        parts = [f"total={self.meters['total'].avg:.4f}"]
        for k in self.KEYS[:-1]:
            parts.append(f"{k.replace('loss_','')}: {self.meters[k].avg:.4f}")
        return " | ".join(parts)


# ─────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────
class CheckpointManager:
    """
    Saves best model (by mAP) and last model.
    Directory structure:
      weights/
        best_model.pth
        last_model.pth
    """
    def __init__(self, save_dir: str = "weights", metric: str = "mAP_50"):
        self.save_dir   = Path(save_dir)
        self.metric     = metric
        self.best_score = -1.0
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        scaler=None,
    ):
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics":   metrics,
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()

        # Always save last
        last_path = self.save_dir / "last_model.pth"
        torch.save(state, last_path)

        # Save best if improved
        score = metrics.get(self.metric, 0.0)
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            best_path = self.save_dir / "best_model.pth"
            shutil.copy2(last_path, best_path)

        return is_best

    def load(self, path: str, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """Stops training if monitored metric does not improve for `patience` epochs."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, metric: str = "mAP_50"):
        self.patience  = patience
        self.min_delta = min_delta
        self.metric    = metric
        self.best      = -1.0
        self.counter   = 0
        self.triggered = False

    def step(self, metrics: Dict[str, float]) -> bool:
        score = metrics.get(self.metric, 0.0)
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─────────────────────────────────────────────
# Warmup + Cosine Annealing Scheduler
# ─────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `min_lr`.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            factor = (ep + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale factor to ensure min_lr
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


# ─────────────────────────────────────────────
# Gradient Clipping Wrapper
# ─────────────────────────────────────────────
def clip_gradients(model: torch.nn.Module, max_norm: float = 5.0) -> float:
    """Clips gradient norm and returns the norm before clipping."""
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


# ─────────────────────────────────────────────
# Epoch Time Formatter
# ─────────────────────────────────────────────
def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
