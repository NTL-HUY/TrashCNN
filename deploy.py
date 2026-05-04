"""
deploy.py - Inference script for Waste Detection
Supports: single image, batch of images, video file
Outputs : annotated images/video with bounding boxes, labels, confidence

Usage:
  # Single image
  python deploy.py --source path/to/image.jpg --weights weights/best_model.pth

  # Folder of images
  python deploy.py --source path/to/images/ --weights weights/best_model.pth

  # Video
  python deploy.py --source path/to/video.mp4 --weights weights/best_model.pth --save_video

  # With custom thresholds
  python deploy.py --source img.jpg --weights weights/best_model.pth --score_thresh 0.4 --nms_thresh 0.5
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from dataset import TARGET_CLASSES, NUM_CLASSES
from model import build_faster_rcnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Color palette per class
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "background": (128, 128, 128),
    "plastic": (255, 100, 50),  # orange
    "metal": (50, 150, 255),  # blue
    "paper": (80, 200, 80),  # green
    "glass": (150, 80, 220),  # purple
    "other": (200, 150, 50),  # brown
}


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy Faster R-CNN Waste Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input ──
    inp = parser.add_argument_group("Input")
    inp.add_argument("--source", type=str, required=True,
                     help="Path to image file, image folder, or video file")
    inp.add_argument("--weights", type=str, default="weights/best_model.pth",
                     help="Path to model weights (.pth)")
    inp.add_argument("--extensions", type=str, nargs="+",
                     default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
                     help="Image file extensions to process (for folder input)")

    # ── Model ──
    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--score_thresh", type=float, default=0.4,
                            help="Minimum confidence score for detections")
    model_args.add_argument("--nms_thresh", type=float, default=0.5,
                            help="NMS IoU threshold")
    model_args.add_argument("--max_detections", type=int, default=100,
                            help="Maximum number of detections per image")
    model_args.add_argument("--min_size", type=int, default=800,
                            help="Minimum image resize dimension")
    model_args.add_argument("--max_size", type=int, default=1333,
                            help="Maximum image resize dimension")

    # ── Device ──
    dev = parser.add_argument_group("Device")
    dev.add_argument("--device", type=str, default=None,
                     help="Device (cuda / cpu). Auto-detected if not specified.")
    dev.add_argument("--num_workers", type=int, default=0,
                     help="DataLoader workers (0 = main thread)")

    # ── Output ──
    out = parser.add_argument_group("Output")
    out.add_argument("--output_dir", type=str, default="predictions",
                     help="Directory to save prediction outputs")
    out.add_argument("--save_json", action="store_true",
                     help="Save predictions as JSON")
    out.add_argument("--save_video", action="store_true",
                     help="Save annotated video (for video input)")
    out.add_argument("--no_save_image", action="store_true",
                     help="Do not save annotated images (only display/JSON)")
    out.add_argument("--show", action="store_true",
                     help="Display predictions in window (requires display)")
    out.add_argument("--hide_labels", action="store_true",
                     help="Hide class labels on boxes")
    out.add_argument("--hide_conf", action="store_true",
                     help="Hide confidence scores on boxes")
    out.add_argument("--line_width", type=int, default=2,
                     help="Bounding box line width")
    out.add_argument("--font_size", type=int, default=14,
                     help="Label font size")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────
def load_model(weights_path: str, device: torch.device, args) -> torch.nn.Module:
    logger.info(f"Loading model from: {weights_path}")
    model = build_faster_rcnn(
        num_classes=NUM_CLASSES,
        min_size=args.min_size,
        max_size=args.max_size,
        box_nms_thresh=args.nms_thresh,
        box_score_thresh=args.score_thresh,
        box_detections_per_img=args.max_detections,
    )

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # Print training metrics if available
    metrics = ckpt.get("metrics", {})
    if metrics:
        logger.info(f"Loaded checkpoint | epoch={ckpt.get('epoch', '?')} | {metrics}")

    return model


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor [C, H, W]."""
    img_tensor = TF.to_tensor(pil_image.convert("RGB"))
    return img_tensor


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(
        model: torch.nn.Module,
        image_tensor: torch.Tensor,
        device: torch.device,
) -> Dict:
    """Run inference on a single image tensor."""
    inputs = [image_tensor.to(device)]
    outputs = model(inputs)
    pred = outputs[0]
    return {
        "boxes": pred["boxes"].cpu().numpy(),
        "labels": pred["labels"].cpu().numpy(),
        "scores": pred["scores"].cpu().numpy(),
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def draw_predictions(
        pil_image: Image.Image,
        predictions: Dict,
        score_thresh: float = 0.4,
        hide_labels: bool = False,
        hide_conf: bool = False,
        line_width: int = 2,
        font_size: int = 14,
) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image."""
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(font_size - 2, 10))
    except Exception:
        font = ImageFont.load_default()
        small = font

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    # Sort by score descending for better visualization
    order = np.argsort(scores)[::-1]
    boxes, labels, scores = boxes[order], labels[order], scores[order]

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        cls_name = TARGET_CLASSES[int(label)] if int(label) < len(TARGET_CLASSES) else "unknown"
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        for i in range(line_width):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # Build label text
        if not hide_labels and not hide_conf:
            text = f"{cls_name} {score:.2f}"
        elif not hide_labels:
            text = cls_name
        elif not hide_conf:
            text = f"{score:.2f}"
        else:
            text = ""

        if text:
            # Label background
            bbox_text = draw.textbbox((x1, y1), text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            pad = 3
            label_y1 = max(y1 - th - 2 * pad, 0)
            label_y2 = label_y1 + th + 2 * pad
            draw.rectangle([x1, label_y1, x1 + tw + 2 * pad, label_y2], fill=color)
            draw.text((x1 + pad, label_y1 + pad), text, fill=(255, 255, 255), font=font)

    return image


def add_summary_overlay(
        pil_image: Image.Image,
        predictions: Dict,
        score_thresh: float,
        inference_ms: float,
) -> Image.Image:
    """Add a semi-transparent summary box showing detection counts."""
    image = pil_image.copy()
    draw = ImageDraw.Draw(image, "RGBA")

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    mask = scores >= score_thresh
    labels_filtered = labels[mask]

    # Count per class
    class_counts = {}
    for lbl in labels_filtered:
        name = TARGET_CLASSES[int(lbl)] if int(lbl) < len(TARGET_CLASSES) else "?"
        class_counts[name] = class_counts.get(name, 0) + 1

    lines = [f"Detections: {mask.sum()}  |  {inference_ms:.0f}ms"]
    for cls, cnt in sorted(class_counts.items()):
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        lines.append(f"  {cls}: {cnt}")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    x, y, pad = 10, 10, 8
    line_h = 18
    box_h = len(lines) * line_h + 2 * pad
    box_w = 220
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((x + pad, y + pad + i * line_h), line, fill=(255, 255, 255), font=font)

    return image


# ─────────────────────────────────────────────
# Image Inference
# ─────────────────────────────────────────────
def process_image(
        model,
        image_path: Path,
        device: torch.device,
        args,
        output_dir: Path,
) -> Dict:
    pil_image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess_image(pil_image)

    t0 = time.time()
    predictions = predict(model, img_tensor, device)
    inference_ms = (time.time() - t0) * 1000

    n_detected = int((predictions["scores"] >= args.score_thresh).sum())
    logger.info(
        f"[{image_path.name}] Detections: {n_detected} | "
        f"Inference: {inference_ms:.1f}ms"
    )

    # Draw
    annotated = draw_predictions(
        pil_image=pil_image,
        predictions=predictions,
        score_thresh=args.score_thresh,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        line_width=args.line_width,
        font_size=args.font_size,
    )
    annotated = add_summary_overlay(annotated, predictions, args.score_thresh, inference_ms)

    result = {
        "file": str(image_path),
        "inference_ms": round(inference_ms, 1),
        "detections": [],
    }
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score < args.score_thresh:
            continue
        result["detections"].append({
            "class": TARGET_CLASSES[int(label)],
            "score": round(float(score), 4),
            "box": [round(float(x), 1) for x in box],
        })

    # Save
    if not args.no_save_image:
        out_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
        annotated.save(out_path)
        logger.info(f"  Saved → {out_path}")

    if args.show:
        cv_img = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        cv2.imshow("Waste Detection", cv_img)
        cv2.waitKey(0)

    return result


# ─────────────────────────────────────────────
# Video Inference
# ─────────────────────────────────────────────
def process_video(
        model,
        video_path: Path,
        device: torch.device,
        args,
        output_dir: Path,
) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path.name} | {width}x{height} @ {fps}fps | {total} frames")

    writer = None
    if args.save_video:
        out_video_path = output_dir / f"{video_path.stem}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        logger.info(f"Saving video → {out_video_path}")

    frame_idx = 0
    total_ms = 0.0
    all_detections = []

    from tqdm import tqdm
    pbar = tqdm(total=total, desc=f"Processing {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess_image(pil_frame)

        t0 = time.time()
        predictions = predict(model, img_tensor, device)
        ms = (time.time() - t0) * 1000
        total_ms += ms

        annotated = draw_predictions(
            pil_image=pil_frame,
            predictions=predictions,
            score_thresh=args.score_thresh,
            hide_labels=args.hide_labels,
            hide_conf=args.hide_conf,
            line_width=args.line_width,
            font_size=args.font_size,
        )
        annotated = add_summary_overlay(annotated, predictions, args.score_thresh, ms)

        if writer:
            out_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            writer.write(out_frame)

        if args.show:
            cv_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            cv2.imshow("Waste Detection", cv_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        pbar.update(1)
        pbar.set_postfix({"fps": f"{1000 / max(ms, 1):.1f}", "ms": f"{ms:.0f}"})

    cap.release()
    pbar.close()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_ms = total_ms / max(frame_idx, 1)
    logger.info(
        f"Video done: {frame_idx} frames | "
        f"Avg inference: {avg_ms:.1f}ms ({1000 / max(avg_ms, 1):.1f} FPS)"
    )
    return {"file": str(video_path), "frames": frame_idx, "avg_inference_ms": round(avg_ms, 1)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Model ──
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model = load_model(args.weights, device, args)
    logger.info(f"Model loaded ✓  |  Classes: {TARGET_CLASSES[1:]}")

    # ── Determine Input Type ──
    source = Path(args.source)
    results = []

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # ── Video ──
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    if source.is_file() and source.suffix.lower() in video_exts:
        logger.info(f"Mode: VIDEO | {source}")
        res = process_video(model, source, device, args, output_dir)
        results.append(res)

    # ── Single image ──
    elif source.is_file() and source.suffix.lower() in [e.lower() for e in args.extensions]:
        logger.info(f"Mode: IMAGE | {source}")
        res = process_image(model, source, device, args, output_dir)
        results.append(res)

    # ── Image folder ──
    elif source.is_dir():
        image_files = sorted([
            f for f in source.iterdir()
            if f.suffix.lower() in [e.lower() for e in args.extensions]
        ])
        logger.info(f"Mode: FOLDER | {len(image_files)} images in {source}")

        if not image_files:
            logger.warning("No images found in folder.")
        else:
            for img_path in image_files:
                res = process_image(model, img_path, device, args, output_dir)
                results.append(res)

    else:
        raise ValueError(f"Unsupported source: {source}")

    # ── Save JSON ──
    if args.save_json and results:
        json_path = output_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved → {json_path}")

    # ── Summary ──
    total_det = sum(len(r.get("detections", [])) for r in results)
    logger.info(f"\n✅ Done. Processed {len(results)} file(s) | Total detections: {total_det}")
    logger.info(f"   Output → {output_dir}/")


if __name__ == "__main__":
    main()
