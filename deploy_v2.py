import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from dataset import TrashDataset, collate_fn
from model import build_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--data_path",    type=str,   default=r"TACO dataset.v1i.coco")
    parser.add_argument("--model_path",   type=str,   default="trained_models/best_model.pth")
    parser.add_argument("--image_path",   type=str,   default=None)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--camera",       action="store_true",
                        help="Chạy real-time với webcam")
    parser.add_argument("--camera_id",   type=int,   default=0,
                        help="ID webcam (mặc định 0)")
    return parser.parse_args()


COLORS = {
    1: (255,   0,   0),
    2: (  0, 255,   0),
    3: (  0,   0, 255),
    4: (255, 165,   0),
    5: (128,   0, 128),
}


def denormalize(tensor):
    """Chuyển normalized tensor (C,H,W) → uint8 numpy (H,W,3) để hiển thị đúng màu."""
    img = tensor.cpu().permute(1, 2, 0).numpy()          # (H,W,3) float
    img = img * IMAGENET_STD + IMAGENET_MEAN              # undo normalize
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def draw_boxes(img_np, boxes, labels, scores, id_to_name,
               score_thresh, show_score=True):
    """Vẽ bounding-boxes lên numpy image (H,W,3) uint8."""
    img = img_np.copy()
    H, W = img.shape[:2]

    scale      = max(W, H) / 800
    thickness  = max(2, int(2 * scale))
    font_scale = max(0.5, 0.6 * scale)
    font_thick = max(1, int(2 * scale))
    pad        = max(5, int(8 * scale))

    for box, lbl, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS.get(lbl, (255, 255, 255))
        name  = id_to_name.get(lbl, "?")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{name} {score:.2f}" if show_score else name
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ty = max(y1 - pad, th + pad)
        cv2.rectangle(img, (x1, ty - th - baseline),
                      (x1 + tw, ty + baseline), color, -1)
        cv2.putText(img, label_text, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thick)
    return img


# ── Single image ──────────────────────────────────────────────────────────────
def test_single_image(args, model, device, id_to_name):
    image_pil    = Image.open(args.image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image_pil).to(device)

    with torch.no_grad():
        pred = model([image_tensor])[0]

    img_np = np.array(image_pil)          # ảnh gốc chưa normalize → màu đúng

    print(f"\n── Ảnh: {args.image_path} ──────────────────────")
    print(f"Pred labels : {pred['labels'].tolist()}")
    print(f"Pred classes: {[id_to_name.get(l.item(), '?') for l in pred['labels']]}")
    print(f"Pred scores : {[round(s.item(), 3) for s in pred['scores']]}")
    print(f"Num boxes   : {len(pred['boxes'])}")

    pred_img = draw_boxes(
        img_np,
        boxes=[b.tolist() for b in pred["boxes"]],
        labels=[l.item() for l in pred["labels"]],
        scores=[s.item() for s in pred["scores"]],
        id_to_name=id_to_name,
        score_thresh=args.score_thresh,
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(pred_img)
    plt.title(f"Prediction — {args.image_path.split('/')[-1]}", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ── Webcam real-time ──────────────────────────────────────────────────────────
def test_camera(args, model, device, id_to_name):
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera ID={args.camera_id}")
        return

    print(f"📷 Camera {args.camera_id} đang chạy — nhấn Q để thoát, S để chụp ảnh")

    to_tensor     = transforms.ToTensor()
    frame_count   = 0
    snapshot_count = 0
    last_pred     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame")
            break

        frame_count += 1

        # ── Inference mỗi 2 frame để giảm lag ──────────────────────
        if frame_count % 2 == 0:
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(rgb).to(device)   # [0,1], chưa normalize
            # Model này không cần normalize ImageNet vì deploy từ PIL trực tiếp;
            # nếu model train với normalize thì thêm dòng dưới:
            # image_tensor = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(image_tensor)
            with torch.no_grad():
                last_pred = model([image_tensor])[0]

        # ── Vẽ box lên frame ────────────────────────────────────────
        if last_pred is not None:
            rgb_drawn = draw_boxes(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                boxes=[b.tolist() for b in last_pred["boxes"]],
                labels=[l.item() for l in last_pred["labels"]],
                scores=[s.item() for s in last_pred["scores"]],
                id_to_name=id_to_name,
                score_thresh=args.score_thresh,
            )
            display = cv2.cvtColor(rgb_drawn, cv2.COLOR_RGB2BGR)
        else:
            display = frame

        cv2.putText(display, f"thresh={args.score_thresh} | Q=quit S=snap",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Trash Detection — Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Thoát camera")
            break
        elif key == ord('s'):
            snapshot_path = f"snapshot_{snapshot_count:03d}.jpg"
            cv2.imwrite(snapshot_path, display)
            snapshot_count += 1
            print(f"📸 Đã lưu {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()


# ── Batch từ val dataset ──────────────────────────────────────────────────────
def test_batch(args, model, device, id_to_name, val_dataset):
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(val_loader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        preds = model(images)   # eval mode → list of dicts, không cần targets

    for i in range(len(images)):
        print(f"\n── Ảnh {i} ──────────────────────────")
        print(f"GT labels  : {targets[i]['labels'].tolist()}")
        print(f"GT classes : {[id_to_name.get(l.item(), '?') for l in targets[i]['labels']]}")
        print(f"Pred labels: {preds[i]['labels'].tolist()}")
        print(f"Pred classes: {[id_to_name.get(l.item(), '?') for l in preds[i]['labels']]}")
        print(f"Pred scores: {[round(s.item(), 3) for s in preds[i]['scores']]}")

    fig, axes = plt.subplots(len(images), 2, figsize=(14, 5 * len(images)))
    if len(images) == 1:
        axes = [axes]

    for i in range(len(images)):
        # FIX: denormalize tensor để visualize màu đúng
        img_np = denormalize(images[i])

        gt_img = draw_boxes(
            img_np,
            boxes=[b.tolist() for b in targets[i]["boxes"]],
            labels=[l.item() for l in targets[i]["labels"]],
            scores=[1.0] * len(targets[i]["labels"]),
            id_to_name=id_to_name,
            score_thresh=0.0,
            show_score=False,
        )
        pred_img = draw_boxes(
            img_np,
            boxes=[b.tolist() for b in preds[i]["boxes"]],
            labels=[l.item() for l in preds[i]["labels"]],
            scores=[s.item() for s in preds[i]["scores"]],
            id_to_name=id_to_name,
            score_thresh=args.score_thresh,
        )

        axes[i][0].imshow(gt_img);   axes[i][0].set_title(f"Ảnh {i} — Ground Truth"); axes[i][0].axis("off")
        axes[i][1].imshow(pred_img); axes[i][1].set_title(f"Ảnh {i} — Prediction");   axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = TrashDataset(root=args.data_path, split="test")
    model       = build_model(num_classes=val_dataset.get_num_classes()).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    print(f"======== best mAP: {checkpoint['best_map']:.4f}")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    id_to_name = {cat["id"] + 1: cat["name"] for cat in val_dataset.categories}

    if args.camera:
        test_camera(args, model, device, id_to_name)
    elif args.image_path:
        test_single_image(args, model, device, id_to_name)
    else:
        test_batch(args, model, device, id_to_name, val_dataset)