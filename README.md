# 🗑️ Waste Detection & Classification

Faster R-CNN với ResNet-50 + FPN train từ đầu (không dùng pretrained weights) để phát hiện và phân loại rác thải thành 5 lớp: **plastic, metal, paper, glass, other**.

---

## 📁 Cấu trúc Project

```
waste_detection/
├── dataset.py          # Download TACO, filter 5 class, DataLoader
├── model.py            # ResNet-50 + FPN + Faster R-CNN (from scratch)
├── utils.py            # mAP, Checkpoint, EarlyStopping, LR Scheduler
├── train.py            # Training loop với TensorBoard
├── deploy.py           # Inference trên ảnh/video
├── requirements.txt
├── data/
│   └── taco/
│       ├── images/
│       ├── annotations.json
│       └── annotations_filtered.json
├── weights/            # (tự tạo khi train)
│   ├── best_model.pth
│   └── last_model.pth
└── runs/               # (tự tạo khi train)
    └── train/          # TensorBoard logs
```

---

## ⚡ Quick Start (Google Colab T4)

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Download Dataset + Train

```bash
# Download TACO dataset và bắt đầu train
python train.py \
  --data_dir data/taco \
  --num_epochs 50 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.01 \
  --amp \
  --optimizer sgd \
  --warmup_epochs 3 \
  --grad_clip 5.0 \
  --weights_dir weights \
  --log_dir runs/train
```

### 3. Theo dõi TensorBoard

```bash
tensorboard --logdir runs/train
```

### 4. Inference

```bash
# Single image
python deploy.py \
  --source path/to/image.jpg \
  --weights weights/best_model.pth \
  --score_thresh 0.4

# Folder ảnh
python deploy.py \
  --source path/to/images/ \
  --weights weights/best_model.pth \
  --output_dir predictions

# Video
python deploy.py \
  --source path/to/video.mp4 \
  --weights weights/best_model.pth \
  --save_video
```

---

## 🏗️ Kiến trúc Model

```
Input Image
    │
    ▼
GeneralizedRCNNTransform  (resize 800~1333, normalize)
    │
    ▼
ResNet-50 Backbone (từ scratch)
  └─ Stem: Conv7x7 → BN → ReLU → MaxPool
  └─ Layer1 (C2): 3x Bottleneck, stride=1  → 256ch
  └─ Layer2 (C3): 4x Bottleneck, stride=2  → 512ch
  └─ Layer3 (C4): 6x Bottleneck, stride=2  → 1024ch
  └─ Layer4 (C5): 3x Bottleneck, stride=2  → 2048ch
    │
    ▼
Feature Pyramid Network (FPN)
  └─ P2 (stride 4)  → 256ch
  └─ P3 (stride 8)  → 256ch
  └─ P4 (stride 16) → 256ch
  └─ P5 (stride 32) → 256ch
  └─ P6 (pooling)   → 256ch
    │
    ▼
Region Proposal Network (RPN)
  └─ Anchors: sizes=(32,64,128,256,512), ratios=(0.5,1.0,2.0)
  └─ RPNHead → objectness scores + bbox deltas
  └─ NMS → ~2000 proposals (train) / 1000 (test)
    │
    ▼
RoI Align (Multi-Scale, 7×7)
    │
    ▼
TwoMLPHead (FC 12544 → 1024 → 1024)
    │
    ▼
FastRCNNPredictor
  └─ Classification: 6 classes (+ background)
  └─ Regression    : 6 × 4 bbox deltas
    │
    ▼
Output: boxes, labels, scores
```

---

## 🎛️ Training Flags

### train.py

| Flag | Default | Mô tả |
|------|---------|-------|
| `--data_dir` | `data/taco` | Thư mục chứa dataset |
| `--ann_file` | auto | Path đến file annotations đã filter |
| `--skip_download` | False | Bỏ qua download (dùng data có sẵn) |
| `--num_epochs` | 50 | Số epoch train |
| `--batch_size` | 4 | Batch size |
| `--num_workers` | 4 | Số worker cho DataLoader |
| `--seed` | 42 | Random seed |
| `--resume` | None | Resume từ checkpoint |
| `--grad_accum_steps` | 1 | Gradient accumulation |
| `--grad_clip` | 5.0 | Max gradient norm |
| `--amp` | False | Mixed precision (AMP) |
| `--val_every` | 1 | Validate mỗi N epoch |
| `--optimizer` | sgd | `sgd` hoặc `adamw` |
| `--lr` | 0.01 | Learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight_decay` | 1e-4 | L2 regularization |
| `--warmup_epochs` | 3 | Warmup epochs |
| `--min_lr` | 1e-6 | LR tối thiểu (cosine end) |
| `--min_size` | 800 | Resize ngắn nhất |
| `--max_size` | 1333 | Resize dài nhất |
| `--early_stop_patience` | 15 | Patience cho early stopping |
| `--early_stop_metric` | mAP_50 | Metric theo dõi |
| `--weights_dir` | weights | Thư mục lưu model |
| `--log_dir` | runs/train | TensorBoard log dir |
| `--save_metric` | mAP_50 | Metric chọn best model |

### deploy.py

| Flag | Default | Mô tả |
|------|---------|-------|
| `--source` | *(required)* | Ảnh / folder / video |
| `--weights` | weights/best_model.pth | Path đến model |
| `--score_thresh` | 0.4 | Ngưỡng confidence |
| `--nms_thresh` | 0.5 | NMS IoU threshold |
| `--max_detections` | 100 | Max detections/ảnh |
| `--device` | auto | `cuda` hoặc `cpu` |
| `--output_dir` | predictions | Thư mục lưu kết quả |
| `--save_json` | False | Lưu JSON predictions |
| `--save_video` | False | Lưu video đã annotate |
| `--no_save_image` | False | Không lưu ảnh kết quả |
| `--show` | False | Hiển thị cửa sổ preview |
| `--hide_labels` | False | Ẩn tên class |
| `--hide_conf` | False | Ẩn confidence score |
| `--line_width` | 2 | Độ dày bounding box |
| `--font_size` | 14 | Cỡ chữ nhãn |
| `--num_workers` | 0 | DataLoader workers |

### dataset.py (standalone)

```bash
# Chỉ download dataset
python dataset.py --data_dir data/taco
```

---

## 📊 Classes & Colors

| Class      | ID | Color (RGB) |
|------------|----|-------------|
| background | 0  | gray |
| plastic    | 1  | orange (255,100,50) |
| metal      | 2  | blue (50,150,255) |
| paper      | 3  | green (80,200,80) |
| glass      | 4  | purple (150,80,220) |
| other      | 5  | brown (200,150,50) |
---

## 🚀 Recommended Settings cho T4 GPU (Colab)

```bash
python train.py \
  --data_dir data/taco \
  --num_epochs 60 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.01 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --min_lr 1e-6 \
  --amp \
  --grad_clip 5.0 \
  --grad_accum_steps 2 \
  --val_every 2 \
  --early_stop_patience 20 \
  --save_metric mAP_50 \
  --weights_dir weights \
  --log_dir runs/train
```

> **Ghi chú:**
> - `--amp` giảm VRAM ~40%, tăng tốc ~30% trên T4
> - `--grad_accum_steps 2` → effective batch = 8 mà không tốn thêm VRAM
> - `--val_every 2` tiết kiệm thời gian; mAP tính tốn nhiều hơn loss

---

## 📈 Metrics Hiển thị

Mỗi epoch sẽ in ra:
```
Epoch  5/50 | Loss: 0.8234 | mAP: 0.3412 | mAP@50: 0.5812 | mAP@75: 0.2901 | LR: 9.50e-03 | Time: 4m 23s ← BEST
  Per-class: AP_plastic: 0.612 | AP_metal: 0.534 | AP_paper: 0.489 | AP_glass: 0.398 | AP_other: 0.471
```

TensorBoard theo dõi:
- `train_epoch/` — total loss, cls loss, reg loss, rpn loss, lr
- `train_step/` — step-level losses & lr
- `val/` — mAP, mAP_50, mAP_75, per-class AP
- `summary/` — overlay chart loss + mAP
- `test/` — final test set metrics

---

## 📝 Notes

- **Không dùng pretrained weights**: Mọi layer khởi tạo từ Kaiming/Xavier initialization
- **Augmentation**: horizontal flip, brightness/contrast/saturation/hue jitter
- **Scheduler**: Linear warmup → Cosine annealing
- **AMP**: `torch.cuda.amp` để train ổn định trên T4
- **Gradient clipping**: tránh gradient explosion khi train từ scratch
- **Dataset split**: 80% train / 10% val / 10% test (cố định với seed)
