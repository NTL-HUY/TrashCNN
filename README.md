# TrashCNN – Phát hiện rác với FasterRCNN

Hệ thống phát hiện và phân loại rác trong ảnh sử dụng **FasterRCNN** trên tập dữ liệu [TACO](http://tacodataset.org/).  
Kiến trúc (ResNet50, FPN, RPN, ROI Head) được implement bằng PyTorch

---

## Kiến trúc

```
Image
  │
  ▼
ResNet50Backbone   (Bottleneck blocks, Kaiming init)
  │  layer1 → 256 ch  │  layer2 → 512 ch
  │  layer3 → 1024 ch │  layer4 → 2048 ch
  ▼
FPN                (Feature Pyramid Network – 5 levels P2–P6, 256 ch mỗi level)
  ▼
RPN                (Region Proposal Network – 3 anchor ratios/cell, NMS)
  ▼
ROI Align          (multi-level, gán proposal → FPN level theo diện tích)
  ▼
BoxHead            (2× FC 1024 + Dropout 0.3)
  ▼
BoxPredictor       (cls: 6 classes, reg: 6×4 deltas)
  ▼
Output: boxes, labels, scores
```

### 5 Superclass (+ background)

| Index | Tên | Ví dụ |
|-------|-----|-------|
| 0 | background | — |
| 1 | plastic | chai nhựa, túi nilon, ống hút |
| 2 | paper | hộp carton, cốc giấy, báo |
| 3 | metal | lon nước, hộp sắt, nắp chai |
| 4 | glass | chai thuỷ tinh, mảnh kính |
| 5 | other | pin, thuốc lá, dây thừng |

---

## Cài đặt

```bash
# Clone repo
git clone https://github.com/your-username/TrashCNN.git
cd TrashCNN

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Cài dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics pillow matplotlib opencv-python tensorboard
```

> **Yêu cầu:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## Cấu trúc thư mục

```
TrashCNN/
├── data/
│   ├── annotations.json          ← file annotation TACO (COCO format)
│   ├── batch_1/                  ← ảnh gốc tải về từ Flickr
│   ├── batch_2/
│   └── processed/                ← ảnh sau resize (tạo bởi preprocess.py)
│
├── src/
│   ├── config.py                 ← toàn bộ hyperparameter và đường dẫn
│   ├── model.py                  ← FasterRCNN
│   ├── dataset.py                ← TACODataset (COCO format loader)
│   ├── transforms.py             ← augmentation pipeline
│   ├── trainer.py                ← train/val loop
│   ├── evaluator.py              ← tính mAP (torchmetrics)
│   └── utils.py                  ← device, logger, plot
│
├── scripts/
│   ├── preprocess.py             ← resize ảnh → data/processed/
│   └── visualize.py              ← vẽ prediction lên ảnh
│
├── checkpoints/                  ← model checkpoint (.pth)
├── logs/
│   ├── training_log.json         ← loss theo epoch
│   ├── loss_curve.png            ← đồ thị loss
│   └── tb/                       ← TensorBoard event files
│
├── main.py                       ← CLI entry point
└── README.md
```

---

## Luồng sử dụng

### Bước 1 – Tải dataset TACO

```bash
python data/download.py --dataset_path data/annotations.json
```

### Bước 2 – Tiền xử lý ảnh

```bash
python scripts/preprocess.py
```

Resize ảnh gốc (3–5K px) về `IMAGE_MAX_SIZE=800px`, lưu vào `data/processed/`.

### Bước 3 – Train

```bash
python main.py train
```

Tuỳ chọn:

```bash
python main.py train \
  --epochs 60 \
  --batch-size 2 \
  --patience 10 \
  --save-every 5 \
  --print-freq 20
```

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--epochs` | 60 | Số epoch tối đa |
| `--batch-size` | 2 | Batch size (tăng nếu VRAM ≥ 8GB) |
| `--patience` | 10 | Early stopping patience |
| `--save-every` | 5 | Lưu checkpoint mỗi N epoch |
| `--print-freq` | 20 | In log mỗi N batch |
| `--resume` | — | Resume từ checkpoint |
| `--no-tensorboard` | — | Tắt TensorBoard logging |

Resume từ checkpoint:

```bash
python main.py train --resume checkpoints/epoch_020.pth --epochs 60
```

### Bước 4 – Theo dõi training (TensorBoard)

```python
# Trong Jupyter / Colab
%load_ext tensorboard
%tensorboard --logdir logs/tb
```

Hoặc terminal:

```bash
tensorboard --logdir logs/tb
# Mở trình duyệt: http://localhost:6006
```

Các metric hiển thị trên TensorBoard:

| Tag | Mô tả |
|-----|-------|
| `Train/total` | Total train loss theo epoch |
| `Train/loss_objectness` | RPN objectness loss |
| `Train/loss_rpn_box_reg` | RPN box regression loss |
| `Train/loss_classifier` | ROI classification loss |
| `Train/loss_box_reg` | ROI box regression loss |
| `Train/learning_rate` | LR hiện tại (ReduceLROnPlateau) |
| `Val/total` | Total validation loss |
| `Train/batch_total` | Loss từng batch (granular) |

### Bước 5 – Đánh giá

```bash
python main.py eval --checkpoint checkpoints/best.pth --score-thresh 0.05
```

Output:

```
======================================================
  KẾT QUẢ ĐÁNH GIÁ (COCO mAP)
======================================================
  mAP  @[0.50:0.95] : 0.1823
  mAP  @[0.50]      : 0.3210
  mAP  @[0.75]      : 0.1540
  mAR  (max 100)    : 0.2890
------------------------------------------------------
  mAP per class:
    plastic     : 0.2100
    paper       : 0.1750
    metal       : 0.1900
    glass       : 0.1200
    other       : 0.1165
======================================================
```

Kết quả lưu vào `logs/eval_results.json`.

> **Nếu mAP = 0.0:** hạ threshold xuống `--score-thresh 0.01` để xem có prediction không.

### Bước 6 – Inference ảnh đơn

```bash
python main.py predict \
  --checkpoint checkpoints/best.pth \
  --image path/to/image.jpg \
  --score-thresh 0.3 \
  --output output/result.jpg
```

---

## Cấu hình (config.py)

Tất cả hyperparameter tập trung tại `src/config.py`.  
Các thông số quan trọng nhất:

```python
# Training
LEARNING_RATE = 0.001    # Adam LR
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 60

# LR Scheduler (ReduceLROnPlateau)
LR_PATIENCE   = 5        # giảm LR sau N epoch val loss không cải thiện
LR_FACTOR     = 0.5      # nhân LR với 0.5
LR_MIN        = 1e-6

# Regularisation
DROPOUT_RATE  = 0.3      # dropout trong BoxHead

# Image size
IMAGE_MAX_SIZE = 800
IMAGE_MIN_SIZE = 600
```

---

## Dependencies

| Package | Phiên bản | Dùng cho |
|---------|-----------|---------|
| `torch` | ≥ 2.0 | core |
| `torchvision` | ≥ 0.15 | ops (nms, roi_align, box_iou) |
| `torchmetrics` | ≥ 1.0 | tính mAP |
| `Pillow` | ≥ 9.0 | đọc ảnh |
| `numpy` | ≥ 1.23 | xử lý array |
| `matplotlib` | ≥ 3.6 | vẽ loss curve |
| `opencv-python` | ≥ 4.7 | vẽ bounding box |
| `tensorboard` | ≥ 2.12 | visualize training |

---