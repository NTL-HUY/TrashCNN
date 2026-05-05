# 🗑️ Waste Detection – FasterRCNN + ResNet50 (Transfer Learning)

Dự án phát hiện và phân loại rác thải trong ảnh sử dụng **FasterRCNN** kết hợp **ResNet50-FPN** làm backbone.  
Dataset: **TACO** (Trash Annotations in Context) – 60 category gốc được remap thành **5 superclass**.

---

| Thành phần | Trạng thái | Lý do |
|---|---|---|
| **ResNet50 + FPN backbone** | ❄️ ĐÓNG BĂNG (frozen) | Giữ nguyên đặc trưng ImageNet |
| **RPN** (Region Proposal Network) | 🔥 TRAIN | Học cách đề xuất vùng chứa rác |
| **ROI Head** (Box Predictor) | 🔥 TRAIN | Học cách phân loại 5 superclass |

---

## 📂 Cấu trúc thư mục

```
waste_detection/
│
├── download.py                  ← (có sẵn) tải TACO dataset
│
├── data/
│   ├── annotations.json         ← (có sẵn) annotation COCO format
│   ├── batch_1/                 ← ảnh gốc (tải về bằng download.py)
│   ├── batch_2/
│   ├── ...
│   └── processed/               ← ảnh đã resize (tạo bởi preprocess.py)
│       ├── batch_1/
│       └── ...
│
├── src/                         ← source code chính
│   ├── __init__.py
│   ├── config.py                ← ⚙️  MỌI cấu hình: path, hyperparams, class mapping
│   ├── dataset.py               ← 📦  TACODataset (COCO → 5 superclass)
│   ├── transforms.py            ← 🖼️  Pipeline tiền xử lý & augmentation
│   ├── model.py                 ← 🧠  Build FasterRCNN, đóng băng backbone
│   ├── trainer.py               ← 🏋️  Training & validation loop, EarlyStopping
│   ├── evaluator.py             ← 📊  Tính mAP (torchmetrics)
│   └── utils.py                 ← 🔧  DataLoader, LossLogger, visualize
│
├── scripts/
│   ├── preprocess.py            ← ✂️  Resize ảnh TACO → data/processed/
│   └── visualize.py             ← 🎨  Vẽ bounding box lên ảnh kết quả
│
├── checkpoints/                 ← checkpoint được lưu tại đây
├── logs/                        ← training_log.json, loss_curve.png
│
├── main.py                      ← 🚀  Entry point (train / eval / predict)
└── requirements.txt
```

---

## 🏷️ Mapping 60 Class TACO → 5 Superclass

| Superclass | Index | Ví dụ category TACO |
|---|:---:|---|
| **plastic** | 1 | Bottle, Plastic bag, Straw, Styrofoam, Cup, Lid... |
| **paper**   | 2 | Paper, Carton, Newspaper, Cardboard, Paper bag... |
| **metal**   | 3 | Can, Aerosol, Aluminium foil, Scrap metal, Battery... |
| **glass**   | 4 | Broken glass, Glass bottle, Glass jar... |
| **other**   | 5 | Cigarette, Rope, Shoe, Unlabeled litter... |
| background  | 0 | *(reserved bởi FasterRCNN)* |

> Mapping đầy đủ được định nghĩa trong `src/config.py → TACO_TO_SUPERCLASS`.  
> Bất kỳ category TACO nào không có trong dict sẽ tự động về `other`.

---

## ⚙️ Cài đặt môi trường

### 1. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 2. Cài PyTorch (chọn đúng phiên bản CUDA)

Truy cập https://pytorch.org/get-started/locally/ để lấy lệnh cài phù hợp GPU.  
Ví dụ CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy

### Bước 1 – Tải TACO dataset

```bash
python download.py
```
Ảnh được tải về `data/batch_1/`, `data/batch_2/`, ...

---

### Bước 2 – Tiền xử lý ảnh *(chạy 1 lần)*

TACO chứa ảnh rất lớn (2000–6000px). Script này resize xuống max 800px và lưu vào `data/processed/`:

```bash
python scripts/preprocess.py
```

Tùy chọn nâng cao:

```bash
python scripts/preprocess.py \
  --max-size 800 \     # cạnh dài tối đa (px)
  --min-size 600 \     # cạnh ngắn tối thiểu (px)
  --workers 8 \        # số thread song song
  --quality 90         # JPEG quality
```

> Sau bước này `data/processed/` sẽ có cùng cấu trúc với `data/` nhưng ảnh nhỏ hơn đáng kể.

---

### Bước 3 – Training

```bash
python main.py train
```

Tùy chỉnh:

```bash
python main.py train \
  --epochs 30 \         # số epoch
  --batch-size 2 \      # batch size (nhỏ vì ảnh lớn)
  --workers 4 \         # DataLoader workers
  --patience 7 \        # early stopping patience
  --save-every 5 \      # lưu checkpoint mỗi 5 epoch
  --print-freq 20       # in log mỗi 20 batch
```

Resume từ checkpoint:

```bash
python main.py train --resume checkpoints/epoch_015.pth
```

**Output:**
- `checkpoints/best.pth` – checkpoint tốt nhất (val loss thấp nhất)
- `checkpoints/epoch_XXX.pth` – checkpoint theo epoch
- `logs/training_log.json` – lịch sử loss
- `logs/loss_curve.png` – đồ thị loss

---

### Bước 4 – Đánh giá (mAP)

```bash
python main.py eval --checkpoint checkpoints/best.pth
```

Kết quả mẫu:
```
=======================================================
  KẾT QUẢ ĐÁNH GIÁ (COCO mAP)
=======================================================
  mAP  @[0.50:0.95] : 0.3241
  mAP  @[0.50]      : 0.5812
  mAP  @[0.75]      : 0.3107
  mAR  (max 100)    : 0.4520
-------------------------------------------------------
  mAP per class:
    plastic     : 0.4102
    paper       : 0.3218
    metal       : 0.2901
    glass       : 0.2544
    other       : 0.2441
=======================================================
```

---

### Bước 5 – Inference ảnh mới

```bash
python main.py predict \
  --checkpoint checkpoints/best.pth \
  --image /đường/dẫn/tới/ảnh.jpg \
  --output output/prediction.jpg \
  --score-thresh 0.3
```

---

### Visualize nhiều ảnh từ test set

```bash
python scripts/visualize.py \
  --checkpoint checkpoints/best.pth \
  --n 20 \
  --output-dir output/visualize
```

---

## 🧪 Giải thích kỹ thuật

### Tại sao không fine-tune backbone?

| | Chỉ train head | Fine-tuning toàn bộ |
|---|---|---|
| **Tốc độ train** | Nhanh hơn 3–5× | Chậm |
| **Nguy cơ overfitting** | Thấp | Cao nếu dataset nhỏ |
| **Yêu cầu VRAM** | Thấp hơn | Cao hơn |
| **mAP kỳ vọng** | Tốt (đủ cho bài toán này) | Có thể tốt hơn nếu đủ data |

Với TACO dataset (~1500 ảnh), **Feature Extraction** là lựa chọn hợp lý để tránh overfitting.

---

### Pipeline xử lý ảnh

```
Ảnh gốc (3000px+)
      │
      ▼ preprocess.py (offline, 1 lần)
Ảnh đã resize (≤800px, lưu vào disk)
      │
      ▼ transforms.py (online, mỗi epoch)
  [Train]  Resize → RandomHFlip → RandomBrightness → ToTensor → Normalize(ImageNet)
  [Val]    Resize → ToTensor → Normalize(ImageNet)
      │
      ▼
Tensor [C, H, W] → FasterRCNN
```

---

### Cấu hình quan trọng trong `src/config.py`

```python
IMAGE_MAX_SIZE  = 800    # Resize xuống nếu cạnh dài > giá trị này
IMAGE_MIN_SIZE  = 600    # Resize lên nếu cạnh ngắn < giá trị này
FREEZE_BACKBONE = True   # True = Feature Extraction
BATCH_SIZE      = 2      # Tăng nếu VRAM đủ lớn
NUM_EPOCHS      = 30
LEARNING_RATE   = 0.005
NUM_CLASSES     = 6      # 5 superclass + 1 background
```

---

## 🐛 Troubleshooting

**CUDA out of memory:**
- Giảm `BATCH_SIZE` trong `config.py` xuống 1
- Giảm `IMAGE_MAX_SIZE` xuống 600 hoặc 640

**Không tìm thấy ảnh khi train:**
- Đảm bảo đã chạy `download.py` và `scripts/preprocess.py`
- Kiểm tra `data/processed/` có ảnh chưa

**mAP thấp:**
- Train thêm epoch (`--epochs 50`)
- Hạ `--score-thresh` xuống 0.2 khi eval
- Kiểm tra class distribution bằng `dataset.get_class_distribution()`

---

## 📦 Dependency chính

| Thư viện | Vai trò |
|---|---|
| `torch` + `torchvision` | FasterRCNN, ResNet50, training |
| `torchmetrics` | Tính mAP chuẩn COCO |
| `Pillow` | Đọc/resize ảnh |
| `opencv-python` | Vẽ bounding box |
| `matplotlib` | Đồ thị loss |
