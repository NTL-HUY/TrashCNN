"""
config.py – Cấu hình toàn bộ dự án
====================================
Một chỗ duy nhất để thay đổi đường dẫn, siêu tham số,
và mapping từ 60 class TACO → 5 superclass.
"""

import os

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations.json")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")      # ảnh đã resize
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR         = os.path.join(ROOT_DIR, "logs")

# ---------------------------------------------------------------------------
# Tiền xử lý ảnh
# ---------------------------------------------------------------------------
IMAGE_MAX_SIZE  = 800
IMAGE_MIN_SIZE  = 600
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE   = 2
NUM_EPOCHS   = 50
NUM_WORKERS  = 4
FREEZE_BACKBONE     = False
PRETRAINED_BACKBONE = True
LEARNING_RATE          = 0.005
BACKBONE_LEARNING_RATE = 0.0005
MOMENTUM               = 0.9
WEIGHT_DECAY           = 0.0005
LR_STEP_SIZE = 8
LR_GAMMA     = 0.5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
NUM_CLASSES = 6   # 5 superclass + 1 background (index 0)

# ---------------------------------------------------------------------------
# Superclass mapping
# ---------------------------------------------------------------------------
SUPERCLASS_NAMES = {
    0: "background",
    1: "plastic",
    2: "paper",
    3: "metal",
    4: "glass",
    5: "other",
}

SUPERCLASS_COLORS = {
    0: (128, 128, 128),   # background – xám
    1: (0,   165, 255),   # plastic    – cam
    2: (0,   255,   0),   # paper      – xanh lá
    3: (255,  50,  50),   # metal      – đỏ
    4: (255, 255,   0),   # glass      – vàng
    5: (200,   0, 200),   # other      – tím
}

# ---------------------------------------------------------------------------
# Mapping: tên category TACO (lowercase) → superclass index
# ---------------------------------------------------------------------------
TACO_TO_SUPERCLASS: dict[str, int] = {
    # 1: plastic
    "other plastic bottle": 1,
    "clear plastic bottle": 1,
    "plastic bottle cap": 1,
    "carded blister pack": 1,
    "disposable plastic cup": 1,
    "foam cup": 1,
    "other plastic cup": 1,
    "plastic lid": 1,
    "other plastic": 1,
    "plastic film": 1,
    "six pack rings": 1,
    "garbage bag": 1,
    "other plastic wrapper": 1,
    "single-use carrier bag": 1,
    "polypropylene bag": 1,
    "crisp packet": 1,
    "spread tub": 1,
    "tupperware": 1,
    "disposable food container": 1,
    "foam food container": 1,
    "other plastic container": 1,
    "plastic glooves": 1,
    "plastic utensils": 1,
    "squeezable tube": 1,
    "plastic straw": 1,
    "styrofoam piece": 1,

    # 2: paper
    "toilet tube": 2,
    "other carton": 2,
    "egg carton": 2,
    "drink carton": 2,
    "corrugated carton": 2,
    "meal carton": 2,
    "pizza box": 2,
    "paper cup": 2,
    "magazine paper": 2,
    "tissues": 2,
    "wrapping paper": 2,
    "normal paper": 2,
    "paper bag": 2,
    "plastified paper bag": 2,
    "paper straw": 2,

    # 3: metal
    "aluminium foil": 3,
    "aluminium blister pack": 3,
    "metal bottle cap": 3,
    "food can": 3,
    "aerosol": 3,
    "drink can": 3,
    "metal lid": 3,
    "pop tab": 3,
    "scrap metal": 3,

    # 4: glass
    "glass bottle": 4,
    "broken glass": 4,
    "glass cup": 4,
    "glass jar": 4,

    # 5: other
    "battery": 5,
    "food waste": 5,
    "rope & strings": 5,
    "shoe": 5,
    "unlabeled litter": 5,
    "cigarette": 5,
}