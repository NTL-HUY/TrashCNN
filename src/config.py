"""
config.py
====================================
Model build from scratch: ResNet50 + FPN + RPN + ROI Head
"""

import os

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations.json")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR         = os.path.join(ROOT_DIR, "logs")

# ---------------------------------------------------------------------------
# Tiền xử lý ảnh
# ---------------------------------------------------------------------------
IMAGE_MAX_SIZE = 800
IMAGE_MIN_SIZE = 600
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]

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
BATCH_SIZE  = 2
NUM_EPOCHS  = 60
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Optimizer – Adam
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4

# ReduceLROnPlateau: giảm LR khi val loss plateau
LR_PATIENCE = 5
LR_FACTOR   = 0.5
LR_MIN      = 1e-6

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
NUM_CLASSES = 6   # 5 superclass + 1 background

# ResNet50 backbone
RESNET_LAYERS = [3, 4, 6, 3]   # số Bottleneck block tại layer1–4

# FPN
FPN_OUT_CHANNELS = 256

# RPN
RPN_ANCHOR_SIZES         = (32, 64, 128, 256, 512)
RPN_ANCHOR_RATIOS        = (0.5, 1.0, 2.0)
RPN_PRE_NMS_TOP_N        = {"training": 2000, "testing": 1000}
RPN_POST_NMS_TOP_N       = {"training": 1000, "testing": 300}
RPN_NMS_THRESH           = 0.7
RPN_FG_IOU_THRESH        = 0.7
RPN_BG_IOU_THRESH        = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION    = 0.5

# ROI Head
ROI_BOX_SCORE_THRESH      = 0.05
ROI_NMS_THRESH            = 0.5
ROI_DETECTIONS_PER_IMG    = 100
ROI_FG_IOU_THRESH         = 0.5
ROI_BG_IOU_THRESH_HI      = 0.5
ROI_BG_IOU_THRESH_LO      = 0.0
ROI_BATCH_SIZE_PER_IMAGE  = 512
ROI_POSITIVE_FRACTION     = 0.25
ROI_POOLER_OUTPUT_SIZE    = 7
ROI_POOLER_SAMPLING_RATIO = 2

# Dropout
DROPOUT_RATE = 0.3

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
AUGMENT_HFLIP_PROB  = 0.5
AUGMENT_VFLIP_PROB  = 0.2
AUGMENT_BRIGHTNESS  = 0.3
AUGMENT_CONTRAST    = 0.3
AUGMENT_SATURATION  = 0.3
AUGMENT_HUE         = 0.1

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