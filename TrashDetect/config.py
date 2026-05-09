from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    OUTPUT_DIR      = "../FolderDemo/output_stats"
    # SPLITS          = ["train", "valid", "test"]
    DATA_GLASS_PATH = r"D:\Projects\Workspace\Coding\Dataset\detect glass.v1i.coco"
    DATA_TACO_PATH = r"D:\Projects\Workspace\Coding\Dataset\TACO dataset.v1i.coco"
    SPLITS: list = field(default_factory=lambda: ["train", "valid", "test"])
    SEED: int = 42

    # Categories muốn GIỮ LẠI trong TACO (tên sau khi đã chuẩn hóa tên)
    # None = giữ tất cả
    KEEP_CATEGORIES: Optional[list] = field(
        default_factory=lambda: ["plastic", "cardboard", "paper", "metal", "glass"]
    )

    # Undersample class trội quá mức – chỉ áp dụng cho split train
    # key = tên class, value = số annotation tối đa
    UNDERSAMPLE: dict = field(
        default_factory=lambda: {"plastic": 500}
    )

    # Giới hạn bbox tối thiểu (pixel) để coi là hợp lệ
    MIN_BBOX_W: float = 2.0
    MIN_BBOX_H: float = 2.0
    MIN_BBOX_AREA: float = 16.0  # pixel²

    # Cho phép bbox vượt biên ảnh không? False = clip lại
    CLIP_BBOX: bool = True
