"""
preprocess.py  –  Pipeline tiền xử lý COCO dataset
=====================================================
Các bước:
  1. Filter / remove categories không cần
  2. Validate & clean bbox (loại bbox invalid)
  3. Reindex category ID về 0-based liên tục
  4. Merge Glass dataset vào TACO để bổ sung class 'glass'
  5. Undersample class 'plastic' (dominant) xuống ngưỡng cân bằng
  6. Loại ảnh không còn annotation
  7. Xuất file COCO đã xử lý + báo cáo thống kê trước/sau

Usage:
    python preprocess.py               # dùng Config mặc định
"""
import argparse
import copy
import json
import os
import random
import shutil
import warnings
from collections import Counter, defaultdict
from typing import Optional
from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_coco(base_path: str, split: str) -> Optional[dict]:
    path = os.path.join(base_path, split, "_annotations.coco.json")
    if not os.path.exists(path):
        warnings.warn(f"[SKIP] Not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded  {path}")
    return data

def save_coco(
    coco: dict,
    base_path: str,
    split: str,
    filename: str = "_annotations.cleaned.coco.json"
):
    out_dir = os.path.join(base_path, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"  Saved   {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – FILTER CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

def filter_category_ids(coco: dict, remove_ids: set) -> dict:
    """Xóa category theo ID và annotation tương ứng."""
    coco["categories"] = [c for c in coco["categories"] if c["id"] not in remove_ids]
    coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] not in remove_ids]
    return coco

def keep_categories_by_name(coco: dict, keep_names: list) -> dict:
    """Chỉ giữ các category có tên trong keep_names."""
    keep_set = set(keep_names)
    remove_ids = {c["id"] for c in coco["categories"] if c["name"] not in keep_set}
    return filter_category_ids(coco, remove_ids)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – VALIDATE & CLEAN BBOX
# ─────────────────────────────────────────────────────────────────────────────

def get_image_sizes(coco: dict) -> dict:
    """Trả về dict image_id -> (width, height)."""
    return {img["id"]: (img.get("width", 0), img.get("height", 0))
            for img in coco.get("images", [])}


def validate_bbox(
    bbox: list,
    img_w: float,
    img_h: float,
    min_w: float = CFG.MIN_BBOX_W,
    min_h: float = CFG.MIN_BBOX_H,
    min_area: float = CFG.MIN_BBOX_AREA,
    clip: bool = CFG.CLIP_BBOX,
) -> Optional[list]:
    """
    Kiểm tra và sửa bbox [x, y, w, h] (COCO format).
    Trả về bbox đã clean, hoặc None nếu không thể cứu.

    Các trường hợp xử lý:
      - w hoặc h <= 0         → None (discard)
      - x/y âm               → clip về 0
      - bbox vượt biên ảnh   → clip nếu clip=True
      - w/h sau clip < min_w/h → None (discard)
      - area < min_area       → None (discard)
    """
    if not bbox or len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]

    # w, h không dương
    if w <= 0 or h <= 0:
        return None

    # Tọa độ âm
    if x < 0:
        w += x
        x = 0.0
    if y < 0:
        h += y
        y = 0.0

    # Vượt biên ảnh
    if clip and img_w > 0 and img_h > 0:
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        w = x2 - x
        h = y2 - y

    # Kích thước tối thiểu
    if w < min_w or h < min_h:
        return None

    # Diện tích tối thiểu
    if w * h < min_area:
        return None

    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def clean_bboxes(coco: dict) -> tuple[dict, dict]:
    """
    Áp dụng validate_bbox cho toàn bộ annotation.
    Trả về (coco đã clean, report dict).
    """
    img_sizes = get_image_sizes(coco)
    report = {"total": 0, "removed": 0, "clipped": 0}
    clean_anns = []

    for ann in coco["annotations"]:
        report["total"] += 1
        bbox_orig = ann.get("bbox")
        img_w, img_h = img_sizes.get(ann["image_id"], (0, 0))

        cleaned = validate_bbox(bbox_orig, img_w, img_h)
        if cleaned is None:
            report["removed"] += 1
            continue

        if cleaned != bbox_orig:
            report["clipped"] += 1

        ann = {**ann, "bbox": cleaned}

        # Cập nhật area nếu có
        if "area" in ann:
            ann["area"] = round(cleaned[2] * cleaned[3], 4)

        clean_anns.append(ann)

    coco["annotations"] = clean_anns
    return coco, report

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – MERGE GLASS DATASET INTO TACO
# ─────────────────────────────────────────────────────────────────────────────

def _max_id(items: list, key: str = "id") -> int:
    return max((item[key] for item in items), default=0)

def merge_glass_into_taco(taco: dict, glass: dict) -> tuple[dict, int]:
    """
    Gộp Glass COCO vào TACO COCO.

    Chiến lược:
      - Tìm (hoặc tạo) category 'glass' trong TACO
      - Offset image_id và annotation_id của Glass để tránh trùng
      - Đổi category_id của tất cả annotation Glass về glass_id trong TACO
      - Thêm images + annotations vào TACO

    Trả về (taco_merged, n_added_annotations).
    """
    # ── tìm glass category trong TACO ──────────────────────────────────────
    glass_cat = next((c for c in taco["categories"] if c["name"] == "glass"), None)
    if glass_cat is None:
        new_cat_id = _max_id(taco["categories"]) + 1
        glass_cat = {"id": new_cat_id, "name": "glass", "supercategory": "waste"}
        taco["categories"].append(glass_cat)
    taco_glass_id = glass_cat["id"]

    # ── offset IDs lấy id cuối cùng của taco ─────────────────────────────────────────────────────────
    img_offset = _max_id(taco["images"]) + 1
    ann_offset = _max_id(taco["annotations"]) + 1

    # ── build glass thay đổi lại id của img glass theo taco ────────────────────────
    glass_img_id2new = {}
    new_images = []
    for img in glass.get("images", []):
        new_id = img["id"] + img_offset
        glass_img_id2new[img["id"]] = new_id
        new_filename = "glass_" + img["file_name"]
        new_img = {
            **img,
            "id": new_id,
            "file_name": new_filename
        }
        new_images.append(new_img)

    # ── remap annotations thay đổi ann_id, img_id,category_id  ───────────────────────────────────────────────────
    new_anns = []
    for ann in glass.get("annotations", []):
        new_ann = {
            **ann,
            "id":          ann["id"] + ann_offset,
            "image_id":    glass_img_id2new.get(ann["image_id"], ann["image_id"] + img_offset),
            "category_id": taco_glass_id,
        }
        new_anns.append(new_ann)

    taco["images"]      = taco.get("images", [])      + new_images
    taco["annotations"] = taco.get("annotations", []) + new_anns

    return taco, len(new_anns)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – UNDERSAMPLE DOMINANT CLASS
# ─────────────────────────────────────────────────────────────────────────────

def undersample_class(
    coco: dict,
    class_name: str,
    max_count: int,
    seed: int = 42,
) -> tuple[dict, int]:
    """
    Giới hạn số annotation của class_name xuống max_count.
    Trả về (coco, n_removed).
    """
    target_id = next(
        (c["id"] for c in coco["categories"] if c["name"] == class_name), None
    )
    if target_id is None:
        warnings.warn(f"[undersample] class '{class_name}' not found – skipped.")
        return coco, 0

    target_anns = [a for a in coco["annotations"] if a["category_id"] == target_id]
    other_anns  = [a for a in coco["annotations"] if a["category_id"] != target_id]

    if len(target_anns) <= max_count:
        return coco, 0   # không cần undersample

    random.seed(seed)
    sampled = random.sample(target_anns, max_count)
    n_removed = len(target_anns) - max_count

    coco["annotations"] = other_anns + sampled
    return coco, n_removed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – REMOVE UNANNOTATED IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def filter_images_without_annotations(coco: dict) -> tuple[dict, int]:
    used_ids = {a["image_id"] for a in coco["annotations"]}
    before = len(coco["images"])
    coco["images"] = [img for img in coco["images"] if img["id"] in used_ids]
    return coco, before - len(coco["images"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – UTILITY
# ─────────────────────────────────────────────────────────────────────────────



def category_stats(coco: dict) -> Counter:
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    return Counter(id2name.get(a["category_id"], "?") for a in coco["annotations"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats_line(label: str, coco: dict):
    stats = category_stats(coco)
    total = sum(stats.values())
    cats  = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    print(f"    [{label:12s}] images={len(coco['images']):,}  anns={total:,}  | {cats}")


def preprocess_split(
    split: str,
    cfg: Config = CFG,
) -> Optional[dict]:

    print(f"\n{'─'*65}")
    print(f"  SPLIT: {split.upper()}")
    print(f"{'─'*65}")

    # ── Load ────────────────────────────────────────────────────────────────
    taco  = load_coco(cfg.DATA_TACO_PATH,  split)
    glass = load_coco(cfg.DATA_GLASS_PATH, split)

    if taco is None:
        print("  [ERROR] TACO not found – skip split.")
        return None

    taco  = copy.deepcopy(taco)
    if glass:
        glass = copy.deepcopy(glass)

    _print_stats_line("raw", taco)

    # ── 2. Keep only desired categories ─────────────────────────────────────
    if cfg.KEEP_CATEGORIES:
        taco = keep_categories_by_name(taco, cfg.KEEP_CATEGORIES)
    _print_stats_line("after filter", taco)

    # ── 3. Validate & clean bboxes ──────────────────────────────────────────
    taco, bbox_report = clean_bboxes(taco)
    if glass:
        glass, _ = clean_bboxes(glass)
    print(f"  ✓ BBox cleaned: {bbox_report['removed']} removed, "
          f"{bbox_report['clipped']} clipped  (total={bbox_report['total']})")

    # ── 4. Merge Glass dataset ───────────────────────────────────────────────
    if glass:
        taco, n_added = merge_glass_into_taco(taco, glass)
        print(f"  ✓ Merged Glass: +{n_added} annotations added")
        _print_stats_line("after merge", taco)
        copy_glass_images(cfg, split)
    else:
        print("  ! Glass dataset not found – merge skipped")



    # ── 6. Undersample (train only) ──────────────────────────────────────────
    if split == "train":
        for class_name, max_count in cfg.UNDERSAMPLE.items():
            taco, n_removed = undersample_class(taco, class_name, max_count, cfg.SEED)
            if n_removed:
                print(f"  ✓ Undersampled '{class_name}': -{n_removed} annotations → max {max_count}")
        _print_stats_line("after sample", taco)

    # ── 7. Remove images without annotations ────────────────────────────────
    taco, n_imgs_removed = filter_images_without_annotations(taco)
    print(f"  ✓ Removed {n_imgs_removed} unannotated images")

    # ── Final stats ──────────────────────────────────────────────────────────
    _print_stats_line("FINAL", taco)

    total = sum(category_stats(taco).values())
    print(f"\n  Category distribution (final):")
    for name, cnt in sorted(category_stats(taco).items(), key=lambda x: -x[1]):
        bar = "█" * int(40 * cnt / total)
        print(f"    {name:<15} {cnt:>5}  {100*cnt/total:>5.1f}%  {bar}")

    return taco

def copy_glass_images(cfg: Config, split: str):

    src_dir = os.path.join(cfg.DATA_GLASS_PATH, split)
    dst_dir = os.path.join(cfg.DATA_TACO_PATH, split)

    copied = 0

    for file in os.listdir(src_dir):

        if file.endswith((".jpg", ".jpeg", ".png")):

            src = os.path.join(src_dir, file)

            # thêm prefix glass_
            dst = os.path.join(dst_dir, "glass_" + file)

            shutil.copy2(src, dst)
            copied += 1

    print(f"  ✓ Copied {copied} glass images")


def run_all(cfg: Config = CFG):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n" + "═"*65)
    print("  DATASET PREPROCESSING PIPELINE")
    print("═"*65)

    for split in cfg.SPLITS:
        result = preprocess_split(split, cfg)
        if result is not None:
            save_coco(
                result,
                cfg.DATA_TACO_PATH,
                split,
                "_annotations.processed.coco.json"
            )

    print("\n" + "═"*65)
    print(f"  Done. Output → {cfg.OUTPUT_DIR}")
    print("═"*65)


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO Dataset Preprocessing Pipeline"
    )

    parser.add_argument(
        "--taco-path",
        type=str,
        default=CFG.DATA_TACO_PATH,
        help="Path tới dataset TACO"
    )

    parser.add_argument(
        "--glass-path",
        type=str,
        default=CFG.DATA_GLASS_PATH,
        help="Path tới dataset Glass"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=CFG.OUTPUT_DIR,
        help="Output directory"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=CFG.SPLITS,
        help="Danh sách split cần xử lý"
    )

    parser.add_argument(
        "--no-glass",
        action="store_true",
        help="Không merge glass dataset"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CFG.SEED,
        help="Random seed"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    CFG.DATA_TACO_PATH = args.taco_path
    CFG.DATA_GLASS_PATH = args.glass_path
    CFG.OUTPUT_DIR = args.output_dir
    CFG.SPLITS = args.splits
    CFG.SEED = args.seed

    if args.no_glass:
        CFG.DATA_GLASS_PATH = None
    run_all()

