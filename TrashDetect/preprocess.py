import random
from collections import defaultdict


def filter_categories(coco: dict, remove_ids: set) -> dict:
    """Xóa các category không cần thiết và annotation tương ứng."""
    coco["categories"] = [c for c in coco["categories"] if c["id"] not in remove_ids]
    coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] not in remove_ids]
    return coco


def reindex_categories(coco: dict) -> tuple[dict, dict]:
    """
    Reindex category ID về 0-based liên tục.
    Trả về (coco đã cập nhật, mapping old_id -> new_id).
    """
    old_to_new = {}
    for i, cat in enumerate(coco["categories"]):
        old_to_new[cat["id"]] = i
        cat["id"] = i

    for ann in coco["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]

    return coco, old_to_new


def undersample_class(coco: dict, class_name: str, max_count: int, seed: int = 42) -> dict:
    """
    Giới hạn số annotation của một class cụ thể (dùng để cân bằng dữ liệu).
    Ví dụ: undersample 'plastic' xuống còn max_count annotation.
    """
    target_id = next(
        (c["id"] for c in coco["categories"] if c["name"] == class_name),
        None
    )
    if target_id is None:
        return coco  # class không tồn tại, bỏ qua

    target_anns = [a for a in coco["annotations"] if a["category_id"] == target_id]
    other_anns  = [a for a in coco["annotations"] if a["category_id"] != target_id]

    random.seed(seed)
    sampled = random.sample(target_anns, min(max_count, len(target_anns)))

    coco["annotations"] = other_anns + sampled
    return coco


def filter_images_without_annotations(coco: dict) -> dict:
    """Loại bỏ ảnh không còn annotation nào (sau khi lọc/undersample)."""
    used_image_ids = {a["image_id"] for a in coco["annotations"]}
    coco["images"] = [img for img in coco["images"] if img["id"] in used_image_ids]
    return coco


def build_img_to_anns(coco: dict) -> defaultdict:
    """Tạo dict: image_id -> list[annotation]."""
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    return img_to_anns


def preprocess_coco(coco: dict, split: str, seed: int = 42) -> dict:
    """
    Pipeline tiền xử lý COCO hoàn chỉnh:
      1. Xóa category 'trash' (id=0) và 'other' (id=4)
      2. Reindex category ID
      3. Undersample 'plastic' nếu là split train
      4. Lọc ảnh không có annotation

    Trả về dict coco đã xử lý.
    """
    REMOVE_IDS = {0, 4}

    coco = filter_categories(coco, REMOVE_IDS)
    coco, _ = reindex_categories(coco)

    if split == "train":
        coco = undersample_class(coco, class_name="plastic", max_count=500, seed=seed)

    coco = filter_images_without_annotations(coco)
    return coco