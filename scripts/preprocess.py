"""
scripts/preprocess.py – Tiền xử lý ảnh TACO trước khi train
=============================================================
TACO chứa ảnh gốc rất lớn (2000–6000px), gây ra:
  - Load chậm khi train vì phải đọc file nặng
  - Tốn RAM/VRAM khi resize on-the-fly mỗi iteration

Script này đọc toàn bộ ảnh trong data/batch_*/, resize xuống kích thước
phù hợp rồi lưu vào data/processed/ – giữ nguyên cấu trúc thư mục.

Chạy 1 lần duy nhất trước khi train:
  python scripts/preprocess.py

Sau khi chạy xong, cấu trúc data/ trông như sau:
  data/
  ├── batch_1/   (ảnh gốc, giữ nguyên)
  ├── batch_2/
  ├── ...
  ├── processed/
  │   ├── batch_1/   (ảnh đã resize)
  │   ├── batch_2/
  │   └── ...
  └── annotations.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thêm thư mục gốc vào PYTHONPATH để import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageFile

# Cho phép PIL đọc ảnh bị cắt/corrupt một phần
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.config import (
    DATA_DIR,
    PROCESSED_DIR,
    ANNOTATION_FILE,
    IMAGE_MAX_SIZE,
    IMAGE_MIN_SIZE,
)


# ---------------------------------------------------------------------------
# Resize 1 ảnh
# ---------------------------------------------------------------------------

def resize_image(
    src_path: Path,
    dst_path: Path,
    min_size: int = IMAGE_MIN_SIZE,
    max_size: int = IMAGE_MAX_SIZE,
    quality: int = 90,
) -> tuple[bool, str]:
    """
    Resize 1 ảnh và lưu vào dst_path.

    Thuật toán resize:
      1. Tính scale = min_size / cạnh_ngắn
      2. Nếu scale * cạnh_dài > max_size → scale = max_size / cạnh_dài
      3. Resize với LANCZOS (chất lượng tốt nhất)

    Returns:
        (success, message)
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Nếu đã tồn tại → bỏ qua để tránh re-process
        if dst_path.exists():
            return True, "existed"

        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        # Tính scale
        scale = min_size / min(w, h)
        if scale * max(w, h) > max_size:
            scale = max_size / max(w, h)

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if scale < 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Lưu ảnh (giữ định dạng gốc nếu JPEG, còn lại → JPEG)
        suffix = dst_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            img.save(dst_path, "JPEG", quality=quality, optimize=True)
        else:
            img.save(dst_path, "PNG", optimize=True)

        return True, f"{w}x{h} → {new_w}x{new_h} (scale={scale:.3f})"

    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TACO images: resize để tăng tốc training"
    )
    parser.add_argument(
        "--annotation", default=ANNOTATION_FILE,
        help=f"Đường dẫn annotations.json (default: {ANNOTATION_FILE})"
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help=f"Thư mục gốc chứa batch_* (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir", default=PROCESSED_DIR,
        help=f"Thư mục lưu ảnh đã resize (default: {PROCESSED_DIR})"
    )
    parser.add_argument(
        "--max-size", type=int, default=IMAGE_MAX_SIZE,
        help=f"Kích thước cạnh dài tối đa (default: {IMAGE_MAX_SIZE})"
    )
    parser.add_argument(
        "--min-size", type=int, default=IMAGE_MIN_SIZE,
        help=f"Kích thước cạnh ngắn tối thiểu (default: {IMAGE_MIN_SIZE})"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Số thread xử lý song song (default: 8)"
    )
    parser.add_argument(
        "--quality", type=int, default=90,
        help="JPEG quality (default: 90)"
    )
    args = parser.parse_args()

    # ── Đọc danh sách ảnh từ annotations.json ────────────────────────────
    print(f"[Preprocess] Đọc annotations: {args.annotation}")
    with open(args.annotation) as f:
        coco = json.load(f)

    images = coco["images"]
    print(f"[Preprocess] Tổng số ảnh: {len(images)}")
    print(f"[Preprocess] Resize về: min={args.min_size}px, max={args.max_size}px")
    print(f"[Preprocess] Output   : {args.output_dir}")
    print(f"[Preprocess] Workers  : {args.workers}")
    print()

    # ── Chuẩn bị danh sách task ───────────────────────────────────────────
    tasks = []
    for img_info in images:
        file_name = img_info["file_name"]
        src = Path(args.data_dir) / file_name
        dst = Path(args.output_dir) / file_name

        if not src.exists():
            print(f"[WARN] Không tìm thấy: {src}")
            continue

        tasks.append((src, dst))

    print(f"[Preprocess] Sẽ xử lý: {len(tasks)} ảnh")

    # ── Xử lý song song ───────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0
    skip_count    = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                resize_image, src, dst,
                args.min_size, args.max_size, args.quality
            ): (src, dst)
            for src, dst in tasks
        }

        for i, future in enumerate(as_completed(futures), 1):
            src, dst = futures[future]
            ok, msg  = future.result()

            if msg == "existed":
                skip_count += 1
            elif ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"[FAIL] {src.name}: {msg}")

            if i % 100 == 0 or i == len(tasks):
                print(f"[Preprocess] Tiến độ: {i}/{len(tasks)}  "
                      f"(OK:{success_count} | Skip:{skip_count} | Fail:{fail_count})")

    # ── Tóm tắt ──────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print(f"  Hoàn thành tiền xử lý ảnh TACO")
    print(f"  ✔ Resize thành công : {success_count}")
    print(f"  ⟳ Đã tồn tại (skip): {skip_count}")
    print(f"  ✘ Lỗi               : {fail_count}")
    print(f"  Output              : {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
