"""
Tiện ích chuẩn bị dataset tự động từ `ImagesTest/` và fine-tune YOLO11 tùy chỉnh (tạo `guidance_model.pt`).

Quy trình:
  1. Bỏ ảnh chụp thực tế vào `ImagesTest/cau_thang/`, `ImagesTest/ho_sau/`, `ImagesTest/tuong_cot/`, `ImagesTest/vat_can_xa/`.
  2. Sinh nhãn gợi ý YOLO tự động bằng chính hệ thống đa mô hình hiện tại:
       .venv\\Scripts\\python.exe backend/training/prepare_and_train_yolo.py --mode auto-label
  3. Kiểm tra lại nhãn trong `backend/training/custom_dataset/` (nếu cần), rồi train:
       .venv\\Scripts\\python.exe backend/training/prepare_and_train_yolo.py --mode train --epochs 30
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CUSTOM_CLASSES = ["stairs", "pothole", "dropoff", "wall", "column", "chair", "table", "person", "obstacle"]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CUSTOM_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_auto_labeled_dataset(images_dir: Path, dataset_dir: Path) -> int:
    """Dùng YOLO + Depth + SegFormer + Geometric 3D để auto-label thành bộ dataset YOLOv8/11."""
    from app.vision.depth_service import depth_service
    from app.vision.pipeline import analyze_frame_sync

    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_results" not in p.parts
    )
    if not images:
        print(f"📂 Không có ảnh nào trong {images_dir} để gắn nhãn.")
        return 0

    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    for idx, img_path in enumerate(images):
        img_bytes = img_path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        depth_service.compute_depth_map(frame)
        res = analyze_frame_sync(img_bytes, is_stream=False, include_visuals=False)

        yolo_lines: List[str] = []
        for obj in res.get("objects") or []:
            cname = str(obj.get("class_name", "")).lower().strip()
            if cname in ("step", "curb", "doorstep", "threshold"):
                cname = "stairs"
            elif cname == "pillar":
                cname = "column"
            elif cname in ("dining table", "desk"):
                cname = "table"
            if cname not in CLASS_TO_ID:
                cname = "obstacle"

            b = obj.get("bounding_box") or {}
            x1, y1, x2, y2 = float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx, cy = (x1 + bw / 2.0) / w, (y1 + bh / 2.0) / h
            nw, nh = bw / w, bh / h
            yolo_lines.append(f"{CLASS_TO_ID[cname]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        is_val = (idx % 5 == 0)
        dst_img = (val_img_dir if is_val else train_img_dir) / f"{idx:04d}_{img_path.name}"
        dst_lbl = (val_lbl_dir if is_val else train_lbl_dir) / f"{idx:04d}_{img_path.stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        labeled_count += 1

    yaml_path = dataset_dir / "data.yaml"
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CUSTOM_CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"✅ Đã auto-label {labeled_count} ảnh vào {dataset_dir} (config: {yaml_path})")
    return labeled_count


def train_custom_yolo(dataset_dir: Path, epochs: int = 30, imgsz: int = 416) -> None:
    """Fine-tune YOLO11n trên dataset vừa chuẩn bị và lưu weights thành `guidance_model.pt`."""
    from ultralytics import YOLO

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Chưa có {yaml_path}. Hãy chạy `--mode auto-label` trước.")

    base_weights = PROJECT_ROOT / "yolo11n.pt"
    model = YOLO(str(base_weights if base_weights.exists() else "yolo11n.pt"))
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=str(PROJECT_ROOT / "runs"),
        name="hkdn_custom_yolo11",
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        target_pt = PROJECT_ROOT / "guidance_model.pt"
        shutil.copy2(best_pt, target_pt)
        print(f"🎉 Đã xuất model mới tại: {target_pt} (yolo_service sẽ tự ưu tiên load file này!)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label và Fine-tune YOLO11 từ thư mục ImagesTest.")
    parser.add_argument("--mode", choices=["auto-label", "train"], required=True)
    parser.add_argument("--images-dir", default=str(PROJECT_ROOT / "ImagesTest"))
    parser.add_argument("--dataset-dir", default=str(BACKEND_DIR / "training" / "custom_dataset"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    if args.mode == "auto-label":
        prepare_auto_labeled_dataset(Path(args.images_dir), Path(args.dataset_dir))
    else:
        train_custom_yolo(Path(args.dataset_dir), epochs=args.epochs, imgsz=args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
