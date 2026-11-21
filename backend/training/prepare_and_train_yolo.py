"""
Tiện ích chuẩn bị dataset tự động từ `ImagesTest/` + HuggingFace Open Dataset (`Abtinzandi/Obstacle-Detection-Dataset-YOLO`)
và fine-tune YOLO11 tùy chỉnh (xuất ra `guidance_model.pt`).

Quy trình:
  1. Tự động gán nhãn ảnh chụp thực tế trong `ImagesTest/` bằng YOLOE-26s + ObstacleYOLO11n + SegFormer:
       .venv\\Scripts\\python.exe backend/training/prepare_and_train_yolo.py --mode auto-label
  2. (Tùy chọn) Tải thêm mẫu từ bộ dữ liệu chuẩn người khiếm thị 24.326 ảnh trên HuggingFace:
       .venv\\Scripts\\python.exe backend/training/prepare_and_train_yolo.py --mode download-hf --max-samples 500
  3. Train fine-tune YOLO11 (khuyên dùng Colab T4 GPU qua `backend/training/HKDN_Train_Custom_YOLO.ipynb` nếu tập dữ liệu lớn):
       .venv\\Scripts\\python.exe backend/training/prepare_and_train_yolo.py --mode train --epochs 30
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CUSTOM_CLASSES = [
    "stairs",
    "step",
    "pothole",
    "open_manhole",
    "dropoff",
    "mirror",
    "glass_door",
    "wall",
    "column",
    "chair",
    "table",
    "couch",
    "person",
    "vehicle",
    "obstacle",
]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CUSTOM_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ánh xạ từ nhãn COCO / YOLOE / H1merka / ROD-Dataset (25 lớp) về 15 lớp dẫn đường thống nhất
DATASET_CLASS_REMAP: Dict[str, str] = {
    "stairs": "stairs",
    "stair": "stairs",
    "staircase": "stairs",
    "flight of stairs": "stairs",
    "steps": "stairs",
    "step": "step",
    "curb": "step",
    "doorstep": "step",
    "threshold": "step",
    "bump": "step",
    "pothole": "pothole",
    "water puddle": "pothole",
    "puddle": "pothole",
    "manhole": "open_manhole",
    "open_manhole": "open_manhole",
    "open manhole": "open_manhole",
    "hole": "dropoff",
    "dropoff": "dropoff",
    "mirror": "mirror",
    "glass_door": "glass_door",
    "glass door": "glass_door",
    "wall": "wall",
    "column": "column",
    "pillar": "column",
    "pole": "column",
    "electrical pole": "column",
    "bollard": "column",
    "chair": "chair",
    "bench": "chair",
    "table": "table",
    "dining table": "table",
    "desk": "table",
    "couch": "couch",
    "sofa": "couch",
    "person": "person",
    "bike": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
}


def map_label_to_custom_class(raw_name: str) -> str:
    """Chuẩn hóa bất kỳ tên lớp nguồn nào về bộ 15 lớp chuẩn của hệ thống."""
    key = str(raw_name or "").lower().strip()
    return DATASET_CLASS_REMAP.get(key, "obstacle")


def _write_data_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CUSTOM_CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def prepare_auto_labeled_dataset(images_dir: Path, dataset_dir: Path) -> int:
    """Dùng YOLOE-26s + ObstacleYOLO11n + Depth + SegFormer + Geometric 3D để auto-label `ImagesTest/`."""
    import time
    from app.vision.depth_service import depth_service
    from app.vision.pipeline import analyze_frame_sync

    t0 = time.time()
    while not getattr(depth_service, "load_complete", False) and (time.time() - t0) < 60.0:
        if depth_service.is_loaded and depth_service.seg_loaded and depth_service.pothole_model is not None:
            break
        time.sleep(0.2)

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
        depth_service.compute_depth_map(frame, force_aux=True)
        res = analyze_frame_sync(img_bytes, is_stream=False, include_visuals=False)

        yolo_lines: List[str] = []
        for obj in res.get("objects") or []:
            cname = map_label_to_custom_class(str(obj.get("class_name", "")))
            b = obj.get("bounding_box") or {}
            x1, y1, x2, y2 = float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx, cy = (x1 + bw / 2.0) / w, (y1 + bh / 2.0) / h
            nw, nh = bw / w, bh / h
            yolo_lines.append(f"{CLASS_TO_ID[cname]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        is_val = (idx % 5 == 0)
        dst_img = (val_img_dir if is_val else train_img_dir) / f"local_{idx:04d}_{img_path.name}"
        dst_lbl = (val_lbl_dir if is_val else train_lbl_dir) / f"local_{idx:04d}_{img_path.stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        labeled_count += 1

    yaml_path = _write_data_yaml(dataset_dir)
    print(f"✅ Đã auto-label {labeled_count} ảnh từ {images_dir} -> {dataset_dir} (config: {yaml_path})")
    return labeled_count


def download_hf_rod_dataset(
    dataset_dir: Path,
    repo_id: str = "Abtinzandi/Obstacle-Detection-Dataset-YOLO",
    max_samples: Optional[int] = 500,
) -> int:
    """
    Tải bộ dữ liệu thực tế cho người khiếm thị (24.326 ảnh góc nhìn smartphone, MIT license)
    từ HuggingFace Hub và ánh xạ nhãn về 15 lớp chuẩn của hệ thống.
    """
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    yaml_file = hf_hub_download(repo_id, "data.yaml", repo_type="dataset")
    meta = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
    src_names = meta.get("names", [])
    if isinstance(src_names, dict):
        src_names = [src_names[k] for k in sorted(src_names.keys())]

    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    train_imgs = [f for f in repo_files if f.startswith("train/images/") and Path(f).suffix.lower() in IMAGE_EXTS]
    valid_imgs = [f for f in repo_files if f.startswith("valid/images/") and Path(f).suffix.lower() in IMAGE_EXTS]

    if max_samples is not None and max_samples > 0:
        n_val = max(1, max_samples // 5)
        n_tr = max(1, max_samples - n_val)
        train_imgs = train_imgs[:n_tr]
        valid_imgs = valid_imgs[:n_val]

    imported = 0
    for split_dst, img_list in (("train", train_imgs), ("val", valid_imgs)):
        out_img_dir = dataset_dir / "images" / split_dst
        out_lbl_dir = dataset_dir / "labels" / split_dst
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for rel_img in img_list:
            rel_lbl = rel_img.replace("/images/", "/labels/", 1)
            rel_lbl = str(Path(rel_lbl).with_suffix(".txt")).replace("\\", "/")
            if rel_lbl not in repo_files:
                continue
            local_img = hf_hub_download(repo_id, rel_img, repo_type="dataset")
            local_lbl = hf_hub_download(repo_id, rel_lbl, repo_type="dataset")

            remapped_lines: List[str] = []
            for line in Path(local_lbl).read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_id = int(float(parts[0]))
                if src_id < 0 or src_id >= len(src_names):
                    continue
                raw_cls = str(src_names[src_id])
                if raw_cls.lower() in ("building", "road", "pedestrian crosswalk", "tree"):
                    continue
                dst_cls = map_label_to_custom_class(raw_cls)
                remapped_lines.append(f"{CLASS_TO_ID[dst_cls]} " + " ".join(parts[1:5]))

            if not remapped_lines:
                continue
            stem = Path(rel_img).stem
            ext = Path(rel_img).suffix
            shutil.copy2(local_img, out_img_dir / f"hf_{stem}{ext}")
            (out_lbl_dir / f"hf_{stem}.txt").write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
            imported += 1

    yaml_path = _write_data_yaml(dataset_dir)
    print(f"✅ Đã nhập {imported} ảnh có nhãn từ `{repo_id}` -> {dataset_dir} (config: {yaml_path})")
    return imported


def train_custom_yolo(
    dataset_dir: Path,
    epochs: int = 30,
    imgsz: int = 416,
    base_model: str = "yolo11n.pt",
) -> None:
    """Fine-tune YOLO11 trên dataset vừa chuẩn bị và xuất ra `guidance_model.pt`."""
    from ultralytics import YOLO

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Chưa có {yaml_path}. Hãy chạy `--mode auto-label` hoặc `--mode download-hf` trước.")

    base_weights = PROJECT_ROOT / base_model
    model = YOLO(str(base_weights if base_weights.exists() else base_model))
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
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset & Fine-tune YOLO11 cho dẫn đường người khiếm thị.")
    parser.add_argument("--mode", choices=["auto-label", "download-hf", "train"], required=True)
    parser.add_argument("--images-dir", default=str(PROJECT_ROOT / "ImagesTest"))
    parser.add_argument("--dataset-dir", default=str(BACKEND_DIR / "training" / "custom_dataset"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--max-samples", type=int, default=500, help="Giới hạn số ảnh tải từ HuggingFace (0 = tải toàn bộ 24K ảnh).")
    args = parser.parse_args()

    if args.mode == "auto-label":
        prepare_auto_labeled_dataset(Path(args.images_dir), Path(args.dataset_dir))
    elif args.mode == "download-hf":
        limit = None if args.max_samples <= 0 else args.max_samples
        download_hf_rod_dataset(Path(args.dataset_dir), max_samples=limit)
    else:
        train_custom_yolo(Path(args.dataset_dir), epochs=args.epochs, imgsz=args.imgsz, base_model=args.base_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
