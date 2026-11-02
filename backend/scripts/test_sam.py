r"""
Test SAM: YOLO phát hiện vật -> MobileSAM phân đoạn theo hộp -> vẽ mặt nạ ra ảnh.

Chạy (từ thư mục gốc dự án):
    .\.venv\Scripts\python.exe backend\scripts\test_sam.py [đường_dẫn_ảnh]

Kết quả: test_images/result_sam.jpg
"""

import io
import os
import sys
import time

import cv2
import numpy as np

# Console Windows mặc định là CP1252 -> ép UTF-8 để in được tiếng Việt
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.vision.sam_service import sam_service  # noqa: E402


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "test_images", "sample.jpg")
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Không đọc được ảnh: {img_path}")
        sys.exit(1)
    print(f"Ảnh: {img_path}  kích thước {frame.shape[1]}x{frame.shape[0]}")

    # 1. YOLO lấy hộp (dùng model có sẵn trong dự án)
    from ultralytics import YOLO

    yolo_path = os.path.join(ROOT, "yolo11n.pt")
    yolo = YOLO(yolo_path if os.path.exists(yolo_path) else "yolov8n.pt")
    det = yolo(frame, verbose=False, conf=0.25)[0]
    boxes = det.boxes.xyxy.cpu().numpy().tolist()
    names = [yolo.names[int(c)] for c in det.boxes.cls.cpu().numpy()]
    print(f"YOLO phát hiện {len(boxes)} vật: {names}")
    if not boxes:
        print("Không có vật nào để phân đoạn.")
        return

    # 2. SAM phân đoạn theo hộp (lần 1 gồm cả thời gian nạp model, lần 2 mới là tốc độ thật)
    t0 = time.time()
    segs = sam_service.segment_boxes(frame, boxes)
    t_first = time.time() - t0
    if not segs and sam_service.load_error:
        print(f"LỖI nạp SAM: {sam_service.load_error}")
        sys.exit(2)

    t0 = time.time()
    segs = sam_service.segment_boxes(frame, boxes)
    t_warm = time.time() - t0
    print(f"SAM lần 1 (gồm nạp model): {t_first:.2f}s | lần 2 (tốc độ thật): {t_warm:.2f}s | {len(segs)} mặt nạ")

    # 3. Vẽ kết quả
    overlay = frame.copy()
    rng = np.random.default_rng(7)
    for s, name in zip(segs, names):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        poly = np.array(s["polygon"], dtype=np.int32)
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)
        b = s["bounding_box"]
        cv2.putText(frame, f"{name} {s['area_ratio']*100:.1f}%", (b["x1"], max(15, b["y1"] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    out_path = os.path.join(ROOT, "test_images", "result_sam.jpg")
    cv2.imwrite(out_path, out)
    print(f"Đã lưu: {out_path}")


if __name__ == "__main__":
    main()
