"""
detect_lite.py - Bản nhẹ tối ưu riêng cho CPU:
- Hình YOLO chạy mượt liên tục trên từng khung hình.
- Khoảng cách (Depth) và tên bề mặt phía trước (tường, sàn...) cập nhật định kỳ
  trong luồng nền (background thread) mỗi DEPTH_EVERY khung hình để không làm đơ hình.
- Phát hiện ổ gà cập nhật mỗi POTHOLE_EVERY khung hình.

Cách chạy:
    python detect_lite.py                      # dùng webcam
    python detect_lite.py video.mp4            # dùng video
    python detect_lite.py video.mp4 --save out_lite.mp4
Nhấn 'q' để thoát cửa sổ.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import time
import threading
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForSemanticSegmentation

# =====================================================================
# CẤU HÌNH TỐI ƯU TỐC ĐỘ CHO CPU (ĐIỀU CHỈNH TẠI ĐÂY)
# =====================================================================
# Nếu vẫn giật: giảm WIDTH xuống 320, tăng DEPTH_EVERY lên 30, tăng POTHOLE_EVERY lên 10
WIDTH = 480
DEPTH_EVERY = 15
POTHOLE_EVERY = 5

# Đi ngoài đường: đổi Indoor thành Outdoor ở dòng DEPTH_MODEL:
# "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
YOLO_MODEL = "yolo11n.pt" if os.path.exists("yolo11n.pt") else "yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    print(f"[LITE] Thiết bị chạy: {DEVICE.upper()} | WIDTH={WIDTH} | DEPTH_EVERY={DEPTH_EVERY} | POTHOLE_EVERY={POTHOLE_EVERY}")

    print(f"[1/3] Đang tải YOLO ({YOLO_MODEL})...")
    yolo_model = YOLO(YOLO_MODEL)

    print(f"[2/3] Đang tải Depth ({DEPTH_MODEL})...")
    try:
        depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(DEVICE).eval()
    except Exception as e:
        print(f"[WARN] Chuyển sang model depth dự phòng do lỗi tải {DEPTH_MODEL}: {e}")
        fallback_depth = "LiheYoung/depth-anything-small-hf"
        depth_processor = AutoImageProcessor.from_pretrained(fallback_depth)
        depth_model = AutoModelForDepthEstimation.from_pretrained(fallback_depth).to(DEVICE).eval()

    print(f"[3/3] Đang tải Segmentation ({SEG_MODEL})...")
    seg_processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(DEVICE).eval()

    pothole_model = YOLO("pothole.pt") if os.path.exists("pothole.pt") else None
    return yolo_model, depth_processor, depth_model, seg_processor, seg_model, pothole_model


def compute_depth_and_surface(frame_bgr: np.ndarray, depth_processor, depth_model, seg_processor, seg_model):
    """Tính depth map (mét) và bề mặt chiếm ưu thế ở trung tâm khung hình."""
    h, w = frame_bgr.shape[:2]
    # Thu nhỏ ảnh đầu vào cho Depth & Seg để chạy nhanh trên CPU
    small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (256, 192)), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(small_rgb)

    # 1. Depth
    d_inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        d_out = depth_model(**d_inputs).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        d_out.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    if "metric" not in DEPTH_MODEL.lower():
        d_min, d_max = np.percentile(depth_map, 5), np.percentile(depth_map, 95)
        norm = np.clip((depth_map - d_min) / max(1e-5, (d_max - d_min)), 0.05, 1.0)
        depth_map = 1.5 / norm

    depth_map = np.clip(depth_map, 0.2, 20.0)

    # 2. Segmentation
    s_inputs = seg_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        s_logits = seg_model(**s_inputs).logits

    seg_map = s_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    sh, sw = seg_map.shape
    center_seg = seg_map[int(sh * 0.30):int(sh * 0.70), int(sw * 0.30):int(sw * 0.70)]
    if center_seg.size > 0:
        vals, counts = np.unique(center_seg, return_counts=True)
        dom_id = int(vals[np.argmax(counts)])
        label = seg_model.config.id2label.get(dom_id, "surface").split(",")[0].split(";")[0].strip()
    else:
        label = "surface"

    center_depth = depth_map[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    dist = float(np.median(center_depth)) if center_depth.size > 0 else 0.0
    front_text = f"Truoc mat: {label} {dist:.1f}m"
    return depth_map, front_text


def get_box_median_distance(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    if depth_map is None:
        return 0.0
    h, w = depth_map.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx1, cy1 = max(0, int(x1 + 0.2 * bw)), max(0, int(y1 + 0.2 * bh))
    cx2, cy2 = min(w, int(x2 - 0.2 * bw)), min(h, int(y2 - 0.2 * bh))
    roi = depth_map[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        roi = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0
    return float(np.median(roi))


def detect_potholes_lite(frame_bgr: np.ndarray, depth_map: np.ndarray, pothole_model):
    potholes = []
    if depth_map is None:
        return potholes

    h, w = frame_bgr.shape[:2]
    if pothole_model is not None:
        res = pothole_model(frame_bgr, verbose=False, conf=0.35, imgsz=WIDTH)
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dist = get_box_median_distance(depth_map, x1, y1, x2, y2)
                potholes.append((x1, y1, x2, y2, dist))
        return potholes

    y_start = int(h * 0.55)
    x_offset = int(w * 0.15)
    ground_bgr = frame_bgr[y_start:h, x_offset:int(w * 0.85)]
    ground_depth = depth_map[y_start:h, x_offset:int(w * 0.85)]
    if ground_bgr.size == 0 or ground_depth.size == 0:
        return potholes

    gray = cv2.cvtColor(ground_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val = max(25, int(np.mean(blur) - 1.6 * np.std(blur)))
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ground_area = ground_bgr.shape[0] * ground_bgr.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if ground_area * 0.015 < area < ground_area * 0.18:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if 0.4 <= (rw / float(max(1, rh))) <= 3.5:
                patch_d = ground_depth[ry:ry + rh, rx:rx + rw]
                if patch_d.size > 0 and np.std(patch_d) > 0.08:
                    x1, y1 = rx + x_offset, ry + y_start
                    x2, y2 = x1 + rw, y1 + rh
                    dist = get_box_median_distance(depth_map, x1, y1, x2, y2)
                    potholes.append((x1, y1, x2, y2, dist))

    return potholes[:3]


def resolve_video_source(arg: str):
    if arg is None:
        return 0
    if os.path.exists(arg):
        return arg
    candidates = [
        os.path.join("test_images", arg),
        os.path.join("test_images", "sample_video.mp4"),
        os.path.join("test_images", "Video_test_3d.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[INFO] Tự động dùng video mẫu: '{c}'")
            return c
    return arg


def main():
    parser = argparse.ArgumentParser(description="Chạy bản nhẹ detect_lite.py tối ưu cho CPU")
    parser.add_argument("source", nargs="?", default=None, help="Đường dẫn video (bỏ trống để dùng webcam)")
    parser.add_argument("--save", type=str, default=None, help="Lưu video kết quả (VD: output_lite.mp4)")
    parser.add_argument("--max-frames", type=int, default=0, help="Giới hạn số frame chạy thử (0 = chạy hết)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy ẩn không mở cửa sổ (dùng khi test tự động)")
    args = parser.parse_args()

    source = resolve_video_source(args.source)
    yolo_model, depth_processor, depth_model, seg_processor, seg_model, pothole_model = load_models()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở nguồn video/webcam: {source}")
        sys.exit(1)

    shared = {
        "depth_map": None,
        "front_text": "Truoc mat: dang tinh...",
        "busy": False,
    }
    lock = threading.Lock()

    def bg_worker(frame_copy):
        try:
            d_map, f_text = compute_depth_and_surface(
                frame_copy, depth_processor, depth_model, seg_processor, seg_model
            )
            with lock:
                shared["depth_map"] = d_map
                shared["front_text"] = f_text
        except Exception as e:
            print(f"[WARN] Lỗi luồng Depth/Seg: {e}")
        finally:
            with lock:
                shared["busy"] = False

    writer = None
    frame_idx = 0
    last_potholes = []
    t_start = time.time()

    print("[LITE] Đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        oh, ow = raw_frame.shape[:2]
        scale = WIDTH / float(ow)
        nh = int(oh * scale)
        frame = cv2.resize(raw_frame, (WIDTH, nh))

        # Khởi tạo đồng bộ ở frame đầu tiên để có ngay depth map
        if frame_idx == 1:
            d_map, f_text = compute_depth_and_surface(
                frame, depth_processor, depth_model, seg_processor, seg_model
            )
            with lock:
                shared["depth_map"] = d_map
                shared["front_text"] = f_text
        elif frame_idx % DEPTH_EVERY == 0:
            with lock:
                is_busy = shared["busy"]
                if not is_busy:
                    shared["busy"] = True
                    threading.Thread(target=bg_worker, args=(frame.copy(),), daemon=True).start()

        with lock:
            cur_depth = shared["depth_map"]
            cur_front_text = shared["front_text"]

        # Chạy YOLO mượt trên từng frame
        yolo_res = yolo_model(frame, verbose=False, conf=0.40, imgsz=WIDTH)

        # Cập nhật ổ gà mỗi POTHOLE_EVERY frames
        if frame_idx == 1 or frame_idx % POTHOLE_EVERY == 0:
            last_potholes = detect_potholes_lite(frame, cur_depth, pothole_model)

        # Vẽ YOLO + khoảng cách median
        for r in yolo_res:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = yolo_model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dist = get_box_median_distance(cur_depth, x1, y1, x2, y2)
                color = (0, 165, 255) if (dist > 0 and dist < 1.8) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"{cls_name} {dist:.1f}m" if dist > 0 else cls_name
                cv2.putText(frame, lbl, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vẽ ổ gà khung đỏ
        for x1, y1, x2, y2, dist in last_potholes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {dist:.1f}m", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Vẽ HUD góc trái trên
        cv2.rectangle(frame, (6, 6), (min(WIDTH - 6, 310), 36), (0, 0, 0), -1)
        cv2.putText(frame, cur_front_text, (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if args.save:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (WIDTH, nh))
            writer.write(frame)

        if not args.no_gui:
            cv2.imshow("AI Visual Guidance - LITE (CPU Optimized)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    elapsed = max(1e-3, time.time() - t_start)
    print(f"[DONE] Đã xử lý {frame_idx} frames trong {elapsed:.2f}s ({frame_idx / elapsed:.2f} FPS).")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Đã lưu video kết quả tại: {args.save}")
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
