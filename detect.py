"""
detect.py - Bản đầy đủ ghép sẵn 4 model:
1. YOLO (Nhận diện vật thể chung: người, ghế, bàn, xe...)
2. Depth Anything V2 Metric (Ước lượng khoảng cách thực tế bằng mét)
3. SegFormer Semantic Segmentation (Nhận diện bề mặt phía trước: tường, sàn, đường, vỉa hè...)
4. Pothole Detector (Nhận diện ổ gà / mặt đường hư hỏng kèm khoảng cách)

Cách chạy:
    python detect.py                      # dùng webcam
    python detect.py video.mp4            # dùng video
    python detect.py video.mp4 --save out.mp4  # lưu video kết quả cho báo cáo
Nhấn 'q' để thoát cửa sổ.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForSemanticSegmentation
from huggingface_hub import hf_hub_download

# =====================================================================
# CẤU HÌNH (ĐIỀU CHỈNH TẠI ĐÂY)
# =====================================================================
# Máy yếu bị giật: tăng SKIP = 3 lên 5 hoặc 6
SKIP = 3

# Đi ngoài đường: đổi Indoor thành Outdoor ở dòng DEPTH_MODEL bên dưới:
# "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Model phân đoạn ngữ nghĩa (nhận diện tường, sàn, vỉa hè, mặt đường...)
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# Model YOLO nhận diện vật thể
YOLO_MODEL = "yolo11n.pt" if os.path.exists("yolo11n.pt") else "yolov8n.pt"

# Model phát hiện ổ gà (đã tải sẵn pothole.pt tại thư mục dự án từ keremberke/yolov8n-pothole-segmentation)
POTHOLE_HF_REPO = "keremberke/yolov8n-pothole-segmentation"
POTHOLE_HF_FILE = "best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    print(f"[INFO] Thiết bị chạy: {DEVICE.upper()}")

    # 1. Load YOLO
    print(f"[1/4] Đang tải YOLO model ({YOLO_MODEL})...")
    yolo_model = YOLO(YOLO_MODEL)

    # 2. Load Depth Anything V2 Metric
    print(f"[2/4] Đang tải Depth model ({DEPTH_MODEL})...")
    try:
        depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(DEVICE).eval()
    except Exception as e:
        print(f"[WARN] Không tải được {DEPTH_MODEL} ({e}). Chuyển sang LiheYoung/depth-anything-small-hf...")
        fallback_depth = "LiheYoung/depth-anything-small-hf"
        depth_processor = AutoImageProcessor.from_pretrained(fallback_depth)
        depth_model = AutoModelForDepthEstimation.from_pretrained(fallback_depth).to(DEVICE).eval()

    # 3. Load Semantic Segmentation (ADE20K: wall, floor, road, sidewalk...)
    print(f"[3/4] Đang tải Segmentation model ({SEG_MODEL})...")
    seg_processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(DEVICE).eval()

    # 4. Load Pothole model (hoặc dùng fallback nếu repo HF đã đổi tên)
    print("[4/4] Đang kiểm tra model ổ gà (Pothole)...")
    pothole_model = None
    local_pothole = "pothole.pt"
    if os.path.exists(local_pothole):
        try:
            pothole_model = YOLO(local_pothole)
            print(f"[INFO] Đã tải model ổ gà nội bộ: {local_pothole}")
        except Exception as e:
            print(f"[WARN] Lỗi đọc {local_pothole}: {e}")
    else:
        try:
            weight_path = hf_hub_download(repo_id=POTHOLE_HF_REPO, filename=POTHOLE_HF_FILE)
            pothole_model = YOLO(weight_path)
            print("[INFO] Đã tải model ổ gà từ HuggingFace.")
        except Exception:
            print("[INFO] Không dùng repo ổ gà ngoài; sử dụng thuật toán phát hiện ổ gà/hố sụt từ Depth + Contour mặt đất.")

    return yolo_model, depth_processor, depth_model, seg_processor, seg_model, pothole_model


def compute_depth_map(frame_bgr: np.ndarray, depth_processor, depth_model) -> np.ndarray:
    """Trả về bản đồ khoảng cách (đơn vị: mét) cùng kích thước với frame."""
    h, w = frame_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Nếu model trả về relative disparity thay vì metric depth, chuẩn hóa về khoảng 0.4m - 10.0m
    if "metric" not in DEPTH_MODEL.lower():
        d_min, d_max = np.percentile(prediction, 5), np.percentile(prediction, 95)
        norm = np.clip((prediction - d_min) / max(1e-5, (d_max - d_min)), 0.05, 1.0)
        prediction = 1.5 / norm

    return np.clip(prediction, 0.2, 20.0)


def get_box_median_distance(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Lấy median khoảng cách (mét) của vùng trung tâm vật thể trong depth map."""
    h, w = depth_map.shape[:2]
    # Lấy vùng lõi 60% của bounding box để tránh lấy nhầm nền phía sau
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx1 = max(0, int(x1 + 0.2 * bw))
    cy1 = max(0, int(y1 + 0.2 * bh))
    cx2 = min(w, int(x2 - 0.2 * bw))
    cy2 = min(h, int(y2 - 0.2 * bh))

    roi = depth_map[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        roi = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0
    return float(np.median(roi))


def analyze_front_surface(frame_bgr: np.ndarray, depth_map: np.ndarray, seg_processor, seg_model) -> str:
    """
    Xác định bề mặt/vật thể chiếm nhiều nhất ở vùng giữa khung hình (tường, sàn, ghế...)
    và khoảng cách median (mét).
    """
    h, w = frame_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = seg_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = outputs.logits  # (1, num_classes, H/4, W/4)

    seg_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    sh, sw = seg_map.shape

    # Vùng trung tâm khung hình (từ 30% -> 70% theo cả 2 chiều)
    center_seg = seg_map[int(sh * 0.30):int(sh * 0.70), int(sw * 0.30):int(sw * 0.70)]
    if center_seg.size == 0:
        return "Truoc mat: unknown"

    vals, counts = np.unique(center_seg, return_counts=True)
    dominant_id = int(vals[np.argmax(counts)])
    label = seg_model.config.id2label.get(dominant_id, "surface").split(",")[0].split(";")[0].strip()

    # Lấy median depth ở vùng giữa tương ứng
    center_depth = depth_map[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    dist = float(np.median(center_depth)) if center_depth.size > 0 else 0.0
    return f"Truoc mat: {label} {dist:.1f}m"


def detect_potholes(frame_bgr: np.ndarray, depth_map: np.ndarray, pothole_model):
    """
    Phát hiện ổ gà và trả về danh sách [(x1, y1, x2, y2, dist_m), ...].
    Vẽ khung đỏ kèm khoảng cách.
    """
    h, w = frame_bgr.shape[:2]
    potholes = []

    if pothole_model is not None:
        results = pothole_model(frame_bgr, verbose=False, conf=0.35)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dist = get_box_median_distance(depth_map, x1, y1, x2, y2)
                potholes.append((x1, y1, x2, y2, dist))
        return potholes

    # Fallback thông minh: tìm vùng trũng bất thường/tối màu trên mặt đất phía trước (nửa dưới khung hình)
    y_start = int(h * 0.55)
    ground_bgr = frame_bgr[y_start:h, int(w * 0.15):int(w * 0.85)]
    ground_depth = depth_map[y_start:h, int(w * 0.15):int(w * 0.85)]
    if ground_bgr.size == 0 or ground_depth.size == 0:
        return potholes

    gray = cv2.cvtColor(ground_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Tìm các vùng tối cục bộ + có độ lệch sâu so với mặt phẳng xung quanh
    mean_val = float(np.mean(blur))
    std_val = float(np.std(blur))
    thresh_val = max(25, int(mean_val - 1.6 * std_val))
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_offset = int(w * 0.15)
    ground_area = ground_bgr.shape[0] * ground_bgr.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if ground_area * 0.015 < area < ground_area * 0.18:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / float(max(1, rh))
            if 0.4 <= aspect <= 3.5:
                # Kiểm tra độ biến thiên chiều sâu (ổ gà có độ chênh depth tại biên)
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
    # Tự động tìm trong thư mục test_images nếu người dùng gõ 'video.mp4' hoặc 'sample_video.mp4'
    candidates = [
        os.path.join("test_images", arg),
        os.path.join("test_images", "sample_video.mp4"),
        os.path.join("test_images", "Video_test_3d.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[INFO] Không thấy '{arg}' ở thư mục gốc, tự động dùng: '{c}'")
            return c
    return arg


def main():
    parser = argparse.ArgumentParser(description="Chạy YOLO + Depth + Segmentation + Pothole (Bản đầy đủ)")
    parser.add_argument("source", nargs="?", default=None, help="Đường dẫn video (bỏ trống để dùng webcam)")
    parser.add_argument("--save", type=str, default=None, help="Đường dẫn lưu video kết quả (VD: output.mp4)")
    parser.add_argument("--max-frames", type=int, default=0, help="Giới hạn số frame chạy thử (0 = chạy hết)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không mở cửa sổ hiển thị (dùng khi test tự động)")
    args = parser.parse_args()

    source = resolve_video_source(args.source)
    yolo_model, depth_processor, depth_model, seg_processor, seg_model, pothole_model = load_models()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở nguồn hình ảnh/video: {source}")
        sys.exit(1)

    writer = None
    frame_idx = 0
    last_depth_map = None
    last_front_text = "Truoc mat: dang tinh..."
    last_detections = []
    last_potholes = []

    print("[INFO] Bắt đầu xử lý... Nhấn 'q' trên cửa sổ video để thoát.")
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        if frame_idx == 1 or frame_idx % SKIP == 0:
            # 1. Tính Depth Map
            last_depth_map = compute_depth_map(frame, depth_processor, depth_model)

            # 2. Phân tích bề mặt phía trước (SegFormer + Depth)
            last_front_text = analyze_front_surface(frame, last_depth_map, seg_processor, seg_model)

            # 3. Chạy YOLO nhận diện vật thể + đo khoảng cách median
            yolo_res = yolo_model(frame, verbose=False, conf=0.40)
            last_detections = []
            for r in yolo_res:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = yolo_model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dist = get_box_median_distance(last_depth_map, x1, y1, x2, y2)
                    last_detections.append((cls_name, x1, y1, x2, y2, dist))

            # 4. Phát hiện ổ gà
            last_potholes = detect_potholes(frame, last_depth_map, pothole_model)

        # Vẽ kết quả YOLO (khung xanh lá + khoảng cách mét)
        for cls_name, x1, y1, x2, y2, dist in last_detections:
            color = (0, 165, 255) if dist < 1.8 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {dist:.1f}m"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ ổ gà (khung đỏ kèm khoảng cách)
        for x1, y1, x2, y2, dist in last_potholes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {dist:.1f}m", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Vẽ thông tin "Truoc mat: <bề mặt> <khoảng cách>m" ở góc trái trên
        cv2.rectangle(frame, (8, 8), (380, 44), (0, 0, 0), -1)
        cv2.putText(frame, last_front_text, (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if args.save:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
            writer.write(frame)

        if not args.no_gui:
            cv2.imshow("AI Visual Guidance - Full (YOLO + Depth + Seg + Pothole)", frame)
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
