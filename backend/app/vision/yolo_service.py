import os
import cv2
import numpy as np
from ultralytics import YOLO

# Tải model ở ngoài hàm để load 1 lần khi bật server
# Thứ tự ưu tiên: Custom model -> YOLO11n (Nano siêu nhanh ~25ms cho real-time 4 model song song) -> YOLOv8n -> YOLO11s
model_file = None
candidates = [
    os.getenv("YOLO_MODEL"),
    "guidance_model.onnx",
    "guidance_model.pt",
    "yolo11n.pt",
    "yolov8n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo26n.pt",
]

for name in candidates:
    if not name:
        continue
    search_paths = [
        name,
        os.path.join(os.path.dirname(__file__), "..", "..", name),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", name)
    ]
    for p in search_paths:
        if os.path.exists(p):
            model_file = p
            break
    if model_file:
        break

if not model_file:
    model_file = "yolov8n.pt"

print(f"Đang khởi tạo model YOLO: {model_file}...")
model = YOLO(model_file)

# Tự động dùng GPU (CUDA) nếu có -> nhanh gấp nhiều lần CPU
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
USE_HALF = (DEVICE != "cpu")
DEFAULT_IMGSZ = 480 if DEVICE != "cpu" else 416
print(f"YOLO chạy trên thiết bị: {DEVICE} (imgsz={DEFAULT_IMGSZ})")

# Khởi động nóng (warm-up) để lần gọi đầu tiên không bị chậm
extra_kw = {"half": True} if USE_HALF else {}
try:
    model(np.zeros((416, 416, 3), dtype=np.uint8), verbose=False, device=DEVICE, imgsz=DEFAULT_IMGSZ, **extra_kw)
except Exception as _e:
    print(f"[WARN] Warm-up YOLO lỗi: {_e}")


def _ensure_bgr_frame(image_or_bytes) -> np.ndarray:
    """Chuyển đổi đầu vào (bytes hoặc np.ndarray BGR) thành mảng ảnh BGR mà không giải mã lặp lại."""
    if isinstance(image_or_bytes, np.ndarray):
        if image_or_bytes.size == 0:
            raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
        return image_or_bytes
    if not image_or_bytes:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
    nparr = np.frombuffer(image_or_bytes, np.uint8)
    if nparr.size == 0:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
    return img


def run_inference(image_or_bytes, imgsz: int = DEFAULT_IMGSZ):
    """
    Chạy nhận diện trên ảnh dạng byte hoặc frame_bgr.
    Trả về danh sách object và chiều rộng ảnh (để tính position).
    """
    img = _ensure_bgr_frame(image_or_bytes)
    height, width, _ = img.shape

    # Chạy YOLO inference ở độ phân giải tối ưu DEFAULT_IMGSZ (416 trên CPU)
    results = model(img, verbose=False, device=DEVICE, imgsz=imgsz, **extra_kw)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    return detections, width


def run_tracking(image_or_bytes, imgsz: int = DEFAULT_IMGSZ):
    """
    Chạy nhận diện Bám đuổi (Tracking) trên 1 frame ảnh thời gian thực.
    Hỗ trợ truyền trực tiếp `frame_bgr` đã giải mã từ pipeline để tiết kiệm 2 lần cv2.imdecode,
    kết hợp suy luận ở kích thước `imgsz=416` giúp giảm ~65% độ trễ luồng chính.
    """
    img = _ensure_bgr_frame(image_or_bytes)
    height, width, _ = img.shape

    # Chạy YOLO Tracking với ngưỡng tin cậy conf=0.25 để bắt nhạy ghế, chai nước, cặp sách, bàn
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.25,
                          device=DEVICE, imgsz=imgsz, **extra_kw)

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # Lấy tracking ID nếu có
            obj_id = int(boxes.id[i].item()) if (boxes.id is not None and len(boxes.id) > i) else None

            detections.append({
                "id": obj_id,
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    return detections, width, height

