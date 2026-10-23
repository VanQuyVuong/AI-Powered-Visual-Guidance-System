import os
import cv2
import numpy as np
from ultralytics import YOLO

# Ưu tiên load model YOLO26n (bản mới nhất 2026), tự động fallback yolo11n nếu cần
DEFAULT_MODEL = "yolo26n.pt"
model_file = os.getenv("YOLO_MODEL", DEFAULT_MODEL)

# Tìm đường dẫn file model trong thư mục backend hoặc thư mục gốc dự án
if not os.path.isabs(model_file):
    candidates = [
        model_file,
        os.path.join(os.path.dirname(__file__), "..", "..", model_file),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", model_file),
        "yolo26n.pt",
        "yolo11n.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            model_file = c
            break

print(f"Đang khởi tạo model YOLO ({os.path.basename(model_file)} - bản mới nhất)...")
model = YOLO(model_file)

# Tự động dùng GPU (CUDA) nếu có -> nhanh gấp nhiều lần CPU
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
USE_HALF = (DEVICE != "cpu")
print(f"YOLO chạy trên thiết bị: {DEVICE}")

# Khởi động nóng (warm-up) để lần gọi đầu tiên không bị chậm vài giây
try:
    extra_kw = {"half": True} if USE_HALF else {}
    model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, device=DEVICE, **extra_kw)
except Exception as _e:
    print(f"[WARN] Warm-up YOLO lỗi: {_e}")

def run_inference(image_bytes: bytes):
    """
    Chạy nhận diện trên ảnh dạng byte (vì ảnh gửi qua API web).
    Trả về danh sách object và chiều rộng ảnh (để tính position).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
        
    height, width, _ = img.shape
    
    # Chạy YOLO inference
    results = model(img, verbose=False, device=DEVICE, imgsz=640, **extra_kw)
    
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


def run_tracking(image_bytes: bytes):
    """
    Chạy nhận diện Bám đuổi (Tracking) trên 1 frame ảnh thời gian thực.
    Sử dụng ByteTrack để duy trì ID đối tượng liên tục giữa các khung hình.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
        
    height, width, _ = img.shape
    
    # Chạy YOLO Tracking với ngưỡng tin cậy conf=0.35
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.35,
                          device=DEVICE, imgsz=640, **extra_kw)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = box_coords = boxes.xyxy[i].tolist()
            
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

