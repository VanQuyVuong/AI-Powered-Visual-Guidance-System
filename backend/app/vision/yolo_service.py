import cv2
import numpy as np
from ultralytics import YOLO

# Tải model ở ngoài hàm để nó chỉ load 1 lần khi bật server, không load lại mỗi khi có request
print("Đang khởi tạo model YOLO...")
model = YOLO('yolov8n.pt')

def run_inference(image_bytes: bytes):
    """
    Chạy nhận diện trên ảnh dạng byte (vì ảnh gửi qua API web).
    Trả về danh sách object và chiều rộng ảnh (để tính position).
    """
    # Chuyển đổi dữ liệu byte thành ma trận ảnh để OpenCV và YOLO đọc được
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
        
    height, width, _ = img.shape
    
    # Chạy YOLO
    results = model(img)
    
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
    Chạy nhận diện Bám đuổi (Tracking) trên 1 frame ảnh.
    Giống run_inference nhưng dùng model.track thay vì model() để nhớ ID vật thể.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
        
    height, width, _ = img.shape
    
    # Chạy YOLO Tracking
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    detections = []
    # Ở đây chúng ta có thể mở rộng lấy thêm ID của vật thể (result.boxes.id)
    # Nhưng hiện tại để nhanh gọn ta vẫn dùng chung Schema cũ
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            # Nếu vật thể có ID (Tracking thành công)
            obj_id = int(boxes.id[i].item()) if boxes.id is not None else None
            
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
