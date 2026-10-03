import cv2
import json
from ultralytics import YOLO
import sys
import os

def test_yolo_detection(image_path):
    print(f"Đang tải model YOLOv8n...")
    # Khởi tạo model YOLOv8 nano (nhẹ, nhanh, phù hợp MVP)
    model = YOLO('yolov8n.pt') 

    if not os.path.exists(image_path):
        print(f"LỖI: Không tìm thấy ảnh tại {image_path}")
        return

    print(f"Đang phân tích ảnh: {image_path}")
    
    # Chạy inference
    results = model(image_path)
    
    # Xử lý kết quả theo chuẩn JSON yêu cầu
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # box.cls là tensor chứa ID của class
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            # box.conf là độ tin cậy
            confidence = float(box.conf[0].item())
            
            # box.xyxy là tọa độ [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detection = {
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            }
            detections.append(detection)

    output = {
        "success": True,
        "detections": detections
    }
    
    # In ra JSON đẹp
    print("\n=== KẾT QUẢ DETECTION (JSON) ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Dùng đường dẫn tuyệt đối dựa trên vị trí của file script này
    # Dù bạn chạy lệnh ở thư mục nào thì nó vẫn tìm đúng thư mục test_images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_images_dir = os.path.join(project_root, "test_images")
    sample_image = os.path.join(test_images_dir, "sample.jpg")
    
    os.makedirs(test_images_dir, exist_ok=True)
    if not os.path.exists(sample_image):
        print(f"Tạo một ảnh mẫu màu đen tại {sample_image} vì chưa có ảnh thật...")
        import numpy as np
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(sample_image, dummy_img)
        
    test_yolo_detection(sample_image)
