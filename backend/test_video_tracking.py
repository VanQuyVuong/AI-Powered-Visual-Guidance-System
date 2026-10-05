import cv2
from ultralytics import YOLO
import os
import sys
import numpy as np

# Nhúng thư mục backend vào đường dẫn hệ thống để gọi code THẬT ra test
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)

# IMPORT CODE THẬT TỪ HỆ THỐNG
from app.vision.decision_engine import is_in_roi, determine_position

def run_video_tracking_with_roi(video_path: str, output_path: str):
    print("Đang tải model YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Đang phân tích video kết hợp VÙNG AN TOÀN (ROI)...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Chạy AI lấy dữ liệu thô
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # 2. Vẽ Vùng Quan Tâm (ROI - Hình quạt/thang) lên frame ảnh
        overlay = frame.copy()
        pts = np.array([
            [int(width * 0.2), int(height * 0.4)], # Góc trên trái
            [int(width * 0.8), int(height * 0.4)], # Góc trên phải
            [int(width * 0.8), int(height)],       # Góc dưới phải
            [int(width * 0.2), int(height)]        # Góc dưới trái
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0)) # Tô màu xanh lá
        # Trộn màu xanh lá (30% độ đậm) vào frame gốc
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # 3. Phân tích từng vật thể
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                obj_id = int(boxes.id[i].item()) if boxes.id is not None else 0
                
                # Gọi CODE THẬT: Kiểm tra xem vật thể có lọt vào Vùng nguy hiểm không?
                if is_in_roi(x1, y1, x2, y2, width, height):
                    # NGUY HIỂM: Vẽ khung đỏ chót, chữ to
                    color = (0, 0, 255) # Đỏ (BGR)
                    pos = determine_position(x1, x2, width)
                    text = f"CẢNH BÁO: ID {obj_id} {class_name} ({pos})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    # AN TOÀN: Nằm ngoài rìa -> Vẽ khung xám nhạt, lờ đi
                    color = (200, 200, 200) # Xám
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, f"An toan: {class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
        # Ghi frame đã vẽ vào video đầu ra
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Xong! Bạn hãy mở file: {output_path} để xem kết quả nhé.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_video = os.path.join(project_root, "test_images", "sample_video.mp4")
    output_video = os.path.join(project_root, "test_images", "result_video_with_roi.mp4")
    
    if os.path.exists(input_video):
        run_video_tracking_with_roi(input_video, output_video)
    else:
        print(f"Không tìm thấy {input_video}")
