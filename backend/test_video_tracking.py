import cv2
from ultralytics import YOLO
import os

def run_video_tracking(video_path: str, output_path: str):
    print("Đang tải model YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    # Mở video đầu vào
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Lấy thông số video (chiều rộng, chiều cao, FPS) để tạo video đầu ra
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Khởi tạo công cụ ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Đang phân tích video... Quá trình này có thể mất vài phút tùy độ dài video.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # CHẠY TRACKING: Thay vì model(frame), ta dùng model.track(frame, persist=True)
        # persist=True giúp AI nhớ ID của vật thể từ khung hình trước sang khung hình sau
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Lấy khung hình đã được YOLO vẽ sẵn khung (Bounding Box) và ID
        annotated_frame = results[0].plot()
        
        # Ghi khung hình đã vẽ vào video đầu ra
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    print(f"Đã xử lý xong! Video kết quả được lưu tại: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Đường dẫn file video gốc (bạn cần copy 1 file mp4 vào đây)
    input_video = os.path.join(project_root, "test_images", "sample_video.mp4")
    # File video sau khi phân tích
    output_video = os.path.join(project_root, "test_images", "result_video.mp4")
    
    if not os.path.exists(input_video):
        print(f"Vui lòng copy một đoạn video ngắn (khoảng 5-10 giây) vào thư mục test_images và đổi tên thành sample_video.mp4")
    else:
        run_video_tracking(input_video, output_video)
