import cv2
from ultralytics import YOLO
import os

def test_video_segmentation():
    print("Đang tải model YOLOv8 Segmentation (Phân mảng)...")
    # Sử dụng mô hình '-seg'
    model = YOLO('yolov8n-seg.pt')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_video = os.path.join(project_root, "test_images", "sample_video.mp4")
    output_video = os.path.join(project_root, "test_images", "result_segmentation_video.mp4")
    
    if not os.path.exists(input_video):
        print(f"Lỗi: Không tìm thấy video {input_video}")
        return
        
    print(f"Bắt đầu xử lý video: {input_video}")
    
    cap = cv2.VideoCapture(input_video)
    
    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Phân tích frame với YOLO segmentation
        results = model(frame, verbose=False)
        
        # Vẽ các mảng phân vùng lên frame
        annotated_frame = results[0].plot()
        
        # Ghi frame vào video đầu ra
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Đã xử lý {frame_count}/60 frames (Preview Mode)...")
            
        if frame_count >= 60:
            print("Đã đạt 60 frames, kết thúc sớm để tạo preview nhanh!")
            break
            
    cap.release()
    out.release()
    print(f"Hoàn thành! Video kết quả đã được lưu tại: {output_video}")

if __name__ == "__main__":
    test_video_segmentation()
