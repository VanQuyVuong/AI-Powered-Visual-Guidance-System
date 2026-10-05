import cv2
from ultralytics import YOLO
import os

def test_segmentation():
    print("Đang tải model YOLOv8 Segmentation (Phân mảng)...")
    # Sử dụng mô hình '-seg' thay vì bản thường
    model = YOLO('yolov8n-seg.pt')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Bạn có thể đổi thành ảnh chụp vỉa hè / đường phố để test
    input_image = os.path.join(project_root, "test_images", "sample.jpg")
    output_image = os.path.join(project_root, "test_images", "result_segmentation.jpg")
    
    if not os.path.exists(input_image):
        print(f"Lỗi: Không tìm thấy ảnh {input_image}")
        print("Vui lòng copy 1 bức ảnh chụp ngoài đường vào thư mục test_images và đổi tên thành sample.jpg")
        return
        
    print("Đang phân tích...")
    results = model(input_image)
    
    # Vẽ kết quả (Lần này sẽ có các mảng màu tô kín vật thể thay vì chỉ có hình vuông)
    annotated_img = results[0].plot()
    
    cv2.imwrite(output_image, annotated_img)
    print(f"Thành công! Hãy mở file {output_image} để xem AI phân mảng đồ vật nhé.")

if __name__ == "__main__":
    test_segmentation()
