import cv2
import numpy as np

def detect_tactile_paving(image_bytes: bytes) -> bool:
    """
    Thuật toán nhận diện đường gạch nổi (Tactile Paving) cho người khiếm thị.
    Dựa trên đặc trưng màu VÀNG đặc trưng và xử lý vùng quan tâm (ROI) ở nửa dưới khung hình.
    Trả về True nếu phát hiện người dùng đang đi trên/hướng về đường gạch nổi.
    """
    # 1. Giải mã ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False
        
    height, width, _ = img.shape
    
    # 2. Cắt lấy nửa dưới của bức ảnh (Vùng bước chân)
    roi_bottom = img[int(height*0.5):height, :]
    
    # 3. Chuyển sang không gian màu HSV để dễ dàng bắt màu Vàng
    hsv = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2HSV)
    
    # Dải màu Vàng của gạch nổi (Có thể tinh chỉnh dải màu này khi ra nắng/trong bóng râm)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    
    # 4. Lọc tạo mặt nạ (mask) chỉ giữ lại các điểm có màu Vàng
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 5. Loại bỏ nhiễu bằng thuật toán Morphology (Mở/Đóng)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # 6. Tính tỷ lệ diện tích màu vàng trong vùng quan sát
    yellow_area = cv2.countNonZero(mask_cleaned)
    total_area = roi_bottom.shape[0] * roi_bottom.shape[1]
    yellow_ratio = yellow_area / total_area
    
    # Nếu diện tích màu vàng chiếm hơn 3% khung hình dưới chân, kết luận là có đường gạch
    if yellow_ratio > 0.03:
        return True
        
    return False
