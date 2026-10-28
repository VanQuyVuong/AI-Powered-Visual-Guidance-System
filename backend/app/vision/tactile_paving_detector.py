import cv2
import numpy as np

def detect_tactile_paving(image_bytes: bytes) -> bool:
    """
    Thuật toán nhận diện đường gạch nổi (Tactile Paving) cho người khiếm thị.
    Dựa trên đặc trưng màu VÀNG đặc trưng ở làn bước chân trung tâm (walking corridor).
    Trả về True nếu phát hiện người dùng đang thực sự đi trên dải gạch nổi màu vàng.
    """
    # 1. Giải mã ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False
        
    height, width, _ = img.shape
    
    # 2. Cắt lấy làn bước chân trung tâm (60% chiều rộng ở giữa, 40% chiều cao phía dưới chân)
    # Tránh bị nhiễu bởi xe taxi/xe máy màu vàng ở hai bên lề đường
    y_start = int(height * 0.60)
    x_start = int(width * 0.20)
    x_end = int(width * 0.80)
    roi_bottom = img[y_start:height, x_start:x_end]
    
    # 3. Chuyển sang không gian màu HSV để lọc màu Vàng đặc trưng của gạch khiếm thị
    hsv = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 90, 85])
    upper_yellow = np.array([34, 255, 255])
    
    # 4. Lọc tạo mặt nạ (mask)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 5. Loại bỏ nhiễu bằng thuật toán Morphology (Mở/Đóng)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # 6. Tính tỷ lệ diện tích màu vàng trong làn bước chân
    yellow_area = cv2.countNonZero(mask_cleaned)
    total_area = roi_bottom.shape[0] * roi_bottom.shape[1]
    if total_area == 0:
        return False
        
    yellow_ratio = yellow_area / float(total_area)
    
    # 7. Kiểm tra độ tập trung của khối gạch (contour lớn nhất phải đủ đậm nét dạng dải đường)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_area = max([cv2.contourArea(c) for c in contours], default=0)
    
    # Đường gạch nổi chiếm ít nhất 8% diện tích dưới chân VÀ có mảng liền mạch > 3% tổng diện tích làn
    if yellow_ratio >= 0.08 and (max_contour_area / float(total_area)) >= 0.03:
        return True
        
    return False

