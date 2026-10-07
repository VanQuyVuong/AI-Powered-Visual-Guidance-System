import cv2
import numpy as np

# Sử dụng EasyOCR vì nó nhẹ, hỗ trợ tiếng Việt cực tốt và không cần cài phần mềm ngoài như Tesseract
try:
    import easyocr
    # Khởi tạo model đọc tiếng Việt và tiếng Anh. Model sẽ được tự động tải về lần đầu chạy.
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ Thiếu thư viện easyocr. Hãy chạy: pip install easyocr")

def extract_text(image_bytes: bytes) -> str:
    """
    Nhận diện và trích xuất chữ viết (biển báo, bảng tên, sách) từ ảnh.
    Trọng tâm là hỗ trợ người khiếm thị "đọc" được môi trường xung quanh.
    """
    if not OCR_AVAILABLE:
        return "Tính năng đọc chữ chưa được cài đặt."
        
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return ""
        
    # Chạy OCR
    results = reader.readtext(img)
    
    # Kết quả trả về là list các tuple: (bounding_box, text, confidence)
    # Ta chỉ ghép các đoạn text có độ tin cậy > 0.4 lại thành câu
    detected_texts = []
    for bbox, text, prob in results:
        if prob > 0.4:
            detected_texts.append(text)
            
    if not detected_texts:
        return "Không tìm thấy chữ nào rõ ràng trong ảnh."
        
    return ". ".join(detected_texts)
