# BUG #04: Lỗi `cv2.error` khi truyền byte ảnh rỗng vào Backend Vision

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC (Đồng bộ với `bug_fix/`)
- **Vị trí:**
  - [`backend/app/vision/yolo_service.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/vision/yolo_service.py)
  - [`backend/app/vision/tactile_paving_detector.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/vision/tactile_paving_detector.py)

## 1. Mô tả lỗi
Trong OpenCV 5.0+, khi truyền mảng byte rỗng `b""` vào `cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)`, OpenCV ném ra ngoại lệ C++:
```
cv2.error: OpenCV(5.0.0) ... error: (-215:Assertion failed) !buf.empty() in function 'cv::imdecode_'
```

## 2. Cách đã khắc phục
Kiểm tra `if not image_bytes:` và `if nparr.size == 0:` trước khi gọi `cv2.imdecode` ở cả `yolo_service.py` và `tactile_paving_detector.py`.
