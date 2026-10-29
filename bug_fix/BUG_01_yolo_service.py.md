# 🐛 BUG REPORT #01: cv2.error khi truyền byte ảnh rỗng vào YOLO Service [ĐÃ KHẮC PHỤC ✅]

- **Tệp nguồn:** [`backend/app/vision/yolo_service.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/vision/yolo_service.py)
- **Hàm bị ảnh hưởng:** `run_inference(image_bytes)` và `run_tracking(image_bytes)`
- **Mức độ nghiêm trọng:** `MEDIUM`
- **Trạng thái:** **ĐÃ SỬA VÀ PASS TEST 100% ✅**
- **Ngày phát hiện:** 2026-10-29

---

## 1. Hiện tượng & Log lỗi ban đầu
Khi client gửi một frame rỗng 0 bytes (`b""`), thư viện OpenCV 5.0+ trong `yolo_service.py` bị sập do vi phạm assertion `!buf.empty()`:

```python
cv2.error: OpenCV(5.0.0) D:\a\opencv-python\opencv-python\opencv\modules\imgcodecs\src\loadsave.cpp:1291: 
error: (-215:Assertion failed) !buf.empty() in function 'cv::imdecode_'
```

## 2. Nguyên nhân gốc rễ (Root Cause)
Hàm gọi `cv2.imdecode(nparr, ...)` ngay cả khi `image_bytes` rỗng hoặc `nparr.size == 0`.

## 3. Cách khắc phục đã thực hiện (Applied Fix)
Đã thêm kiểm tra mảng byte rỗng ở đầu hàm `run_inference` và `run_tracking`:
```python
if not image_bytes:
    raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")

nparr = np.frombuffer(image_bytes, np.uint8)
if nparr.size == 0:
    raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
```

## 4. Kết quả kiểm chứng
- Chạy hàm `run_inference(b"")`: Bắt lỗi `ValueError` chuẩn xác, không còn crash OpenCV.
- Chạy toàn bộ 24 bài test hệ thống: **PASS 100%**.
