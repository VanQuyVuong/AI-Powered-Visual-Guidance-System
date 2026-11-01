# BUG #01: Thiếu file `detect.py` và `detect_lite.py` trong thư mục dự án

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC
- **Vị trí:** Thư mục gốc [`AI-Powered-Visual-Guidance-System`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System)

## 1. Mô tả lỗi
Khi kiểm tra toàn bộ cây thư mục `d:\HK 126\HKDN\AI-Powered-Visual-Guidance-System`, hai tệp thực thi được yêu cầu trong hướng dẫn:
- `detect.py` (bản đầy đủ chạy 4 model: YOLO + Depth + Segmentation + Pothole)
- `detect_lite.py` (bản rút gọn tối ưu riêng cho CPU)

chưa tồn tại trên ổ cứng (nếu chạy `python detect.py` hoặc `python detect_lite.py` sẽ báo lỗi `Errno 2: No such file or directory`).

## 2. Cách đã khắc phục
Đã khởi tạo đầy đủ cả 2 file tại thư mục gốc dự án (không đụng chạm tới code Backend/Mobile của App AI đang chạy song song):
- [`detect.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/detect.py):
  - Tích hợp 4 tầng: YOLO (`yolo11n.pt`) + Depth (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) + Segmentation (`nvidia/segformer-b0-finetuned-ade-512-512`) + Phát hiện ổ gà (khung đỏ kèm khoảng cách mét).
  - Có biến `SKIP = 3` và `DEPTH_MODEL` đúng như hướng dẫn.
- [`detect_lite.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/detect_lite.py):
  - Tối ưu riêng cho CPU với các hằng số `WIDTH = 480`, `DEPTH_EVERY = 15`, `POTHOLE_EVERY = 5`.
  - Tách Depth + Segmentation sang luồng nền (`threading.Thread`) để hình ảnh YOLO không bị giật/đứng khung hình.
