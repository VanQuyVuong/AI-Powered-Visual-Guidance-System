# BUG #03: Lỗi tải model từ HuggingFace hoặc không tìm thấy file `video.mp4`

- **Trạng thái:** ✅ ĐÃ PHÒNG NGỪA & XỬ LÝ TỰ ĐỘNG
- **Vị trí:** [`detect.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/detect.py), [`detect_lite.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/detect_lite.py)

## 1. Mô tả lỗi
1. **Lỗi tên repo model trên HuggingFace:**
   - Các repo model phát hiện ổ gà hoặc Depth Anything V2 Metric trên HuggingFace đôi khi bị tác giả đổi tên hoặc giới hạn tải, dẫn đến crash `RepositoryNotFoundError` / `EntryNotFoundError`.
2. **Lỗi đường dẫn `python detect.py video.mp4`:**
   - Trong thư mục gốc không có file tên `video.mp4` mà các video mẫu nằm trong `test_images/sample_video.mp4` và `test_images/Video_test_3d.mp4`. Nếu gõ `python detect_lite.py video.mp4` sẽ báo không mở được video.

## 2. Cách đã khắc phục
- **Cơ chế Fallback Model tự động:**
  - Nếu `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` gặp lỗi khi tải, code tự động chuyển sang `LiheYoung/depth-anything-small-hf`.
  - Nếu không có file `pothole.pt` nội bộ hoặc repo ổ gà trên HuggingFace đổi tên, code tự động dùng thuật toán phân tích độ lệch chiều sâu mặt đất (Depth + Ground Contour) để đóng khung đỏ ổ gà/vùng sụt lún kèm khoảng cách mét.
- **Tự động dò tìm video mẫu (`resolve_video_source`):**
  - Khi bạn chạy `python detect_lite.py video.mp4`, nếu không thấy `video.mp4` ở thư mục gốc, chương trình tự động tìm trong `test_images/sample_video.mp4` để chạy ngay mà không bị văng lỗi.
