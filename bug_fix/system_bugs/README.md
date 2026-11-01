# DANH SÁCH LỖI & VẤN ĐỀ PHÁT HIỆN TRONG DỰ ÁN (`BUG/`)

Thư mục này lưu trữ toàn bộ các lỗi/vấn đề phát hiện khi kiểm tra dự án và cài đặt bộ công cụ `detect.py` / `detect_lite.py` (YOLO + Depth + Segmentation + Ổ gà), đồng bộ thời gian thực với phiên AI song song.

## Danh mục file lỗi
1. [`BUG_01_Thieu_File_detect_va_detect_lite.md`](./BUG_01_Thieu_File_detect_va_detect_lite.md) — Thiếu 2 file `detect.py` và `detect_lite.py` trong thư mục gốc dự án (**Đã tạo & khắc phục**).
2. [`BUG_02_Moi_Truong_Venv_va_Thieu_Thu_Vien_HF.md`](./BUG_02_Moi_Truong_Venv_va_Thieu_Thu_Vien_HF.md) — Lệch tên môi trường ảo (`.venv` vs `venv`), phiên bản Python 3.13.11 và thiếu `transformers`, `huggingface_hub` (**Đã cài đặt & tạo liên kết `venv`**).
3. [`BUG_03_Rui_Ro_Ten_Repo_Model_HuggingFace.md`](./BUG_03_Rui_Ro_Ten_Repo_Model_HuggingFace.md) — Lỗi không tải được model trên HuggingFace nếu tên repo thay đổi hoặc mất kết nối (**Đã bọc cơ chế Fallback tự động**).
4. [`BUG_04_OpenCV_Empty_Bytes_Backend.md`](./BUG_04_OpenCV_Empty_Bytes_Backend.md) — Lỗi `cv2.imdecode` crash khi nhận mảng byte rỗng `b""` trong `yolo_service.py` và `tactile_paving_detector.py` (**Đã đồng bộ và xử lý**).
