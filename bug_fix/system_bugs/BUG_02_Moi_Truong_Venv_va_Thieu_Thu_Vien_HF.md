# BUG #02: Lệch tên môi trường ảo (`.venv` vs `venv`) và thiếu thư viện `transformers`, `huggingface_hub`

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC
- **Vị trí:** Môi trường ảo Python tại `d:\HK 126\HKDN\AI-Powered-Visual-Guidance-System`

## 1. Mô tả lỗi
1. **Khác tên thư mục môi trường ảo:**
   - Dự án hiện tại đang dùng `.venv` (đã cài sẵn `torch 2.14.1`, `torchvision 0.29.1`, `ultralytics 8.4.170`, `opencv-python 5.0.0.93`, `pillow 12.3.0`, `fastapi`, `easyocr`).
   - Hướng dẫn cài đặt mới lại gõ lệnh `python -m venv venv` và `venv\Scripts\activate`. Nếu tạo mới thư mục `venv` độc lập sẽ tốn thêm hơn 3GB dung lượng và làm mất các thư viện sẵn có của dự án.
2. **Thiếu gói thư viện HuggingFace:**
   - Trong `.venv` chưa cài `transformers` và `huggingface_hub` (gây lỗi `ModuleNotFoundError: No module named 'transformers'` khi chạy `detect.py` / `detect_lite.py`).
3. **Phiên bản Python:**
   - Máy đang cài **Python 3.13.11** (hướng dẫn ghi Python 3.10 hoặc 3.11). Tuy nhiên bản `torch 2.14.1` và `transformers` mới nhất trong `.venv` đã hỗ trợ hoàn toàn Python 3.13.

## 2. Cách đã khắc phục
- Cài đặt bổ sung `transformers` và `huggingface_hub` trực tiếp vào `.venv` hiện có của dự án.
- Tạo Junction liên kết `venv` -> `.venv` để bạn gõ `venv\Scripts\activate` hay `.venv\Scripts\activate` đều hoạt động chung trên một môi trường duy nhất, không xung đột với AI thứ 2.
