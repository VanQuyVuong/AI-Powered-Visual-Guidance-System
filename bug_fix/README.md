# 📋 DANH MỤC BÁO CÁO KIỂM THỬ & SỬA LỖI (BUG FIX & TEST REPORTS)

Thư mục này tổng hợp toàn bộ các báo cáo kiểm thử tính năng tự động và lịch sử khắc phục lỗi của hệ thống.

---

## 1. Báo cáo kiểm thử tự động
- [📄 TEST_REPORT.md](TEST_REPORT.md): Báo cáo toàn diện tình trạng các module (YOLO, Depth Anything, Tracking, ROI, Blaze Voice, STT, WebSocket, OCR, MBAPP Android).

## 2. Các lỗi tính năng (Feature Bug Reports)
- [🐛 BUG_01_yolo_service.py.md](BUG_01_yolo_service.py.md): Xử lý lỗi mảng byte rỗng và khởi tạo YOLO service.
- [🐛 BUG_02_tactile_paving_detector.py.md](BUG_02_tactile_paving_detector.py.md): Xử lý an toàn ảnh rỗng và kiểm tra đường gạch nổi.

## 3. Lỗi hệ thống, môi trường & tích hợp (`system_bugs/`)
- [BUG_01: Thiếu File `detect.py` và `detect_lite.py`](system_bugs/BUG_01_Thieu_File_detect_va_detect_lite.md)
- [BUG_02: Môi trường Venv & thư viện HuggingFace](system_bugs/BUG_02_Moi_Truong_Venv_va_Thieu_Thu_Vien_HF.md)
- [BUG_03: Rủi ro tên Repo Model HuggingFace](system_bugs/BUG_03_Rui_Ro_Ten_Repo_Model_HuggingFace.md)
- [BUG_04: OpenCV Empty Bytes Backend](system_bugs/BUG_04_OpenCV_Empty_Bytes_Backend.md)
- [BUG_05: Lỗi mất nhận diện và không đọc giọng nói](system_bugs/BUG_05_Loi_Mat_Nhan_Dien_Va_Khong_Doc_Giong_Noi.md)
- [BUG_06: Lệch tọa độ Bounding Box HUD](system_bugs/BUG_06_Lech_Toa_Do_Bounding_Box_HUD.md)
