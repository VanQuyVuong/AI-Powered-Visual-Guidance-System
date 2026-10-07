# 00 OVERVIEW - KẾ HOẠCH TỔNG QUAN HỆ THỐNG

**Project:** AI-Powered-Visual-Guidance-System
**Team:** 2 thành viên (Người 1 - AI Backend, Người 2 - Flutter)

Tài liệu này dùng để theo dõi tiến độ công việc dựa trên `TEAM_TASKS.md` và `SYSTEM_ANALYSIS.md`. Đánh dấu `[x]` khi hoàn thành, `[ ]` khi chưa hoàn thành.

## 🚀 Trạng thái Sprint (MVP - Vertical Slice)

### Sprint 1: Khởi động & Nhận diện cơ bản
- [x] **G1 (Người 1):** Cài đặt Python, load YOLO pretrained, xử lý 1 ảnh ra JSON (class, confidence, bbox).
- [ ] **G1 (Người 2):** Khởi tạo Flutter app, giao diện chọn ảnh, hiển thị ảnh preview.

### Sprint 2: Logic & API Backend
- [x] **G2 (Người 1):** Thuật toán tính Position (Left/Center/Right).
- [x] **G2 (Người 1):** Decision Engine (ưu tiên vật thể, tạo câu Guidance tiếng Việt).
- [x] **G2 (Người 1):** Dựng FastAPI server endpoint `/api/v1/vision/analyze`.
- [x] **G2 (Người 2):** Flutter gọi API backend, nhận JSON và hiển thị text cảnh báo lên màn hình.

### Sprint 3: Tích hợp Giọng nói
- [ ] **G5 (Người 1):** Tích hợp Voice API (Blaze TTS) dạng Adapter trên Backend.
- [x] **G5 (Người 2):** Audio player trên Flutter, tự động phát âm thanh khi có kết quả (Đã thay thế tạm bằng Local TTS).

### Sprint 4: Tích hợp End-to-End
- [ ] **End-to-End:** `Camera App → FastAPI → YOLO → Decision Engine → Voice API → Phát âm thanh trên App`.
