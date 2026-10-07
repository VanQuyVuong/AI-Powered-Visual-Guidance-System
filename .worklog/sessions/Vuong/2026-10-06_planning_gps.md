# NHẬT KÝ LÀM VIỆC & LÊN KẾ HOẠCH - 2026-10-06

**Thành viên:** Vuong & AI
**Nội dung:** Thảo luận về kiến trúc hệ thống và hướng đi tiếp theo

## 1. Xác nhận Code Hiện Tại
- Toàn bộ code xử lý logic `decision_engine.py` (Vùng quan tâm ROI, lọc vật thể, vị trí trái/phải, tạo câu cảnh báo) đã được kiểm duyệt, pass 100% Unit Tests.
- Các module này đã được gộp thành công vào luồng chính (`backend/main.py`), phục vụ trực tiếp cho API `/api/v1/vision/analyze` và WebSocket.

## 2. Thống nhất Kế hoạch tính năng mới
- **GPS & Navigation:** Nhất trí rằng GPS sẽ hoạt động song song và độc lập với AI Vision. AI Vision xử lý vật cản vi mô dưới đường, GPS xử lý lộ trình vĩ mô. -> Team có thể code GPS trên Flutter ngay lúc này.
- **Phân vùng đường đi (Segmentation):** Tạm gác lại việc dùng model COCO mặc định. Sắp tới sẽ tiến hành thu thập dữ liệu (ổ gà, gạch người khiếm thị) để train lại (Fine-tune) mô hình YOLOv8-seg chuyên dụng cho dự án.

## 3. Quy chuẩn Git
- Cập nhật nhánh Git thành: `feat/Vuong-backend-vision-engine`.
