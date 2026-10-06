# SPRINT 2 - NHIỆM VỤ G2 (DECISION ENGINE & FASTAPI)

**Trạng thái:** Đang tiến hành 🏃‍♂️

## Mục tiêu
Dựa trên kết quả Bounding Box (tọa độ x, y) của YOLO, tính toán xem vật thể đang nằm ở bên Trái, Phải hay Ở Giữa khung hình. Từ đó sinh ra câu cảnh báo tiếng Việt và bọc tất cả lại thành một API hoàn chỉnh.

## Checklist hoàn thành
- [x] Định nghĩa các cấu trúc dữ liệu (Schemas) chuẩn API Contract bằng Pydantic.
- [x] Viết hàm tính toán vị trí `determine_position(x1, x2, image_width)`.
- [x] Viết hàm sinh câu tiếng Việt `generate_guidance(detections)`.
- [x] Khởi tạo FastAPI Server (Endpoint `POST /api/v1/vision/analyze`).
- [x] Tích hợp YOLO vào FastAPI.
- [x] Test API bằng pytest (28 tests). Phát hiện và sửa bug priority (`.capitalize()` vs lowercase check).

## Ghi chú trong quá trình làm
- Thuật toán chia khung hình làm 3 phần bằng nhau: 0 -> 1/3 (Trái), 1/3 -> 2/3 (Giữa), 2/3 -> 3/3 (Phải).
