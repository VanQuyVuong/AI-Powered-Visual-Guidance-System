# GIAI ĐOẠN 2 - NHIỆM VỤ G3 (VIDEO TRACKING)

**Trạng thái:** Đang tiến hành 🏃‍♂️

## Mục tiêu
Nâng cấp khả năng Nhận diện từ Ảnh tĩnh (Image) lên Video. Thay vì chỉ biết "Có cái xe", AI phải biết "Xe số 1 đang di chuyển từ trái sang phải, Xe số 2 đang đứng im". 

## Checklist hoàn thành
- [ ] Viết script `test_video_tracking.py` để test tính năng bám đuổi vật thể (Tracking) của YOLO.
- [ ] Xử lý video đầu vào (mp4) và xuất ra video đã vẽ hộp (Bounding Box) kèm ID của từng vật thể.
- [ ] (Tương lai) Tích hợp luồng Video vào FastAPI (có thể dùng WebSocket) để chạy Real-time.

## Ghi chú trong quá trình làm
- Dùng tính năng `model.track()` có sẵn của thư viện `ultralytics` kết hợp thuật toán ByteTrack để không phải train lại model từ đầu.
