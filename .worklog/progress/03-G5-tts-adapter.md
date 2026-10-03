# SPRINT 3 - NHIỆM VỤ G5 (TÍCH HỢP BLAZE TTS API)

**Trạng thái:** Đang tiến hành 🏃‍♂️

## Mục tiêu
Tạo một Adapter (Cầu nối) giữa Backend của chúng ta và API giọng nói của doanh nghiệp (Blaze TTS). 
Mục đích là biến câu chữ tiếng Việt thành file âm thanh để truyền về cho App Flutter phát lên.

## Checklist hoàn thành
- [x] Cập nhật file `schemas/vision.py` để thêm trường `audio_url` vào phản hồi.
- [x] Tạo file `app/services/voice_service.py` chứa Interface chuẩn.
- [x] Viết MockVoiceAdapter (Bộ giả lập API) trả về file âm thanh mẫu (trong lúc chờ tài liệu thật từ doanh nghiệp).
- [x] Tích hợp Voice Adapter vào luồng xử lý chính trong `main.py`.
- [ ] Test lại API trên Swagger UI xem đã trả về link audio chưa.

## Ghi chú trong quá trình làm
- Theo luật trong `API_CONTRACT.md`, tuyệt đối KHÔNG tự đoán endpoint hay cấu trúc request của doanh nghiệp khi chưa có tài liệu chính thức. Do đó bắt buộc phải dùng kỹ thuật Mocking (Giả lập) ở giai đoạn này.
