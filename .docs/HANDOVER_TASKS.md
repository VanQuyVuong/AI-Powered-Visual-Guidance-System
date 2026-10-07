# 📌 THÔNG BÁO BÀN GIAO CODE VÀ CÁC TASK TIẾP THEO

Anh em chú ý, hiện tại bộ khung MVP (Bộ lõi) của dự án **AI Visual Guidance** đã hoàn thiện 100%. Các luồng từ App điện thoại đến Backend AI đã thông suốt với nhau.

## ✅ CÁC TÍNH NĂNG ĐÃ LÀM XONG (Hoàn thành bởi Vương)
1. **App Flutter:** Đã có giao diện Camera quét liên tục 1.5s/lần, kết nối WebSocket cực mượt, không nóng máy.
2. **AI Vision (YOLO + OpenCV):** Đã nhận diện được vật cản, tính toán Vùng An Toàn (ROI), và có thuật toán OpenCV tự dò đường gạch vàng cho người khiếm thị (Tactile Paving).
3. **Tính năng phụ trợ:** Đã code xong Định vị GPS, Đọc biển báo/văn bản (OCR), và Trợ lý ảo ra lệnh bằng giọng nói (STT).
4. **Quy trình:** Đã setup xong `.gitignore` bảo mật và chia thư mục chuẩn.

---

## 🚀 DANH SÁCH CÔNG VIỆC CẦN LÀM TIẾP THEO (GIAI ĐOẠN ĐẮP THỊT)
Đề nghị anh em chia nhau các Task sau để ráp vào hệ thống:

### 👨‍💻 Task 1: Train AI (Dành cho thành viên phụ trách AI/Data)
*   **Việc cần làm:** Đi chụp ảnh/quay video hoặc lên mạng tải Dataset về các chướng ngại vật thực tế ở Việt Nam (Ổ gà, nắp cống hở, vạch kẻ đường, biển báo, lề đường).
*   **Mục tiêu:** Fine-tune lại mô hình YOLO (chạy trên Google Colab) để xuất ra file `guidance_model.pt` của riêng nhóm thay cho cái model mặc định hiện tại.

### 🗺️ Task 2: Bản đồ chỉ đường (Dành cho thành viên phụ trách Logic/Flutter)
*   **Việc cần làm:** Lên trang `openrouteservice.org` tạo 1 tài khoản miễn phí để lấy API Key (Dịch vụ tìm đường dành riêng cho người đi bộ - không cần thẻ VISA).
*   **Mục tiêu:** Viết hàm truyền tọa độ GPS hiện tại lên OpenRouteService để lấy lộ trình (Ví dụ: "Đi thẳng 200m rẽ phải"), sau đó kết hợp với Camera AI để tạo ra tính năng Dẫn đường chi tiết từng bước (Sensor Fusion). Đọc ý tưởng số 5 trong file `IDEAS.md` để hiểu chi tiết.

### 🎤 Task 3: Tích hợp API Công ty (Dành cho thành viên phụ trách API)
*   **Việc cần làm:** Liên hệ lấy API Key của công ty (Blaze TTS / STT).
*   **Mục tiêu:** Mở file `tts_service.dart` và `voice_service.dart` trên Flutter, sửa lại code để gọi thẳng qua API của công ty thay vì dùng thư viện offline của máy.

### 🏃 Task 4: Mang ra đường Test (Dành cho toàn bộ team)
*   **Việc cần làm:** Cài App vào một cái điện thoại Android thật. Ra vỉa hè bật App lên đi bộ thử.
*   **Mục tiêu:** Đánh giá xem AI đọc có kịp không, thuật toán dò đường gạch vàng (OpenCV) có bị lỗi khi ra nắng gắt hay bóng râm không để về tinh chỉnh lại thông số màu HSV trong file `backend/app/vision/tactile_paving_detector.py`.
