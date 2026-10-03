# HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN (DÀNH CHO TEAM)

Dự án: **AI-Powered-Visual-Guidance-System**
Phiên bản hiện tại: MVP (Sprint 2)

---

## 1. Yêu cầu hệ thống
- Đã cài đặt **Python 3.9** trở lên.
- Có cài đặt **Git**.

## 2. Cách cài đặt dự án (Setup)
Thành viên mới (hoặc khi cài lại máy) cần làm theo đúng thứ tự sau:

**Bước 1: Clone code về máy**
```bash
git clone https://github.com/VanQuyVuong/AI-Powered-Visual-Guidance-System.git
cd AI-Powered-Visual-Guidance-System
```

**Bước 2: Tạo môi trường ảo (Virtual Environment)**
Tuyệt đối không cài thư viện thẳng vào Windows, hãy dùng môi trường ảo để code không bị xung đột.
```bash
python -m venv .venv
```

**Bước 3: Kích hoạt môi trường ảo**
- Trên Windows (Command Prompt / PowerShell):
  ```bash
  .venv\Scripts\activate
  ```
- Trên Mac/Linux:
  ```bash
  source .venv/bin/activate
  ```
*(Thành công là khi bạn thấy chữ `(.venv)` xuất hiện ở đầu dòng lệnh).*

**Bước 4: Cài đặt thư viện**
```bash
pip install -r backend/requirements.txt
```
*(Thư viện gồm có: `ultralytics` cho YOLO, `opencv-python` cho xử lý ảnh, `fastapi` & `uvicorn` cho server Web API, `python-multipart` để nhận file ảnh).*

---

## 3. Cách chạy Server Backend
Phải đảm bảo đã kích hoạt môi trường ảo `(.venv)` trước khi chạy.
Từ thư mục gốc của dự án, chạy lệnh:
```bash
python backend/main.py
```
- Server sẽ chạy tại: `http://localhost:8000`
- Giao diện test API (Swagger UI): `http://localhost:8000/docs`

---

## 4. Cấu trúc thư mục dự án
Để mọi người không bị rối, đây là giải thích chức năng từng thư mục:

*   📁 **`.docs/`**: Chứa các tài liệu thiết kế ban đầu (System Analysis, API Contract, Team Tasks...). Đây là luật của dự án.
*   📁 **`.worklog/`**: Thư mục cực kỳ quan trọng dùng để lưu trữ tiến độ làm việc, lỗi, và các file phục hồi trạng thái cho AI. (Xem chi tiết trong `.worklog/README.md`).
*   📁 **`backend/`**: Toàn bộ code Python.
    *   `app/api/`: Sẽ chứa code định tuyến API sau này nếu mở rộng.
    *   `app/schemas/`: Định nghĩa kiểu dữ liệu JSON (đầu vào, đầu ra).
    *   `app/vision/`: Chứa các thuật toán AI (YOLO, Decision Engine).
    *   `main.py`: File gốc để bật server.
    *   `requirements.txt`: Danh sách các thư viện cần cài.
*   📁 **`test_images/`**: Thư mục chứa các ảnh mẫu dùng để test API. (Thư mục này bị bỏ qua bởi Git).
*   📄 **`.gitignore`**: Chặn đẩy các file nặng và nhạy cảm (như `.venv`, file model `*.pt`, file ảnh test) lên Github.
