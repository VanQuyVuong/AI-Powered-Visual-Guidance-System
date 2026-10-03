# WORKLOG & STATE RECOVERY (NHẬT KÝ TIẾN ĐỘ & PHỤC HỒI TRẠNG THÁI)

Thư mục này được sinh ra với 2 mục đích:
1. Cho con người (Team) biết dự án đã code đến đâu, còn nợ tính năng gì.
2. Dùng làm **"Điểm lưu trữ" (Save Point)** cho AI. Trong trường hợp máy tính bị tắt hoặc khởi tạo lại phiên làm việc (Chat mới), AI chỉ cần đọc thư mục này là biết hệ thống đã xây dựng đến đâu mà không cần giải thích lại từ đầu.

## Cấu trúc bên trong thư mục này:

### 1. 📁 `progress/` (Tiến độ)
Chứa các file đánh dấu công việc đã hoàn thành.
- **`00-OVERVIEW.md`**: Bản đồ lộ trình tổng quan (Roadmap). Đây là file quan trọng nhất, AI sẽ nhìn vào đây để biết hôm nay phải code cái gì tiếp theo.
- **`01-G1-YOLO-detection.md`**, **`02-G2-decision-engine.md`**...: Các file nhật ký chi tiết cho từng giai đoạn nhỏ.

### 2. 📁 `specs/` (Đặc tả kỹ thuật)
Chứa các file mô tả cấu trúc, luồng hoạt động và quy tắc code đã được thống nhất sau khi phân tích. Các file này được chắt lọc từ thư mục `.docs/` gốc để AI đọc nhanh hơn và không vi phạm quy tắc (Ví dụ: `01-system-architecture.md`).

### 3. 📁 `debug/` (Nhật ký sửa lỗi)
Nơi ghi chép lại các lỗi lớn từng gặp phải và cách sửa, để sau này nếu gặp lại lỗi tương tự, AI hoặc người lập trình không đi vào vết xe đổ. (Hiện tại chưa có lỗi lớn nào cần ghi lại).

---
> 🤖 **Prompt cho AI ở phiên làm việc mới:**
> "Chào bạn, đây là dự án AI Visual Guidance System. Bạn hãy đọc file `README.md` ở thư mục gốc, sau đó đọc file `.worklog/progress/00-OVERVIEW.md` để lấy lại trạng thái làm việc hiện tại, và chúng ta sẽ làm task tiếp theo chưa được tick [x]."
