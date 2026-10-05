# THƯ MỤC NHẬT KÝ CÁ NHÂN CỦA CÁC THÀNH VIÊN

Đây là nơi để mỗi thành viên trong team tự lưu lại "Lịch sử làm việc" hoặc "Lịch sử chat với AI" trên máy của mình. 
Nhờ việc này, các thành viên khác có thể đọc lại và hiểu được: "Hôm qua thằng Vương nó bắt con AI làm những cái gì để ra được đống code đó?".

## Quy tắc (Role) dành cho Team:
1. Mỗi người tự tạo một thư mục mang tên mình ở trong thư mục này (Ví dụ: `Vuong/`, `Huy/`, `Nam/`).
2. Khi bạn làm việc với AI để tạo ra tính năng mới, hãy copy đoạn prompt quan trọng hoặc tóm tắt cách bạn đã "sai khiến" AI và dán vào 1 file `.md` trong thư mục của bạn.
3. Không sửa file của người khác.

---

### Mẫu tham khảo (Template):
**File:** `.worklog/sessions/Vuong/2023-10-04_websocket_tracking.md`
**Nội dung:**
> Hôm nay tao đã yêu cầu AI chuyển từ gửi nguyên video sang gửi từng frame qua WebSocket. 
> Quá trình:
> - AI gợi ý dùng WebSocket để đảm bảo Real-time.
> - Xảy ra lỗi Camera bị chiếm dụng (Device in use) -> Giải quyết bằng cách tắt app khác.
> - Xảy ra lỗi mất kết nối vì chưa cài thư viện `websockets` vào `.venv` -> Đã xử lý.
