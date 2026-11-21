# HƯỚNG DẪN BỎ ẢNH KIỂM THỬ & HIỆU CHUẨN VÀO `ImagesTest/`

Bạn có thể bỏ ảnh chụp thực tế từ điện thoại vào trực tiếp thư mục này, hoặc chia vào các thư mục con dưới đây để hệ thống **tự động chấm điểm đúng/sai** theo từng nhóm tình huống:

- `cau_thang/` — Ảnh cầu thang đi lên, đi xuống, bậc thềm, bậc tam cấp (kỳ vọng bắt được `stairs` / `step`).
- `nen_gach/` — Ảnh nền gạch men, hành lang trống (kỳ vọng `CLEAR` / không báo nhầm thành cầu thang hay bức tường).
- `ho_sau/` — Ảnh hố sâu, ổ gà, bậc hụt chân (kỳ vọng bắt được `dropoff` / `pothole`).
- `vat_can_xa/` — Ảnh bàn, ghế, người ở khoảng cách **> 2.4m (4–5 viên gạch trở lên)** (kỳ vọng báo đúng khoảng cách thực và **không cảnh báo nguy hiểm giả**).
- `tuong_cot/` — Ảnh bức tường gần, cột nhà, tấm bạt chắn ngang (kỳ vọng bắt được `wall` / `column` / `obstacle`).

### Mẹo đo & hiệu chuẩn khoảng cách chính xác cho riêng camera điện thoại của bạn
Nếu đặt tên file có chứa khoảng cách đo thật ngoài đời theo định dạng `<ten>_<so>m.jpg` (ví dụ: `ghe_2.8m.jpg`, `ban_3.0m.png`, `cauthang_1.5m.jpg`), script sẽ tự đối chiếu sai số mét giữa thực tế và model, rồi đề xuất (hoặc ghi thẳng vào `.env`) hệ số hiệu chuẩn **`DEPTH_SCALE_FACTOR`** chuẩn nhất cho camera điện thoại của bạn.

### Lệnh chạy kiểm thử toàn bộ ảnh trong `ImagesTest/`:
```powershell
.venv\Scripts\python.exe backend/evaluate_images_folder.py
```
Tùy chọn tự ghi hệ số hiệu chuẩn `DEPTH_SCALE_FACTOR` vào `.env`:
```powershell
.venv\Scripts\python.exe backend/evaluate_images_folder.py --write-env
```
*(Ảnh vẽ bounding box + độ sâu + cảnh báo giọng nói sẽ lưu tại `ImagesTest/_results/`)*
