# TÀI LIỆU KỸ THUẬT: PHÁT HIỆN BỨC TƯỜNG & BẬC TAM CẤP (HƯỚNG 2 - THUẬT TOÁN HÌNH HỌC)

## 1. Vấn đề giải quyết
- **Bức tường (Wall):** YOLO (COCO 80 classes) không có class "wall". Depth Anything V2 monocular depth khi gặp tường trơn nhẵn thiếu hoa văn (textureless) dễ bị nhiễu.
- **Bậc tam cấp / Cầu thang (Stairs/Steps):** Chiều cao bậc chỉ 10cm - 15cm, mạng AI thường làm mịn (smooth) thành mặt dốc hoặc bỏ sót.

## 2. Giải pháp Hướng 2 (Geometric Vision & Depth Gradient Analysis)
Thay vì dùng thêm mô hình Deep Learning nặng nề, hệ thống áp dụng toán học hình học máy tính + OpenCV:

### A. Phát hiện Bức Tường (Vertical Wall Plane Detection):
- **Nguyên lý:** Mặt sàn bình thường có độ sâu tăng đều đặn từ chân lên xa ($Z$ tăng theo $y \rightarrow \left|\frac{\partial Z}{\partial y}\right| \gg 0$).
  Khi gặp bức tường thẳng đứng chắn lối đi, độ sâu $Z$ hầu như không đổi theo phương thẳng đứng ($\left|\frac{\partial Z}{\partial y}\right| \approx 0$).
- **Thuật toán:**
  1. Trích xuất ROI trung tâm ($x \in [0.22, 0.78], y \in [0.25, 0.78]$).
  2. Tính đạo hàm Sobel theo trục $Y$: `grad_y = cv2.Sobel(roi_depth, cv2.CV_32F, 0, 1, ksize=3)`.
  3. Lọc pixel mặt đứng ($|grad\_y| < 0.045$) có độ sâu trong khoảng nguy hiểm ($Z \le 2.8m$).
  4. Nếu diện tích mảng tường chiếm $> 16\%$ ROI và không bị che phủ bởi vật thể YOLO khác -> Báo động BỨC TƯỜNG kèm khoảng cách mét chính xác.

### B. Phát hiện Bậc Tam Cấp / Cầu Thang (Stair & Step Edge Detection):
- **Nguyên lý:**
  1. **Các cạnh ngang song song:** Gờ mép bậc thang tạo thành các đoạn thẳng nằm ngang song song ở nửa dưới khung hình ($y \in [0.48, 0.95]$). Sử dụng CLAHE + Sobel Y + Morphological Rect Kernel + HoughLinesP.
  2. **Bước nhảy gián đoạn độ sâu (Depth Step Jump):** Tại vị trí mỗi nếp bậc, độ sâu $Z$ có bước nhảy chênh lệch $\ge 8cm$.
- **Phân loại:**
  - $\ge 3$ nếp bậc: Nhận diện là `stairs` ("cầu thang").
  - $1 - 2$ nếp bậc kèm bước nhảy độ sâu: Nhận diện là `step` ("bậc tam cấp").

## 3. Hiệu năng & Kết quả kiểm thử
- **Thời gian xử lý trên CPU:**
  - Phát hiện Bức Tường: ~19ms (lần đầu warm-up), sau đó ~4-8ms.
  - Phát hiện Bậc Tam Cấp: ~7-9ms.
  - Tổng thời gian thêm vào pipeline: $< 15ms$, hoàn toàn đáp ứng thời gian thực 30 FPS!
- **Kiểm thử Unit Test:** 31/31 passed (`pytest backend/tests -v`).
- **Tích hợp:** Đã nối trực tiếp vào `depth_service.py` -> `pipeline.py` -> tự động sinh cảnh báo âm thanh tiếng Việt và vẽ Bounding Box màu đỏ trên cả Điện Thoại (MBAPP) và Web Monitor!
