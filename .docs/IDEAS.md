# Kho Lưu Trữ Ý Tưởng (Ideas Board) 💡

File này dùng để các thành viên trong team ghi chú nhanh bất kỳ ý tưởng điên rồ, sáng tạo hoặc đột phá nào vừa nghĩ ra trong lúc code.
Không cần quan tâm ý tưởng đó có khả thi hay không, cứ ném vào đây để cả team cùng thảo luận!

---

## 1. Vùng An Toàn (Region of Interest - ROI) - Đề xuất bởi Vương
- **Mô tả:** Thay vì quét toàn màn hình và cảnh báo liên tục, AI chỉ xét một "Hình quạt" (hoặc hình thang) nằm ngay phía trước mũi chân người dùng. 
- **Lý do:** Những vật cản hai bên đường không gây nguy hiểm, cảnh báo sẽ làm người dùng bị nhiễu thông tin.
- **Cách làm dự kiến:** Dùng Toán hình học để lọc các Box tọa độ (Bounding Box) trả về từ YOLO. Nằm ngoài vùng quạt -> Bỏ qua. Nằm trong vùng quạt -> Hét toáng lên.
- **Trạng thái:** Sắp triển khai ở Phase tiếp theo.

## 2. Nhận diện Vỉa Hè và Đường Xe Chạy (Image Segmentation / Lane Detection)
- **Mô tả:** Thay vì chỉ vẽ hộp vuông (Bounding Box), AI sẽ phân mảng pixel để kẻ vạch ranh giới giữa Vỉa hè (an toàn) và Lòng đường (nguy hiểm). Báo hiệu rẽ trái/phải khi đường cong.
- **Lý do:** Giúp người khiếm thị đi đúng trên vỉa hè và biết khi nào đến khúc cua hoặc ngã tư có vạch kẻ đường.
- **Cách làm:** Chuyển từ mô hình YOLOv8 Detection sang mô hình YOLOv8 Segmentation (Phân mảng) hoặc huấn luyện một mô hình Lane Detection chuyên biệt.

## 3. Nhận diện Đường Dẫn Khối Nổi (Tactile Paving)
- **Mô tả:** Nhận diện các tấm gạch có gờ nổi màu vàng/đỏ trên vỉa hè dành riêng cho người khiếm thị.
- **Lý do:** Hướng dẫn người dùng bám theo dải gạch nổi này để di chuyển cực kỳ an toàn tại các thành phố hiện đại.
- **Cách làm:** Thu thập ảnh chụp các vạch gạch nổi này và đưa vào quá trình Fine-tuning mô hình AI.

## 4. Xử lý logic trước khi gọi API (Decision Engine)
- **Mô tả:** Dữ liệu AI trả về là các tọa độ khô khan (vd: `curve=30deg, border=left`). Cần có hàm trung gian dịch nó thành câu: *"Đường rẽ cong sang trái, hãy rẽ theo"*.
- **Lý do:** Các API Giọng nói (hoặc LLM) của công ty không thể tự nhìn thấy tọa độ để hiểu. Việc chúng ta dịch sẵn ra câu chữ ngắn gọn giúp hệ thống chạy nhanh hơn và tiết kiệm chi phí gọi API.
