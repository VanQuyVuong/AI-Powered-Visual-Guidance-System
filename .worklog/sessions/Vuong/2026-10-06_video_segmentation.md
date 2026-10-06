# NHẬT KÝ LÀM VIỆC - 2026-10-06

**Tính năng:** Thử nghiệm Video Segmentation
**Người thực hiện:** AI & Vuong

## Quá trình thực hiện:
- Phát hiện đã có file `sample_video.mp4` trong thư mục `test_images`.
- AI tạo script `test_video_segmentation.py` trong thư mục `backend/` để chạy mô hình `yolov8n-seg.pt` trên video.
- Xử lý từng frame video, vẽ các mảng màu (segmentation masks) lên các vật thể (như người, xe cộ) và lưu lại thành `result_segmentation_video.mp4`.
- Quá trình này chứng minh khả năng phân mảng của AI để chuẩn bị cho việc phân tích vỉa hè, lòng đường (như ý tưởng số #2 trong IDEAS.md).

## Lưu ý:
- Vì mô hình `yolov8n-seg.pt` là bản mặc định nên nó sẽ phân mảng các class COCO (như xe cộ, người...). Để phân vùng chính xác vỉa hè và lòng đường, trong tương lai cần fine-tune model trên dataset đường phố hoặc dùng mô hình chuyên dụng cho Lane/Road segmentation.
