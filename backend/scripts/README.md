# 🛠️ BACKEND UTILITIES & SCRIPTS

Thư mục này chứa các công cụ hỗ trợ kiểm thử, đánh giá offline và demo cho AI Guidance System:

| Script | Mô tả | Cách chạy |
| :--- | :--- | :--- |
| `test_media.py` | Kiểm thử pipeline trên ảnh/video offline, vẽ khung và xuất báo cáo | `python backend/scripts/test_media.py --file path/to/video.mp4` |
| `test_all_features.py` | Bộ kiểm thử tích hợp toàn bộ tính năng và tự động xuất báo cáo kiểm thử | `python backend/scripts/test_all_features.py` |
| `test_video_tracking.py` | Kiểm thử ByteTrack và ROI hành lang di chuyển trên video | `python backend/scripts/test_video_tracking.py` |
| `test_video_segmentation.py` | Kiểm thử phân mảng hình ảnh YOLOv8-seg trên video | `python backend/scripts/test_video_segmentation.py` |
| `test_segmentation.py` | Kiểm thử phân mảng trên ảnh tĩnh mẫu | `python backend/scripts/test_segmentation.py` |
| `test_yolo.py` | Kiểm thử nhận diện vật thể YOLO đơn lẻ | `python backend/scripts/test_yolo.py` |
| `test_ws.py` | Kiểm tra kết nối nhanh tới WebSocket `/api/v1/vision/stream` | `python backend/scripts/test_ws.py` |
| `test_websocket.html` | Trang HTML giao diện web kiểm tra stream WebSocket thủ công | Mở trực tiếp trên trình duyệt |
