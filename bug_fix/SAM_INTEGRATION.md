# Tích hợp SAM (Segment Anything) - MobileSAM

## Trạng thái
Đã cài và test chạy được trên CPU laptop. **Chưa gắn vào `backend/main.py`** (tránh đụng phiên AI khác đang sửa file đó).

## File mới (không sửa file cũ nào)
| File | Vai trò |
|---|---|
| `backend/app/vision/sam_service.py` | Nạp lười MobileSAM qua Ultralytics, phân đoạn theo hộp / điểm / toàn ảnh, xuất đa giác nén |
| `backend/app/vision/sam_router.py` | API `POST /api/v1/vision/segment`, `GET /api/v1/vision/segment/status` |
| `backend/scripts/test_sam.py` | Test độc lập: YOLO -> SAM -> ảnh `test_images/result_sam.jpg` |
| `mobile_sam.pt` (gốc dự án, ~39MB) | Trọng số, tự tải lần đầu, đã bị `.gitignore` (`*.pt`) |

## Cách bật API (thêm 2 dòng vào `backend/main.py`, sau `app = FastAPI(...)`)
```python
from app.vision.sam_router import router as sam_router
app.include_router(sam_router)
```

## Kết quả đo (CPU laptop, ảnh 678x452)
- Nạp model: ~4s (một lần, lười - không làm chậm lúc khởi động server).
- Phân đoạn theo điểm chạm: ~1.4s.
- Tự động (YOLO + 12 mặt nạ): ~6.5s lần đầu (đã gồm khởi động YOLO).
- => **KHÔNG dùng cho luồng realtime 30 FPS**. Chỉ dùng theo yêu cầu (nút "Phân tích chi tiết", tìm đồ vật).

## Đổi sang SAM 3
Đặt biến môi trường `SAM_MODEL=sam3.pt`. Trọng số SAM 3 do Meta cấp quyền (gated), phải tự xin và tải về đặt ở gốc dự án.
Chưa kiểm chứng SAM 3 trong môi trường này.

## Chưa làm
- Gắn router vào `main.py`.
- Vẽ mặt nạ trên app (Kotlin `ARHudOverlay.kt` / Flutter).
- Gom vật theo bàn lớn (spatial clustering) - có thể dùng mặt nạ SAM để tính "nằm trên bàn" chính xác hơn hộp YOLO.
