# BUG #05: Lỗi mất hoàn toàn nhận diện (chỉ báo "Phía trước an toàn"), không nhận diện bức tường và có viền đỏ nhưng không đọc giọng nói

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC TRIỆT ĐỂ
- **Ngày phát hiện:** 2026-10-29
- **Các tệp bị ảnh hưởng:**
  - [`backend/main.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/main.py)
  - [`backend/app/schemas/vision.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/schemas/vision.py)
  - [`backend/app/vision/depth_service.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/vision/depth_service.py)
  - [`backend/app/vision/decision_engine.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/vision/decision_engine.py)
  - [`MBAPP/app/src/main/java/com/example/ai_hkdn/service/TtsManager.kt`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/MBAPP/app/src/main/java/com/example/ai_hkdn/service/TtsManager.kt)
  - [`MBAPP/app/src/main/java/com/example/ai_hkdn/service/WebSocketService.kt`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/MBAPP/app/src/main/java/com/example/ai_hkdn/service/WebSocketService.kt)

---

## 1. Nguyên nhân gốc rễ (Root Causes)

### Nguyên nhân A: Tại sao giờ chỉ báo "Phía trước an toàn" dù ghế, cặp, bàn ngay trước mặt?
Trong [`backend/main.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/main.py), khi phát hiện vật cản nguy hiểm (`is_danger == True`), đoạn code mới thêm đã gọi:
```python
det_item = Detection(...)
setattr(det_item, "distance_m", dist_m)
```
Tuy nhiên, class `Detection` trong [`backend/app/schemas/vision.py`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/backend/app/schemas/vision.py) là `BaseModel` của **Pydantic v2** và chưa khai báo trường `distance_m`.
$\rightarrow$ Ngay khi có bất kỳ vật cản nào (ghế, bàn, cặp, tường, người) xuất hiện trước camera, lệnh `setattr` làm **crash ngầm hàm `_analyze_frame_sync`** với lỗi:
```
ValueError: "Detection" object has no field "distance_m"
```
Khi đó WebSocket trả về `{"success": False, "error": "..."}` khiến App điện thoại bỏ qua hoàn toàn frame có vật cản! Chỉ những frame **không có vật cản** (`is_danger == False`) mới không chạy vào dòng `setattr` và trả về `"Phía trước an toàn."`.

### Nguyên nhân B: Tại sao lúc trước có viền đỏ nhận biết được ghế/chai nhưng điện thoại không nói?
1. **Lệch điều kiện giữa màn hình vẽ (ARHudOverlay) và danh sách đọc (roi_detections):**
   - Trên điện thoại, `ARHudOverlay.kt` vẽ viền đỏ cho mọi vật ở giữa (`position == "center"`).
   - Nhưng ở Backend, `roi_detections` lại lọc bỏ các vật đặt trên bàn (`y2 < 40%` chiều cao ảnh như cái chai, cái cặp) hoặc vật có độ tin cậy `0.25 - 0.39`, nên `roi_detections` bị rỗng $\rightarrow$ có viền đỏ nhưng câu trả về vẫn là `"Phía trước an toàn."`.
2. **Câu `"Phía trước an toàn."` khóa cứng bộ đọc TTS trên Android (`TtsManager.kt`):**
   - Trong `TtsManager.kt`, câu `"Phía trước an toàn."` được đọc liên tục mỗi 2.5 giây làm `tts.isSpeaking` luôn bằng `true`.
   - Điều kiện ngắt câu cũ chỉ kiểm tra `priority == "DANGER"`, trong khi Backend chỉ trả về `priority == "HIGH"` hoặc `"MEDIUM"`. Hệ quả là khi có câu cảnh báo ghế/bàn/chai nước gửi về, `TtsManager` thấy đang đọc dở câu `"Phía trước an toàn."` nên **bỏ qua luôn câu cảnh báo vật cản**!
   - Ngoài ra, `WebSocketService.kt` gọi `ttsManager.speak()` trực tiếp từ luồng nền của OkHttp thay vì `MainLooper`.

### Nguyên nhân C: Tại sao không nhận diện được "Bức tường" dù đã có model Depth?
- Trong `depth_service.py` mới chỉ nạp `Depth-Anything-V2` mà **chưa nạp `SegFormer` (`nvidia/segformer-b0-finetuned-ade-512-512`)** để phân biệt bề mặt đứng trước mặt là **bức tường (`wall`)**, **cửa (`door`)**, **cầu thang (`stairs`)** hay mặt sàn (`floor`).
- Vùng quét của `detect_unknown_obstacles` trước đó chỉ lấy `y = 45% -> 85%` (vùng dưới thấp) nên bỏ sót bức tường trước tầm mắt (`y = 25% -> 80%`).

---

## 2. Cách đã khắc phục
1. **Sửa Schema Pydantic v2 (`vision.py`) & `backend/main.py`:**
   - Khai báo đầy đủ `distance_m`, `class_name_vn`, `in_roi`, `is_danger` và `model_config = {"extra": "allow"}` trong `Detection`.
   - Hạ ngưỡng lọc của ghế (`chair`), bàn (`table`, `dining table`), cặp (`backpack`, `handbag`), chai nước (`bottle`), cốc (`cup`) xuống `0.25` và đảm bảo mọi vật xuất hiện trên màn hình đều được đưa vào danh sách đọc kèm khoảng cách mét.
2. **Tích hợp SegFormer + Depth Anything V2 + Pothole vào `depth_service.py`:**
   - Khi đứng trước tường hoặc vật cản (< 2.5m), kết hợp bản đồ độ sâu (Depth Anything V2) và phân đoạn ngữ nghĩa (SegFormer) để báo chính xác: **`"Phía trước 1 mét rưỡi có bức tường."`**, **`"Phía trước 2 mét có cửa ra vào."`**, **`"Phía trước 1 mét có ghế."`**.
3. **Sửa `TtsManager.kt`, `WebSocketService.kt`, `ARHudOverlay.kt` trên App Android (`MBAPP`):**
   - Câu `"Phía trước an toàn"` bị giãn cách tối소 8 giây và **bị ngắt ngay lập tức (`QUEUE_FLUSH`)** ngay khi có bất kỳ cảnh báo vật cản nào (ghế, bàn, chai nước, cặp, bức tường...).
   - Hiển thị nhãn tiếng Việt + số mét trực tiếp trên viền đỏ AR HUD (ví dụ: `ghế 1.5m`, `bức tường 1.2m`).
