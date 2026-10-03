# TEAM TASKS

**Team:** 2 người\
**Trình độ:** Cả hai đều newbie AI.

## 1. Cách chia

Không chia cứng một người chỉ AI và một người chỉ Flutter. Chia
owner/reviewer để cả hai đều hiểu end-to-end.

-   **Người 1:** AI Backend + Vision + Decision Engine
-   **Người 2:** Flutter + API Integration + Audio/UI
-   **Cả hai:** Integration, test, debug, review, demo.

## 2. MVP

``` text
Flutter → Image → Backend → YOLO → Decision → Guidance
→ Blaze TTS → Audio → Flutter
```

Chưa làm: Tracking, OCR, STT, Voice Assistant, Translation, News Audio,
GPS/Maps, custom model, VLM, smart hat thật.

## 3. Người 1 --- AI Backend / Vision

### G1 --- YOLO

``` text
image → YOLO → detection JSON
```

Học: inference, class, confidence, bounding box. Không train model.

### G2 --- Decision Engine

``` text
bbox + image width → left / center / right
detection → priority → guidance text
```

### Backend

``` http
POST /api/v1/vision/analyze
```

### Test

-   ảnh hợp lệ
-   nhiều object
-   không có object
-   left/center/right
-   confidence thấp
-   ảnh lỗi

### Không làm ngay

Tracking, OCR, navigation, custom training, VLM.

## 4. Người 2 --- Flutter / Integration

-   Tạo Flutter prototype.
-   Camera/Gallery.
-   Preview.
-   Gọi `POST /api/v1/vision/analyze`.
-   Loading/success/error.
-   Hiển thị detection/guidance.
-   Audio player.
-   Hỗ trợ test API thật.

## 5. Owner / Reviewer

  Module            Owner     Reviewer
  ----------------- --------- ----------
  Python setup      Người 1   Người 2
  YOLO              Người 1   Người 2
  Position          Người 1   Người 2
  Guidance          Người 1   Người 2
  FastAPI           Người 1   Người 2
  TTS adapter       Người 1   Người 2
  Flutter project   Người 2   Người 1
  Camera/Gallery    Người 2   Người 1
  HTTP client       Người 2   Người 1
  Result UI         Người 2   Người 1
  Audio player      Người 2   Người 1
  Integration       Cả hai    Cả hai

## 6. Sprint

### Sprint 1

Người 1: Python → YOLO → chạy 1 ảnh → JSON.

Người 2: Flutter → chọn ảnh → preview.

### Sprint 2

Người 1: Position → Guidance → FastAPI.

Người 2: HTTP client → nhận JSON → hiển thị guidance.

### Sprint 3

Người 1: TTS adapter.

Người 2: Audio player.

### Sprint 4

Cả hai:

``` text
Camera → Backend → YOLO → Decision → TTS → Audio
```

## 7. Git

``` text
feature/vision-yolo
feature/decision-engine
feature/backend-api
feature/flutter-camera
feature/flutter-api
feature/company-tts
feature/audio-player
```

Commit:

``` text
feat(vision): add YOLO image detection
feat(guidance): add bbox position classifier
feat(api): add vision analyze endpoint
feat(flutter): add image picker
feat(tts): add company TTS adapter
feat(audio): add audio playback
test(guidance): add position tests
```

## 8. Quy tắc

1.  Không sửa contract mà không báo.
2.  Không commit API key.
3.  Không thêm dependency lớn khi chưa thống nhất.
4.  Không train model cho MVP.
5.  Không làm UI lớn trước khi API chạy.
6.  Mỗi task có cách kiểm chứng.
7.  Người không owner vẫn phải hiểu flow.

## 9. Học AI theo thứ tự

``` text
Image / Bounding Box
→ YOLO inference
→ Confidence
→ Detection JSON
→ left/center/right
→ Decision Engine
→ FastAPI
→ TTS API
→ Flutter
→ Tracking/OCR/STT/LLM...
```
