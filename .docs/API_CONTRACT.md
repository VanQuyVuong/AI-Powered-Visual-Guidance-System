# API CONTRACT

**Project:** AI-Powered-Visual-Guidance-System  
**Document:** API Contract  
**Version:** 0.1  
**Status:** Draft — MVP  
**Last Updated:** 2026-10-02

---

## 1. Mục đích

Tài liệu này định nghĩa hợp đồng giao tiếp giữa các thành phần trong phiên bản MVP của hệ thống:

```text
Flutter Mobile App
        │
        │ HTTP
        ▼
Python Backend
        │
        ├── YOLO Vision
        ├── Scene / Decision Engine
        └── Company Voice API
```

Mục tiêu là thống nhất request, response, tên trường dữ liệu, kiểu dữ liệu, lỗi và ranh giới trách nhiệm giữa Flutter, Backend và AI.

> **Lưu ý:** API của doanh nghiệp cung cấp Voice/STT/TTS/Conversation/Translation chưa được đặc tả trong tài liệu này. Không được tự đoán endpoint, header, authentication, request/response hoặc audio format. Các phần đó sẽ được cập nhật sau khi nhóm nhận tài liệu API chính thức từ doanh nghiệp.

---

# 2. Phạm vi MVP

Luồng MVP:

```text
Ảnh
  ↓
Python Backend
  ↓
YOLO
  ↓
Detection JSON
  ↓
Decision Engine
  ↓
Vietnamese Guidance Text
  ↓
Company Voice API
  ↓
Audio
  ↓
Flutter phát âm thanh
```

### Có trong MVP

- Chọn/chụp ảnh trên Flutter.
- Gửi ảnh tới Backend.
- Backend chạy YOLO.
- Trả về vật thể được phát hiện.
- Xác định vị trí tương đối: `left`, `center`, `right`.
- Tạo câu hướng dẫn tiếng Việt.
- Gọi Voice API của doanh nghiệp.
- Flutter nhận và phát audio.

### Chưa thuộc MVP

- Tracking realtime.
- OCR.
- GPS.
- Google Maps/Routes.
- Nhận diện vạch dẫn đường cho người khiếm thị.
- Depth estimation.
- VLM.
- Guardian/SOS.
- Smart Hat thật.
- Custom model training.

---

# 3. API Endpoint

## 3.1. Analyze Image

```http
POST /api/v1/vision/analyze
```

### Content-Type

```http
multipart/form-data
```

### Request

| Field | Type | Required | Mô tả |
|---|---|---:|---|
| `image` | File | Yes | Ảnh cần phân tích |

Ví dụ:

```text
POST /api/v1/vision/analyze
Content-Type: multipart/form-data
image = test.jpg
```

---

# 4. Response thành công

HTTP Status:

```http
200 OK
```

Ví dụ:

```json
{
  "success": true,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.91,
      "position": "center",
      "bounding_box": {
        "x1": 120,
        "y1": 80,
        "x2": 350,
        "y2": 500
      }
    }
  ],
  "guidance": {
    "text": "Phía trước có một người."
  }
}
```

---

# 5. Detection Schema

Mỗi object được YOLO phát hiện:

```json
{
  "class_name": "person",
  "confidence": 0.91,
  "position": "center",
  "bounding_box": {
    "x1": 120,
    "y1": 80,
    "x2": 350,
    "y2": 500
  }
}
```

### `class_name`

Tên lớp vật thể, ví dụ `person`, `car`, `motorcycle`, `bicycle`, `chair`, `dog`. Tên lớp thực tế phụ thuộc model/dataset.

### `confidence`

Độ tin cậy của model, trong khoảng `0.0 → 1.0`.

### `position`

Vị trí tương đối trong ảnh:

```text
left
center
right
```

Decision Engine xác định giá trị này dựa trên bounding box và kích thước ảnh.

### `bounding_box`

```json
{
  "x1": 120,
  "y1": 80,
  "x2": 350,
  "y2": 500
}
```

`x1, y1` là góc trên bên trái; `x2, y2` là góc dưới bên phải. Đơn vị là pixel theo ảnh đầu vào.

---

# 6. Guidance Schema

```json
{
  "guidance": {
    "text": "Phía trước có một người."
  }
}
```

Decision Engine chuyển:

```text
Detection Data
      ↓
Semantic Interpretation
      ↓
Vietnamese Guidance
```

YOLO **không** chịu trách nhiệm tạo câu hướng dẫn hoặc gọi Voice API.

---

# 7. Response đầy đủ

```json
{
  "success": true,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.94,
      "position": "center",
      "bounding_box": {
        "x1": 410,
        "y1": 100,
        "x2": 620,
        "y2": 600
      }
    },
    {
      "class_name": "car",
      "confidence": 0.88,
      "position": "left",
      "bounding_box": {
        "x1": 20,
        "y1": 220,
        "x2": 300,
        "y2": 650
      }
    }
  ],
  "guidance": {
    "text": "Phía trước có một người và bên trái có một chiếc xe."
  }
}
```

---

# 8. Error Response

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Ảnh không hợp lệ hoặc không thể đọc."
  }
}
```

Các mã lỗi dự kiến:

| Code | Ý nghĩa |
|---|---|
| `INVALID_IMAGE` | File ảnh không hợp lệ |
| `IMAGE_TOO_LARGE` | Ảnh vượt kích thước cho phép |
| `UNSUPPORTED_FORMAT` | Định dạng ảnh không hỗ trợ |
| `VISION_MODEL_ERROR` | Lỗi khi chạy model |
| `GUIDANCE_PROCESSING_ERROR` | Lỗi Decision Engine |
| `VOICE_API_ERROR` | Lỗi gọi Voice API |
| `INTERNAL_SERVER_ERROR` | Lỗi máy chủ |

---

# 9. Trách nhiệm của từng thành phần

## Flutter

- Chọn/chụp ảnh.
- Gửi request tới Backend.
- Hiển thị kết quả.
- Nhận audio.
- Phát audio.
- Xử lý loading/error/success.

Flutter không chạy YOLO trong kiến trúc MVP hiện tại.

## Python Backend

- Nhận request.
- Kiểm tra input.
- Gọi Vision module.
- Gọi Decision Engine.
- Tạo guidance text.
- Gọi API doanh nghiệp khi cần.
- Trả response cho Flutter.

## YOLO Vision

```text
Image → Object Detection → class + confidence + bounding box
```

YOLO không chịu trách nhiệm tạo câu nói, gọi Voice API hoặc điều hướng người dùng.

## Decision Engine

```text
Detection → Position → Priority → Guidance
```

Ví dụ:

```text
YOLO:
person
confidence = 0.91
bbox = ...

Decision Engine:
position = center

Guidance:
"Phía trước có một người."
```

---

# 10. Voice API của doanh nghiệp

Hiện tại nhóm **chưa được tự định nghĩa API contract của doanh nghiệp**.

Cần chờ:

```text
Base URL
Endpoint
HTTP Method
Authentication
API Key/Header
Request Body
Response Body
Audio Format
Content-Type
Timeout
Rate Limit
Error Format
```

Sau khi có tài liệu chính thức, cập nhật phần này.

Kiến trúc nội bộ nên dùng adapter/interface:

```text
Decision Engine
      ↓
Voice Service Interface
      ↓
Company Voice API Adapter
      ↓
Company API
```

---

# 11. Luồng MVP hoàn chỉnh

```text
┌──────────────┐
│    Flutter   │
└──────┬───────┘
       │ POST image
       ▼
┌──────────────┐
│   Backend    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     YOLO     │
└──────┬───────┘
       │ Detection JSON
       ▼
┌────────────────────┐
│  Decision Engine   │
└─────────┬──────────┘
          │ Guidance Text
          ▼
┌────────────────────┐
│ Company Voice API  │
└─────────┬──────────┘
          │ Audio
          ▼
┌──────────────┐
│    Flutter   │
│ Play Audio   │
└──────────────┘
```

---

# 12. Ví dụ End-to-End

### Bước 1 — Flutter gửi ảnh

```text
test.jpg
```

### Bước 2 — YOLO phát hiện

```json
{
  "class_name": "person",
  "confidence": 0.91,
  "position": "center"
}
```

### Bước 3 — Decision Engine

```text
Phía trước có một người.
```

### Bước 4 — Voice Service

```text
Guidance Text
      ↓
Company Voice API
      ↓
Audio
```

### Bước 5 — Flutter

```text
Nhận audio
   ↓
Phát audio
```

---

# 13. API Versioning

API sử dụng version trong URL:

```text
/api/v1/
```

Ví dụ:

```text
/api/v1/vision/analyze
```

Nếu có thay đổi lớn không tương thích:

```text
/api/v2/
```

Không tự ý phá vỡ contract của version hiện tại.

---

# 14. Quy tắc dữ liệu

- Tên field sử dụng `snake_case`.
- JSON response sử dụng UTF-8.
- Confidence sử dụng số thực.
- Bounding box sử dụng số.
- Không đưa API key vào request của client nếu API key phải được bảo mật.
- Secret/API key không được commit vào Git.
- Dữ liệu nhạy cảm không được log tùy tiện.

---

# 15. Testing Contract

### Success

```text
Ảnh hợp lệ
→ 200
→ Có detections
→ Có guidance
```

### Empty detection

```text
Ảnh hợp lệ
→ Không phát hiện object
→ guidance phù hợp
```

### Invalid image

```text
File không phải ảnh
→ INVALID_IMAGE
```

### Unsupported format

```text
Định dạng không hỗ trợ
→ UNSUPPORTED_FORMAT
```

### Model error

```text
YOLO lỗi
→ VISION_MODEL_ERROR
```

### Voice API error

```text
Voice API lỗi
→ VOICE_API_ERROR
```

---

# 16. Definition of Done

```text
[ ] Endpoint được xác định
[ ] Request được xác định
[ ] Detection schema được xác định
[ ] Guidance schema được xác định
[ ] Error schema được xác định
[ ] Trách nhiệm Flutter được xác định
[ ] Trách nhiệm Backend được xác định
[ ] Trách nhiệm YOLO được xác định
[ ] Trách nhiệm Decision Engine được xác định
[ ] Company Voice API chưa rõ được đánh dấu TODO
[ ] Không tự đoán API của doanh nghiệp
[ ] Có ví dụ request/response
[ ] Có test cases cơ bản
```

---

# 17. Trạng thái tài liệu

```text
Status: Draft
```

Các phần có thể thay đổi sau khi:

1. Nhận API documentation từ doanh nghiệp.
2. Có API key/test environment.
3. Hoàn thành G1 YOLO.
4. Hoàn thành G2 Decision Engine.
5. Tích hợp Flutter.
