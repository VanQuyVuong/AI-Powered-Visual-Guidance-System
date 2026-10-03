# AI-Powered-Visual-Guidance-System

## 1. Mục đích tài liệu

Đây là tài liệu **Source of Truth** của team trước khi code. Tài liệu
dùng để:

-   Team hiểu cùng một hệ thống.
-   AI Coding Agent đọc để sinh cấu trúc, code và test.
-   Chia module và nhiệm vụ cho thành viên.
-   Theo dõi MVP, tránh làm lan man.
-   Làm cơ sở cập nhật khi API/model thực tế từ doanh nghiệp được cung
    cấp.

> Nguyên tắc: **Phân tích → đặc tả → contract → giao task → AI sinh code
> → test → review → tích hợp → cập nhật tài liệu.**

------------------------------------------------------------------------

## 2. Bối cảnh đề tài

Đề tài hướng tới hệ thống hỗ trợ người khiếm thị bằng AI, kết hợp
camera, AI/CV, định vị, bản đồ, giọng nói và về lâu dài có thể kết nối
thiết bị đeo/người giám hộ.

Theo tài liệu đề tài, hệ thống hướng tới:

-   Nhận biết môi trường.
-   Phát hiện vật cản/phương tiện.
-   Hỗ trợ định hướng.
-   Nhận biết đường dẫn hỗ trợ người khiếm thị.
-   Đọc chữ.
-   Cảnh báo bất thường/SOS.
-   Kết nối người thân hoặc người giám hộ.
-   Về lâu dài có thể tích hợp cảm biến, IMU, GPS, thiết bị đeo và các
    chức năng an toàn.

**Prototype hiện tại của môn học không cần xây hoàn chỉnh chiếc mũ.**
Trọng tâm hiện tại là **Flutter mobile app mô phỏng hệ thống**, trong đó
doanh nghiệp cung cấp các API/model liên quan đến voice/communication.

------------------------------------------------------------------------

# 3. Mục tiêu kỹ thuật hiện tại

Vertical Slice đầu tiên cần chứng minh được:

``` text
Image / Camera
      ↓
AI Vision
      ↓
Structured Detection
      ↓
Scene Processing / Decision
      ↓
Vietnamese Guidance Text
      ↓
Company TTS API
      ↓
Audio
      ↓
Flutter phát âm thanh
```

Đây là xương sống cần hoàn thành trước.

Không làm tất cả chức năng cùng lúc.

------------------------------------------------------------------------

# 4. Bốn nhóm chức năng lớn

## 4.1. Image / Environment Recognition

Bao gồm:

-   Object Detection.
-   Vị trí tương đối LEFT/CENTER/RIGHT.
-   Priority của vật thể.
-   Tracking khi chuyển sang video.
-   OCR đọc chữ.
-   Nghiên cứu nhận biết đường dẫn hỗ trợ người khiếm thị.
-   Về sau có thể nghiên cứu depth/distance và VLM.

## 4.2. Communication

Doanh nghiệp cung cấp API/model, có thể gồm:

-   Speech-to-Text.
-   Text-to-Speech.
-   Conversation.
-   Translation.
-   Live audio/news.

**Không tự xây lại các API này nếu doanh nghiệp đã cung cấp.**

Chưa có tài liệu API chính thức thì **không đoán
endpoint/request/response**.

## 4.3. Image → Text → Voice

Đây là chức năng cốt lõi của prototype:

``` text
Image
→ YOLO
→ Detection JSON
→ Decision Engine
→ Vietnamese sentence
→ Company TTS
→ Audio
```

YOLO không tự quyết định câu nói và không tự làm TTS.

## 4.4. Navigation

Tách thành:

``` text
GPS
→ vị trí

Maps / Routes
→ dữ liệu tuyến

Navigation Processor
→ quyết định bước hướng dẫn

Voice
→ đọc hướng dẫn
```

Google Maps/Routes không phải toàn bộ AI navigation. Cần một lớp logic
riêng cho cách diễn đạt và trạng thái điều hướng.

------------------------------------------------------------------------

# 5. Kiến trúc tổng thể

``` text
┌─────────────────────────────┐
│         Flutter App         │
│ Camera / Image / Voice / Map│
│ Audio Player                │
└──────────────┬──────────────┘
               │ HTTP
               ▼
┌─────────────────────────────┐
│      Python AI Backend      │
│           FastAPI           │
│                             │
│ Vision / YOLO               │
│ Tracking                    │
│ OCR                         │
│ Scene Processing            │
│ Decision Engine             │
│ Guidance Generator          │
│ Navigation                  │
└──────────────┬──────────────┘
               │
       ┌───────▼────────┐
       │ Company APIs   │
       │ STT / TTS      │
       │ Conversation   │
       │ Translation    │
       └────────────────┘
```

------------------------------------------------------------------------

# 6. Model và công nghệ cần nghiên cứu

  -----------------------------------------------------------------------
  Thành phần        Vai trò           Giai đoạn         Train riêng?
  ----------------- ----------------- ----------------- -----------------
  YOLO pretrained   Object Detection  G1                Không

  YOLO + Tracker    Theo dõi object   G3                Không ở đầu

  OCR               Đọc chữ           G4                Không ở đầu

  Depth/Distance    Khoảng cách       Sau MVP           Chưa quyết định

  Custom Vision     Đường dẫn hỗ      Sau MVP           Có thể cần
                    trợ/object đặc                      
                    thù                                 

  VLM               Hiểu scene phức   Future            Không ưu tiên
                    tạp                                 

  Company STT       Speech → Text     Khi API sẵn sàng  Không

  Company TTS       Text → Speech     MVP voice         Không

  Company           Hội thoại         Sau voice cơ bản  Không
  Conversation                                          

  Translation       Dịch              Khi cần           Không

  GPS + Maps/Routes Navigation        G6                Không

  Decision Engine   Biến detection    G2                **Tự xây logic**
                    thành guidance                      
  -----------------------------------------------------------------------

### Quy tắc model

Không train model ngay.

Trình tự:

``` text
Pretrained Model
→ Test baseline
→ Xác định thiếu gì
→ Thu thập dataset
→ Annotation
→ Train/Fine-tune nếu thực sự cần
→ Evaluation
```

------------------------------------------------------------------------

# 7. G1 --- Object Detection

## Mục tiêu

``` text
Image
→ YOLO
→ JSON
```

Ví dụ:

``` json
{
  "objects": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100, 100, 300, 500]
    }
  ]
}
```

## Việc cần làm

-   Load pretrained YOLO.
-   Nhận ảnh.
-   Inference.
-   Lấy class.
-   Lấy confidence.
-   Lấy bounding box.
-   Chuẩn hóa response.
-   Tạo API endpoint.

## Test

-   Ảnh có người.
-   Ảnh có xe.
-   Nhiều object.
-   Không có object.
-   Ảnh độ phân giải khác nhau.
-   File ảnh lỗi.
-   Input không hợp lệ.

## Definition of Done

-   Model load được.
-   Endpoint chạy.
-   JSON đúng schema.
-   Có test.
-   Error handling cơ bản.
-   Module có thể được gọi độc lập.

------------------------------------------------------------------------

# 8. G2 --- Scene Processing / Decision Engine

YOLO chỉ cho dữ liệu kỹ thuật. Cần biến thành thông tin hữu ích.

``` text
Detection
 ↓
Position
 ↓
Confidence Filter
 ↓
Priority
 ↓
Guidance
```

Ví dụ:

``` json
{
  "class_name": "motorcycle",
  "confidence": 0.91,
  "bbox": [450, 120, 620, 400]
}
```

→

``` json
{
  "class_name": "motorcycle",
  "position": "RIGHT",
  "priority": "HIGH",
  "guidance": "Phía bên phải có xe máy."
}
```

### Quy tắc quan trọng

Không phải object nào cũng cần đọc.

Cần nghiên cứu:

-   Object nào quan trọng?
-   Confidence tối thiểu?
-   CENTER có ưu tiên cao hơn LEFT/RIGHT không?
-   Có tránh đọc lặp lại không?
-   Cảnh báo nào cần ưu tiên?

### Test

Decision Engine phải có thể test bằng **JSON giả**, không phụ thuộc
YOLO.

------------------------------------------------------------------------

# 9. G3 --- Tracking

Khi chuyển sang video:

``` text
Frame 1 → YOLO → Tracker → ID 1
Frame 2 → YOLO → Tracker → ID 1
Frame 3 → YOLO → Tracker → ID 1
```

Mục tiêu:

-   Giữ object ID.
-   Theo dõi qua frame.
-   Lưu lịch sử vị trí.
-   Phân tích chuyển động.

Không kết luận "đang tiến lại gần" chỉ từ một ảnh. Muốn đánh giá khoảng
cách/chuyển động cần dữ liệu nhiều frame và phương pháp phù hợp.

------------------------------------------------------------------------

# 10. G4 --- OCR

Pipeline:

``` text
Image
→ OCR
→ Text
→ Text Processing
→ Guidance
→ TTS
```

Test:

-   Chữ rõ.
-   Chữ nhỏ.
-   Nhiều dòng.
-   Tiếng Việt.
-   Ngoài trời.
-   Chữ nghiêng.
-   Chữ bị che.
-   Không có chữ.

------------------------------------------------------------------------

# 11. Nhận biết đường dẫn hỗ trợ người khiếm thị

Đây là bài toán cần nghiên cứu riêng.

Không giả định YOLO pretrained đã giải quyết được.

Cần xác định:

-   Dataset.
-   Class.
-   Detection hay segmentation.
-   Điều kiện ánh sáng.
-   Góc camera.
-   Che khuất.
-   Đường dẫn bị đứt.
-   Bối cảnh vỉa hè/đường phố.

Giai đoạn đầu:

``` text
Research → Dataset → Baseline → Evaluation
```

Sau đó mới quyết định custom training.

------------------------------------------------------------------------

# 12. G5 --- Company API Integration

Chỉ triển khai theo **tài liệu chính thức của doanh nghiệp**.

Cần xác định:

-   Base URL.
-   Endpoint.
-   HTTP method.
-   Authentication/API key.
-   Header.
-   Request.
-   Response.
-   Error.
-   Timeout.
-   Rate limit.
-   Audio format.
-   File format.

Nên thiết kế adapter:

``` text
Application
   ↓
TTSService / STTService / ConversationService
   ↓
Company API Adapter
   ↓
Company API
```

Không để API key trong source code hoặc Git.

Không để Flutter phải biết chi tiết endpoint nếu backend có thể làm
adapter.

------------------------------------------------------------------------

# 13. G6 --- Navigation

Các thành phần:

``` text
GPS
+
Maps / Routes
+
Route Processor
+
Navigation State
+
Voice
```

Trạng thái đề xuất:

``` text
IDLE
ROUTING
NAVIGATING
OFF_ROUTE
ARRIVED
ERROR
```

Test:

-   GPS hợp lệ.
-   GPS sai số.
-   Mất GPS.
-   Route hợp lệ.
-   Không có route.
-   Lệch route.
-   Đến đích.
-   Mất mạng.

------------------------------------------------------------------------

# 14. G7 --- Flutter Integration

Flutter chỉ cần giao tiếp với contract nội bộ.

Ví dụ:

``` http
POST /api/v1/vision/analyze
```

Backend trả:

``` json
{
  "objects": [],
  "guidance": []
}
```

Flutter chịu trách nhiệm:

-   Camera/gallery.
-   API client.
-   Hiển thị trạng thái.
-   Audio player.
-   Map.
-   User interaction.

Backend chịu trách nhiệm:

-   AI.
-   Processing.
-   Decision.
-   API orchestration.

------------------------------------------------------------------------

# 15. Cấu trúc project đề xuất

``` text
AI-Powered-Visual-Guidance-System/
│
├── docs/
│   ├── SYSTEM_ANALYSIS.md
│   ├── API_CONTRACT.md
│   ├── AI_MODULES.md
│   ├── TEST_PLAN.md
│   ├── TEAM_TASKS.md
│   └── CODING_RULES.md
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── vision/
│   │   ├── tracking/
│   │   ├── ocr/
│   │   ├── guidance/
│   │   └── navigation/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── mobile/
│   └── flutter_app/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
└── README.md
```

> Không tạo tất cả folder chỉ để "trông chuyên nghiệp". Module chỉ được
> tạo khi roadmap tới hoặc có lý do kỹ thuật.

------------------------------------------------------------------------

# 16. API contract nội bộ

Ví dụ:

``` http
POST /api/v1/vision/analyze
```

Input:

``` json
{
  "image": "..."
}
```

Output:

``` json
{
  "request_id": "...",
  "objects": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100,
        "y1": 100,
        "x2": 300,
        "y2": 500
      },
      "position": "CENTER"
    }
  ],
  "guidance": [
    {
      "text": "Phía trước có một người.",
      "priority": "MEDIUM"
    }
  ]
}
```

Schema chính thức phải được chốt trong `API_CONTRACT.md` trước khi AI
sinh code lớn.

------------------------------------------------------------------------

# 17. Testing Strategy

## 17.1. Unit Test

Test logic độc lập:

``` text
bbox → position
confidence → filter
objects → priority
objects → guidance
```

Ví dụ:

``` text
Input: bbox bên trái
Expected: LEFT
```

## 17.2. Integration Test

``` text
Flutter
→ Backend
→ YOLO
→ Scene Processor
→ Response
```

## 17.3. API Test

Kiểm tra:

-   200. 
-   400. 
-   401. 
-   404. 
-   422. 
-   500. 
-   Timeout.
-   Invalid image.
-   Empty request.

## 17.4. Model Test

Theo dõi:

-   False Positive.
-   False Negative.
-   Precision.
-   Recall.
-   mAP khi có dataset phù hợp.
-   Latency.
-   FPS khi xử lý video.

## 17.5. Scenario Test

Ví dụ:

``` text
Scenario:
Một người ở bên trái.

Expected:
"Phía bên trái có một người."
```

``` text
Scenario:
Không có object quan trọng.

Expected:
Không phát cảnh báo không cần thiết.
```

------------------------------------------------------------------------

# 18. Test Dataset

Không chỉ demo bằng ảnh ngẫu nhiên.

Nên xây:

``` text
test_dataset/
├── people/
├── vehicles/
├── obstacles/
├── signs/
├── text/
├── low_light/
├── crowded/
└── edge_cases/
```

Mỗi case nên có:

``` text
ID
Description
Condition
Expected
Actual
Pass/Fail
Notes
```

Nếu custom training:

``` text
Raw
→ Clean
→ Annotate
→ Train/Validation/Test
→ Training
→ Evaluation
→ Model Version
```

------------------------------------------------------------------------

# 19. Vertical Slice --- mốc demo đầu tiên

Đây là mốc quan trọng nhất trước khi mở rộng:

``` text
Flutter
 ↓
Chọn ảnh
 ↓
POST /vision/analyze
 ↓
YOLO
 ↓
Scene Processor
 ↓
"Phía trước có một người."
 ↓
Company TTS
 ↓
Audio
 ↓
Flutter phát tiếng
```

Nếu chạy được luồng này, kiến trúc cơ bản đã được chứng minh.

Sau đó mới mở rộng:

``` text
Image
→ Video
→ Tracking
→ OCR
→ STT
→ Conversation
→ Navigation
→ Guardian
```

------------------------------------------------------------------------

# 20. Roadmap tổng thể

``` text
G0  System Analysis
 ↓
G1  YOLO Detection
 ↓
G2  Scene Processing
 ↓
G3  Tracking
 ↓
G4  OCR
 ↓
G5  Company APIs
 ↓
G6  Navigation
 ↓
G7  Flutter Integration
 ↓
Future: Custom Vision / Sensors / Wearable
```

Đối với prototype môn học, ưu tiên:

``` text
G0 → G1 → G2 → G5(TTS) → G7
```

Sau đó mới mở rộng.

------------------------------------------------------------------------

# 21. Phân chia team

## AI/CV

-   YOLO.
-   Tracking.
-   OCR.
-   Dataset.
-   Model evaluation.

## AI Processing

-   Position.
-   Priority.
-   Scene processing.
-   Guidance.
-   Rule engine.

## Backend/API

-   FastAPI.
-   API contract.
-   Company adapters.
-   Error handling.
-   Security.

## Flutter

-   Camera.
-   Gallery.
-   API client.
-   Audio.
-   Map.
-   UI.

## QA/Product

-   Requirement.
-   Scenario.
-   Test dataset.
-   API test.
-   Integration test.
-   Demo.
-   Documentation.

Một thành viên có thể đảm nhiệm nhiều vai trò.

------------------------------------------------------------------------

# 22. AI Coding Workflow

Team không nên đưa prompt:

> "Xây toàn bộ hệ thống AI hỗ trợ người khiếm thị."

Thay vào đó:

``` text
SYSTEM_ANALYSIS.md
        ↓
MODULE_SPEC
        ↓
API_CONTRACT
        ↓
TASK
        ↓
AI Coding Agent
        ↓
Code
        ↓
Test
        ↓
Review
        ↓
Merge
```

Mỗi task cho AI phải có:

``` text
Context
Goal
Input
Output
Constraints
Existing files
Expected behavior
Test cases
Definition of Done
```

Ví dụ:

``` text
Task:
Implement bbox position classifier.

Input:
image width + bbox.

Output:
LEFT / CENTER / RIGHT.

Constraints:
- Pure Python.
- No YOLO dependency.
- No UI.
- Unit-testable.

Tests:
- left
- center
- right
- boundary cases
```

------------------------------------------------------------------------

# 23. Quy tắc AI Coding Agent

AI được phép:

-   Tạo file/folder theo task.
-   Sinh code.
-   Sinh test.
-   Chạy test.
-   Sửa lỗi.
-   Viết documentation.

AI không được tự ý:

-   Đổi architecture.
-   Đổi framework.
-   Đổi API contract.
-   Đổi model.
-   Xóa module.
-   Đổi response schema.
-   Thêm dependency lớn.
-   Hard-code secret.

Nếu muốn thay đổi lớn, AI phải báo:

``` text
PROPOSED CHANGE
Reason
Impact
Files affected
Migration required
```

Team quyết định trước.

------------------------------------------------------------------------

# 24. Git workflow

Có thể chia branch:

``` text
main
develop

feature/vision-yolo
feature/scene-processing
feature/tracking
feature/ocr
feature/company-tts
feature/flutter-camera
feature/navigation
```

Ví dụ commit:

``` text
feat(vision): add YOLO image detection endpoint
feat(guidance): add object position classifier
test(guidance): add bbox position tests
feat(tts): integrate company voice adapter
```

------------------------------------------------------------------------

# 25. Definition of Done

Một module chỉ hoàn thành khi:

-   [ ] Requirement rõ.
-   [ ] Input rõ.
-   [ ] Output rõ.
-   [ ] Schema/API rõ nếu có.
-   [ ] Code đã được sinh/chỉnh theo task cụ thể.
-   [ ] Test phù hợp đã có.
-   [ ] Test pass.
-   [ ] Error case được xem xét.
-   [ ] README/documentation cập nhật.
-   [ ] Không hard-code secret.
-   [ ] Không phá module khác.
-   [ ] Thành viên khác có thể chạy module.

------------------------------------------------------------------------

# 26. MVP vs Future Scope

## MVP

-   Flutter.
-   Image/camera.
-   YOLO detection.
-   Position.
-   Basic priority.
-   Guidance text.
-   Company TTS.
-   Audio playback.
-   Backend API.
-   Basic test.

## Phase 2

-   Tracking.
-   OCR.
-   STT.
-   Conversation.
-   Translation.
-   Better scene processing.

## Phase 3

-   GPS.
-   Maps/Routes.
-   Navigation state.
-   Guardian/SOS.

## Phase 4

-   Custom tactile-path model.
-   Depth/distance.
-   Sensor fusion.
-   Edge AI.
-   Wearable hardware.

## Future Research

-   Fall detection.
-   Airbag/protective mechanism.
-   Advanced VLM.
-   Offline mode.
-   On-device optimization.

------------------------------------------------------------------------

# 27. Những thứ không làm quá sớm

1.  Không train model từ đầu.
2.  Không xây smart hat hoàn chỉnh ngay.
3.  Không tích hợp tất cả API cùng lúc.
4.  Không làm UI quá lớn trước khi contract ổn định.
5.  Không phụ thuộc VLM ngay từ đầu.
6.  Không coi GPS là navigation hoàn chỉnh.
7.  Không suy ra khoảng cách chỉ từ bounding box.
8.  Không tự xây TTS nếu doanh nghiệp đã cung cấp.
9.  Không commit API key.
10. Không để AI tự ý thay đổi architecture.

------------------------------------------------------------------------

# 28. Security / Privacy

Hệ thống có thể xử lý:

-   Hình ảnh.
-   Giọng nói.
-   Vị trí.
-   Thông tin người dùng.

Do đó:

-   API key không được commit Git.
-   Secret dùng `.env`/secret management.
-   Không đưa dữ liệu thật nhạy cảm vào repository.
-   Hạn chế log dữ liệu nhạy cảm.
-   Xác định dữ liệu nào được lưu.
-   Dùng dữ liệu test phù hợp khi demo.
-   Prototype không được tuyên bố đảm bảo an toàn tuyệt đối.

------------------------------------------------------------------------

# 29. Nguyên tắc an toàn

Hệ thống là **công cụ hỗ trợ**, không thay thế hoàn toàn:

-   Gậy trắng.
-   Người hỗ trợ.
-   Các phương tiện hỗ trợ truyền thống.

Đặc biệt cần thận trọng với:

-   Phát hiện ngã.
-   Túi khí.
-   Cảnh báo va chạm.
-   Hướng dẫn "đi sang trái X bước".
-   Các quyết định có thể ảnh hưởng trực tiếp đến an toàn.

Các chức năng này chỉ nên được nâng cấp khi có phương pháp kiểm thử phù
hợp.

------------------------------------------------------------------------

# 30. Các điểm chưa được tự ý quyết định

Chờ tài liệu/API chính thức của doanh nghiệp:

-   Base URL.
-   Endpoint.
-   API key/header.
-   Request/response.
-   Audio format.
-   Rate limit.
-   Authentication.
-   Conversation context.
-   Translation interface.
-   Chính sách dữ liệu.

Các quyết định kỹ thuật cần nhóm tiếp tục chốt:

-   Có database hay không.
-   Navigation dùng Maps SDK/Routes ở mức nào.
-   Guardian có nằm trong prototype môn học không.
-   Custom tactile-path model dùng detection hay segmentation.
-   Có cần depth camera không.

**Không được đoán API contract.**

------------------------------------------------------------------------

# 31. Việc cần làm ngay

## Task 01 --- Chốt System Analysis

-   Team đọc file này.
-   Chốt MVP.
-   Đánh dấu phần chưa rõ.

## Task 02 --- Tạo API_CONTRACT.md

Chốt:

``` text
POST /api/v1/vision/analyze
```

và schema input/output.

## Task 03 --- G1 YOLO

Mục tiêu duy nhất:

``` text
image → YOLO → JSON
```

## Task 04 --- Test G1

Tạo test image + unit/API tests.

## Task 05 --- G2 Decision Engine

``` text
Detection JSON
→ position
→ priority
→ guidance
```

## Task 06 --- Vertical Slice

``` text
Flutter
→ Backend
→ YOLO
→ Guidance
→ Company TTS
→ Audio
```

**Đây là mốc demo đầu tiên.**

------------------------------------------------------------------------

# 32. Bộ tài liệu nên có trong repo

``` text
docs/
├── SYSTEM_ANALYSIS.md
├── API_CONTRACT.md
├── AI_MODULES.md
├── TEST_PLAN.md
├── TEAM_TASKS.md
└── CODING_RULES.md
```

AI Coding Agent phải đọc các tài liệu liên quan trước khi sinh code.

------------------------------------------------------------------------

# 33. Trạng thái hiện tại

  Module                 Trạng thái
  ---------------------- --------------------------
  System Analysis        🟡 Đang chốt
  YOLO Detection         🟡 Chuẩn bị
  Scene Processing       ⚪ Chưa code
  Tracking               ⚪ Chưa code
  OCR                    ⚪ Chưa code
  Company TTS            🟡 Chờ API/documentation
  Company STT            🟡 Chờ API/documentation
  Company Conversation   🟡 Chờ API/documentation
  Translation            ⚪ Chưa triển khai
  Navigation             ⚪ Chưa triển khai
  Guardian/SOS           ⚪ Chưa triển khai
  Custom Tactile Path    ⚪ Research
  Hardware/Wearable      ⚪ Future

------------------------------------------------------------------------

# 34. Kết luận

Hệ thống không nên được xây theo tư duy:

``` text
"Cần một model AI thật thông minh."
```

Mà nên xây theo:

``` text
"Cần một pipeline ổn định biến dữ liệu môi trường
thành thông tin hữu ích cho người dùng."
```

Xương sống:

``` text
INPUT
 ↓
VISION
 ↓
STRUCTURED DATA
 ↓
SCENE PROCESSING
 ↓
DECISION
 ↓
GUIDANCE TEXT
 ↓
VOICE API
 ↓
AUDIO
 ↓
USER
```

Các thành phần khác được thêm từng bước.

## Quy tắc vàng

> **Phân tích trước → chia module → định nghĩa contract → giao task cho
> AI → sinh code → test → review → tích hợp → cập nhật tài liệu.**

> **Không xây tất cả cùng lúc. Không train model khi chưa cần. Không
> đoán API. Không để AI tự ý thay đổi kiến trúc.**
