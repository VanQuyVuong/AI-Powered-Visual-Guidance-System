# AI MODULES

**Project:** AI-Powered-Visual-Guidance-System\
**Version:** 0.1 --- MVP

## 1. Mục đích

Mô tả các module AI/API, input, output, công nghệ và giai đoạn triển
khai.

## 2. Tổng quan

  -------------------------------------------------------------------------------------
  Module           Công nghệ/API  Input                  Output          Giai đoạn
  ---------------- -------------- ---------------------- --------------- --------------
  Vision Detection YOLO           Image                  Objects,        MVP
                   pretrained                            confidence,     
                                                         bbox            

  Scene / Decision Nhóm tự phát   Detection              Guidance text   MVP
  Engine           triển                                                 

  Text-to-Speech   Blaze TTS      Text                   Audio           MVP

  Speech-to-Text   Blaze STT v1.0 Audio                  Text            Phase 2

  Voice Assistant  STT + LLM +    Speech                 Speech          Phase 2
                   TTS                                                   

  Realtime         Blaze          PCM audio              Partial/Final   Phase 2
  Translation      WebSocket v2.0                        text            

  News Audio       Blaze News     Request                Audio           Future
                   Audio API                             metadata/URL    

  Tracking         YOLO + tracker Video frames           Object tracks   Phase 2

  OCR              OCR engine/API Image                  Text            Phase 2

  Navigation       GPS +          Location/destination   Route data      Phase 3
                   Maps/Routes                                           
  -------------------------------------------------------------------------------------

## 3. Vision Detection

``` text
Image → YOLO → class + confidence + bounding box
```

YOLO chỉ phát hiện vật thể; không tự tạo câu hướng dẫn, gọi TTS hay
quyết định điều hướng.

## 4. Scene / Decision Engine

Đây là phần nhóm tự phát triển:

``` text
Detection → Position → Confidence filtering → Priority → Vietnamese guidance
```

Ví dụ:

``` text
person + center + confidence 0.91
→ "Phía trước có một người."
```

## 5. Text-to-Speech

API mẫu:

``` text
POST https://api.blaze.vn/v1/tts
```

Model mẫu: `v1.5_pro`.

Luồng:

``` text
Guidance text → POST /v1/tts → TTS job ID → GET /v1/tts/{id}/info → audio_url
```

TTS realtime cũng có:

``` text
wss://api.blaze.vn/v1/tts/realtime
```

Model mẫu: `2.0-realtime`.

Realtime để sau MVP.

## 6. Speech-to-Text

``` text
POST https://api.blaze.vn/v1/stt/execute?model=v1.0
```

Input: `audio_file` multipart/form-data.

Output có thể gồm `segments`, `raw_text`, `transcription`,
`is_successful`, `error_message`, `execute_time`.

## 7. Voice Assistant

Đây là pipeline:

``` text
Speech → STT → User Text → LLM → Assistant Text → TTS → Audio
```

Code mẫu dùng `/api/chat/stream`. Tài liệu hiện có chưa đủ để xác định
public LLM endpoint/model/authentication, nên không tự suy đoán.

## 8. Realtime Translation

``` text
wss://api.blaze.vn/v1/demo/stt/realtime
```

Model mẫu: `v2.0`.

``` text
PCM16 audio → WebSocket → ready / partial / final / error
```

Không thuộc MVP.

## 9. News Audio

``` text
GET https://api.blaze.vn/v1/news-audio/list?page=1&page_size=10
```

Cung cấp danh sách nội dung audio và URL. Đây là content/audio service,
không tự gọi là model AI.

## 10. Tracking

Sau MVP:

``` text
Frame → YOLO + Tracker → Object ID / movement
```

## 11. OCR

Sau MVP:

``` text
Image → OCR → Text
```

## 12. Navigation

Sau MVP:

``` text
GPS + Maps/Routes + Navigation Logic + Voice
```

Navigation không phải một model AI duy nhất.

## 13. MVP

``` text
Flutter → Image → Python Backend → YOLO → Decision Engine
→ Guidance Text → Blaze TTS → Audio → Flutter
```

## 14. Nguyên tắc

-   Không hard-code API key.
-   Không commit secret.
-   API doanh nghiệp đi qua adapter/service.
-   Không tự đoán endpoint chưa được tài liệu hóa.
-   Không để Flutter giữ secret nếu secret phải được bảo mật phía
    server.
