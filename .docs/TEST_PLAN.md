# TEST PLAN

**Project:** AI-Powered-Visual-Guidance-System\
**Version:** 0.1 --- MVP

## 1. YOLO

Test ảnh có person, nhiều object, không có object và file không hợp lệ.

Expected:

``` text
class_name
confidence
bounding_box
```

hoặc lỗi phù hợp.

## 2. Position

Input:

``` text
image_width + bbox
```

Expected:

``` text
left / center / right
```

Phải test cả boundary.

## 3. Guidance

Ví dụ:

``` text
person + center → "Phía trước có một người."
car + left → guidance có "bên trái"
```

## 4. API

Test:

``` http
POST /api/v1/vision/analyze
```

Cases:

-   valid image
-   invalid image
-   unsupported format
-   empty detection
-   model error
-   guidance error

## 5. TTS

Sau khi có key/test environment:

-   text hợp lệ
-   tiếng Việt
-   authentication
-   completed status
-   audio URL
-   API error
-   timeout

Không commit API key.

## 6. Flutter

-   chọn ảnh
-   camera
-   preview
-   loading
-   success
-   error
-   guidance
-   audio playback

## 7. End-to-end

``` text
Image
→ Backend
→ YOLO
→ Decision
→ TTS
→ Audio
→ Flutter
```

Definition of Done:

``` text
[ ] Flow chạy được
[ ] Error case không crash
[ ] Không lộ API key
[ ] Hai thành viên đều chạy được project
```
