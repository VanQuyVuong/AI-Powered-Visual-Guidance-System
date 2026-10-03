# CODING RULES

## 1. General

-   Không tự ý đổi architecture.
-   Không tự ý đổi API contract.
-   Không thêm dependency lớn nếu chưa thống nhất.
-   Ưu tiên code đơn giản, dễ đọc.
-   Không tối ưu quá sớm.

## 2. Python

-   Dùng `.venv`.
-   Dependencies ghi vào `requirements.txt`.
-   Tách vision, guidance, API service.
-   Không hard-code secret.
-   Type hint cho function quan trọng.
-   Test logic độc lập với YOLO nếu có thể.

## 3. Flutter

-   Tách UI và API/service.
-   Không để API key bí mật trong app.
-   Xử lý loading/error rõ ràng.
-   Không xây UI lớn trước khi backend contract ổn định.

## 4. AI

-   Không train model từ đầu cho MVP.
-   Không tự đổi model nếu chưa thống nhất.
-   Luôn lưu confidence.
-   Không suy ra distance chính xác chỉ từ bounding box.
-   Không coi YOLO là decision engine.

## 5. Company API

-   Dùng adapter.
-   Đọc tài liệu chính thức trước khi tích hợp.
-   Không đoán request/response.
-   Không commit key.
-   Log lỗi nhưng không log secret.

## 6. Git

Branch:

``` text
feature/...
fix/...
test/...
docs/...
```

Commit:

``` text
feat(...)
fix(...)
test(...)
docs(...)
refactor(...)
```

## 7. AI Coding Agent

AI được phép:

-   tạo file theo task
-   sinh code
-   sinh test
-   sửa lỗi
-   cập nhật docs

AI không được tự ý:

-   đổi architecture
-   đổi framework
-   đổi model
-   đổi contract
-   xóa module
-   thêm dependency lớn
-   hard-code secret

Nếu cần thay đổi lớn:

``` text
PROPOSED CHANGE
Reason
Impact
Files affected
```

Team quyết định trước.
