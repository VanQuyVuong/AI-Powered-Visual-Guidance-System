# BÁO CÁO SỬA LỖI #06: LỆCH TỌA ĐỘ BOUNDING BOX TRÊN MÀN HÌNH ANDROID

## 1. Hiện tượng (Issue)
- Người dùng gửi ảnh chụp màn hình thực tế: Chai trà Tea+ ở góc dưới bên trái có viền nhận diện nhưng bị lệch vị trí, mặt người bên trái bị ô viền đẩy lên đỉnh đầu.
- Model nhận diện đúng vật thể nhưng hiển thị trên màn hình bị dịch chuyển tọa độ (Bounding Box lệch khỏi đối tượng thật).

## 2. Nguyên nhân kỹ thuật (Root Cause)
1. **Aspect Ratio Mismatch:**
   - CameraX `ImageAnalysis` chụp và gửi ảnh phân giải 480x640 (tỷ lệ 3:4).
   - Màn hình Redmi Note 14 Pro 5G có độ phân giải 1220x2712 (tỷ lệ 9:20 rất dài).
   - CameraPreviewView hiển thị camera theo chế độ `FILL_CENTER` (giữ nguyên tỷ lệ 3:4, scale theo cạnh lớn nhất và crop phần thừa đều 2 bên).
   - Trong khi đó, `ARHudOverlay.kt` trước đây dùng phép nhân độc lập:
     `scaleX = screenW / 480` và `scaleY = screenH / 640`.
     -> Hai hệ số scaleX và scaleY khác nhau làm méo khung hình và làm trôi tọa độ của Bounding Box theo chiều dọc.
2. **Khắc phục:**
   - Cập nhật [`CameraPreviewView.kt`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/MBAPP/app/src/main/java/com/example/ai_hkdn/ui/camera/CameraPreviewView.kt): Khóa rõ ràng `scaleType = PreviewView.ScaleType.FILL_CENTER`.
   - Cập nhật [`ARHudOverlay.kt`](file:///d:/HK%20126/HKDN/AI-Powered-Visual-Guidance-System/MBAPP/app/src/main/java/com/example/ai_hkdn/ui/hud/ARHudOverlay.kt): Dùng thuật toán Uniform Aspect Ratio:
     ```kotlin
     val scale = maxOf(screenW / srcW, screenH / srcH)
     val offsetX = (screenW - srcW * scale) / 2f
     val offsetY = (screenH - srcH * scale) / 2f
     val left = box.x1 * scale + offsetX
     val top = box.y1 * scale + offsetY
     ...
     ```
   - Build và nạp APK trực tiếp vào điện thoại Redmi Note 14 Pro qua ADB.

## 3. Kết quả
- Toàn bộ khung viền AR HUD (chai nước, mặt người, laptop) giờ đây khớp tỷ lệ 1:1 với hình ảnh quang học từ camera trên màn hình điện thoại.
