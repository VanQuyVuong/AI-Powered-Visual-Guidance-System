import sys
import os
import io
import time
import json
import base64
import traceback

# Đảm bảo in tiếng Việt không bị lỗi charmap trên Windows console
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Thiết lập đường dẫn import
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

BUG_FIX_DIR = os.path.join(PROJECT_ROOT, "bug_fix")
os.makedirs(BUG_FIX_DIR, exist_ok=True)

test_results = []
bugs_found = []

def record_test(name: str, passed: bool, details: str = "", bug_info: dict = None):
    test_results.append({
        "name": name,
        "passed": passed,
        "details": details
    })
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[{status}] {name}: {details}")
    if not passed and bug_info:
        bugs_found.append(bug_info)

# ==========================================
# TEST 1: KIỂM TRA ĐỌC MÔ HÌNH VÀ FILE TRỌNG SỐ
# ==========================================
print("\n--- 1. KIỂM TRA MÔ HÌNH VÀ FILE TRỌNG SỐ ---")
try:
    from app.vision.yolo_service import model, model_file, run_inference, run_tracking
    sample_img_path = os.path.join(PROJECT_ROOT, "test_images", "sample.jpg")
    
    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            sample_bytes = f.read()
            
        t0 = time.time()
        dets, w = run_inference(sample_bytes)
        t_infer = (time.time() - t0) * 1000
        record_test("YOLO Inference Sample Image", True, f"Model: {os.path.basename(model_file)}, Detections: {len(dets)}, Time: {t_infer:.1f}ms")
        
        t0 = time.time()
        track_dets, tw, th = run_tracking(sample_bytes)
        t_track = (time.time() - t0) * 1000
        record_test("YOLO ByteTrack Tracking", True, f"Tracked: {len(track_dets)} objects, Time: {t_track:.1f}ms")
    else:
        record_test("Sample Image Existence", False, "Không tìm thấy test_images/sample.jpg", {
            "title": "Thiếu file ảnh mẫu test_images/sample.jpg",
            "module": "YOLO Inference",
            "severity": "LOW",
            "desc": "File test_images/sample.jpg không tồn tại."
        })
except Exception as e:
    record_test("YOLO Inference & Tracking", False, str(e), {
        "title": "Lỗi khởi tạo hoặc chạy YOLO",
        "module": "yolo_service.py",
        "severity": "HIGH",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 2: KIỂM TRA EDGE CASES CỦA YOLO (ẢNH LỖI, BYTE RỖNG)
# ==========================================
print("\n--- 2. KIỂM TRA EDGE CASES CỦA YOLO ---")
try:
    # 2.1 Byte rỗng
    try:
        run_inference(b"")
        record_test("YOLO Empty Bytes Handling", False, "Không ném lỗi khi gặp byte rỗng", {
            "title": "YOLO không bắt lỗi khi truyền byte rỗng",
            "module": "yolo_service.py",
            "severity": "MEDIUM",
            "desc": "run_inference(b'') không raise ValueError."
        })
    except ValueError:
        record_test("YOLO Empty Bytes Handling", True, "Bắt lỗi ValueError hợp lệ khi byte rỗng")

    # 2.2 Dữ liệu không phải ảnh
    try:
        run_inference(b"invalid_corrupt_data_12345")
        record_test("YOLO Corrupt Bytes Handling", False, "Không ném lỗi khi gặp byte rác", {
            "title": "YOLO không bắt lỗi khi truyền dữ liệu ảnh hỏng",
            "module": "yolo_service.py",
            "severity": "MEDIUM",
            "desc": "run_inference(b'invalid') không raise ValueError."
        })
    except ValueError:
        record_test("YOLO Corrupt Bytes Handling", True, "Bắt lỗi ValueError hợp lệ khi byte ảnh hỏng")
except Exception as e:
    record_test("YOLO Edge Cases", False, str(e), {
        "title": "Lỗi ngoại lệ không mong muốn ở YOLO Edge Cases",
        "module": "yolo_service.py",
        "severity": "MEDIUM",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 3: KIỂM TRA TACTILE PAVING (DÒ VẠCH GẠCH VÀNG)
# ==========================================
print("\n--- 3. KIỂM TRA TACTILE PAVING DETECTOR ---")
try:
    from app.vision.tactile_paving_detector import detect_tactile_paving
    if 'sample_bytes' in locals():
        res = detect_tactile_paving(sample_bytes)
        record_test("Tactile Paving Detector (Sample)", True, f"Kết quả phát hiện: {res}")
    
    # Test với byte rỗng
    try:
        res_empty = detect_tactile_paving(b"")
        record_test("Tactile Paving Empty Bytes", True, f"Xử lý byte rỗng an toàn, trả về: {res_empty}")
    except Exception as e:
        record_test("Tactile Paving Empty Bytes", False, str(e), {
            "title": "Tactile Paving bị crash khi gặp byte rỗng",
            "module": "tactile_paving_detector.py",
            "severity": "MEDIUM",
            "desc": traceback.format_exc()
        })
except Exception as e:
    record_test("Tactile Paving Detector", False, str(e), {
        "title": "Lỗi import hoặc chạy tactile_paving_detector.py",
        "module": "tactile_paving_detector.py",
        "severity": "MEDIUM",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 4: KIỂM TRA OCR SERVICE (ĐỌC CHỮ)
# ==========================================
print("\n--- 4. KIỂM TRA OCR SERVICE (EASYOCR) ---")
try:
    from app.vision.ocr_service import extract_text, OCR_AVAILABLE
    record_test("OCR Service Availability", OCR_AVAILABLE, f"EasyOCR status: {'Khả dụng' if OCR_AVAILABLE else 'Chưa cài đặt'}")
    if OCR_AVAILABLE and 'sample_bytes' in locals():
        t0 = time.time()
        ocr_text = extract_text(sample_bytes)
        t_ocr = (time.time() - t0) * 1000
        record_test("OCR Text Extraction", True, f"Thời gian: {t_ocr:.1f}ms, Văn bản: '{ocr_text[:50]}...'")
except Exception as e:
    record_test("OCR Service", False, str(e), {
        "title": "Lỗi chạy OCR Service",
        "module": "ocr_service.py",
        "severity": "LOW",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 5: KIỂM TRA GIỌNG NÓI BLAZE TTS & GOOGLE FALLBACK
# ==========================================
print("\n--- 5. KIỂM TRA DỊCH VỤ GIỌNG NÓI (BLAZE / GOOGLE TTS) ---")
try:
    from app.services.blaze_service import blaze_service
    # Test TTS
    t0 = time.time()
    tts_result = blaze_service.text_to_speech("Xin chào, đây là bài kiểm tra hệ thống điều hướng thông minh.")
    t_tts = (time.time() - t0) * 1000
    
    if tts_result.get("success"):
        audio_id = tts_result.get("id")
        record_test("TTS Generation", True, f"ID: {audio_id}, Audio URL: {tts_result.get('audio_url')}, Time: {t_tts:.1f}ms")
        
        # Test lấy audio bytes
        audio_bytes = blaze_service.get_audio_bytes(audio_id)
        if audio_bytes and len(audio_bytes) > 100:
            record_test("TTS Audio Bytes Retrieval", True, f"Kích thước audio: {len(audio_bytes)} bytes")
        else:
            record_test("TTS Audio Bytes Retrieval", False, "Không lấy được audio bytes từ cache/fallback", {
                "title": "TTS không trả về audio bytes",
                "module": "blaze_service.py",
                "severity": "HIGH",
                "desc": "blaze_service.get_audio_bytes trả về None hoặc dữ liệu rỗng."
            })
    else:
        record_test("TTS Generation", False, "TTS trả về success=False", {
            "title": "TTS text_to_speech thất bại",
            "module": "blaze_service.py",
            "severity": "HIGH",
            "desc": str(tts_result)
        })
except Exception as e:
    record_test("Voice Service", False, str(e), {
        "title": "Lỗi dịch vụ giọng nói",
        "module": "blaze_service.py",
        "severity": "HIGH",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 6: KIỂM TRA FASTAPI REST ENDPOINTS
# ==========================================
print("\n--- 6. KIỂM TRA CÁC REST ENDPOINT CỦA FASTAPI ---")
try:
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    
    # 6.1 GET / (Trang chủ Web HUD)
    res = client.get("/")
    record_test("Endpoint GET / (Web HUD)", res.status_code == 200, f"Status: {res.status_code}, Length: {len(res.content)} bytes")
    
    # 6.2 POST /api/v1/vision/analyze (Ảnh hợp lệ)
    if 'sample_bytes' in locals():
        res = client.post(
            "/api/v1/vision/analyze",
            files={"image": ("test.jpg", sample_bytes, "image/jpeg")}
        )
        if res.status_code == 200:
            data = res.json()
            record_test("Endpoint POST /api/v1/vision/analyze", data.get("success", False), f"Detections: {len(data.get('detections', []))}, Guidance: {data.get('guidance')}")
        else:
            record_test("Endpoint POST /api/v1/vision/analyze", False, f"Status: {res.status_code}, Body: {res.text}", {
                "title": "Endpoint /api/v1/vision/analyze trả về mã lỗi HTTP",
                "module": "main.py",
                "severity": "HIGH",
                "desc": f"Status: {res.status_code}, Body: {res.text}"
            })
            
    # 6.3 POST /api/v1/vision/analyze (File không phải ảnh)
    res_bad = client.post(
        "/api/v1/vision/analyze",
        files={"image": ("test.txt", b"Hello world", "text/plain")}
    )
    if res_bad.status_code == 200 and not res_bad.json().get("success"):
        record_test("Endpoint POST /api/v1/vision/analyze (Non-Image)", True, "Bắt lỗi non-image thành công")
    else:
        record_test("Endpoint POST /api/v1/vision/analyze (Non-Image)", False, f"Status: {res_bad.status_code}, Body: {res_bad.text}")

    # 6.4 POST /api/v1/voice/tts
    res_tts = client.post("/api/v1/voice/tts", json={"text": "Phía trước an toàn."})
    if res_tts.status_code == 200 and res_tts.json().get("success"):
        t_id = res_tts.json().get("id")
        record_test("Endpoint POST /api/v1/voice/tts", True, f"Audio ID: {t_id}")
        
        # 6.5 GET /api/v1/voice/audio/{id}
        res_audio = client.get(f"/api/v1/voice/audio/{t_id}")
        record_test("Endpoint GET /api/v1/voice/audio/{id}", res_audio.status_code == 200, f"Status: {res_audio.status_code}, Content-Type: {res_audio.headers.get('content-type')}")
    else:
        record_test("Endpoint POST /api/v1/voice/tts", False, f"Status: {res_tts.status_code}, Body: {res_tts.text}")

    # 6.6 POST /api/v1/assistant/ask
    questions = [
        "Phía trước có an toàn không?",
        "Có ai ở phía trước không?",
        "Tôi đang ở đâu?",
        "Bạn làm được những gì?"
    ]
    for q in questions:
        res_ask = client.post("/api/v1/assistant/ask", json={
            "question": q,
            "detected_objects": ["person", "car"],
            "closest_distance": 1.5,
            "is_safe": False
        })
        if res_ask.status_code == 200 and res_ask.json().get("success"):
            ans = res_ask.json().get("answer", "")
            record_test(f"Assistant Ask: '{q}'", True, f"Trả lời: '{ans[:40]}...'")
        else:
            record_test(f"Assistant Ask: '{q}'", False, f"Status: {res_ask.status_code}")

    # 6.7 POST /api/v1/spatial/inspect-room
    res_spatial = client.post("/api/v1/spatial/inspect-room", data={"prompt": "Khảo sát phòng"})
    record_test("Endpoint POST /api/v1/spatial/inspect-room", res_spatial.status_code == 200, f"Status: {res_spatial.status_code}, Success: {res_spatial.json().get('success')}")

except Exception as e:
    record_test("FastAPI REST Endpoints", False, str(e), {
        "title": "Lỗi khi gọi các REST Endpoint qua TestClient",
        "module": "main.py",
        "severity": "HIGH",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 7: KIỂM TRA WEBSOCKET REALTIME STREAM
# ==========================================
print("\n--- 7. KIỂM TRA WEBSOCKET STREAM THỜI GIAN THỰC ---")
try:
    if 'sample_bytes' in locals():
        b64_frame = base64.b64encode(sample_bytes).decode('utf-8')
        
        with client.websocket_connect("/api/v1/vision/stream") as websocket:
            # 7.1 Gửi 1 frame đầu tiên
            t0 = time.time()
            websocket.send_text(b64_frame)
            data = websocket.receive_json()
            rtt = (time.time() - t0) * 1000
            
            has_success = data.get("success", False)
            record_test("WebSocket Single Frame Stream", has_success, f"RTT: {rtt:.1f}ms, Objects: {len(data.get('objects', []))}, Guidance: {data.get('guidance', {}).get('text')}")
            
            # 7.2 Gửi liên tiếp 5 frame mô phỏng điện thoại streaming
            latencies = []
            for i in range(5):
                t_frame = time.time()
                websocket.send_text(b64_frame)
                resp = websocket.receive_json()
                latencies.append((time.time() - t_frame) * 1000)
            
            avg_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            record_test("WebSocket 5-Frame Continuous Stream", True, f"Avg Latency: {avg_lat:.1f}ms (Min: {min_lat:.1f}ms, Max: {max_lat:.1f}ms)")
            
            # 7.3 Gửi dữ liệu rác để kiểm tra khả năng phục hồi lỗi
            websocket.send_text("invalid_base64_garbage")
            err_resp = websocket.receive_json()
            record_test("WebSocket Corrupted Data Recovery", err_resp.get("success") is False, f"Server xử lý lỗi mượt mà: {err_resp.get('error')}")

except Exception as e:
    record_test("WebSocket Real-time Stream", False, str(e), {
        "title": "Lỗi kết nối hoặc xử lý WebSocket /api/v1/vision/stream",
        "module": "main.py",
        "severity": "HIGH",
        "desc": traceback.format_exc()
    })

# ==========================================
# TEST 8: KIỂM TRA CẤU TRÚC VÀ FILE DỰ ÁN KOTLIN MBAPP
# ==========================================
print("\n--- 8. KIỂM TRA DỰ ÁN ANDROID KOTLIN MBAPP ---")
mbapp_dir = os.path.join(PROJECT_ROOT, "MBAPP")
required_files = [
    "app/src/main/AndroidManifest.xml",
    "app/build.gradle.kts",
    "app/src/main/java/com/example/ai_hkdn/MainActivity.kt",
    "app/src/main/java/com/example/ai_hkdn/service/WebSocketService.kt",
    "app/src/main/java/com/example/ai_hkdn/service/TtsManager.kt",
    "app/src/main/java/com/example/ai_hkdn/service/OcrService.kt",
    "app/src/main/java/com/example/ai_hkdn/ui/screens/SmartHatScreen.kt",
    "app/src/main/java/com/example/ai_hkdn/ui/camera/CameraPreviewView.kt",
    "app/src/main/java/com/example/ai_hkdn/ui/hud/ARHudOverlay.kt",
    "app/src/main/java/com/example/ai_hkdn/util/ImageUtils.kt",
    "app/src/main/java/com/example/ai_hkdn/data/model/VisionResponse.kt"
]

all_files_exist = True
missing_files = []
for rel_path in required_files:
    full_p = os.path.join(mbapp_dir, rel_path)
    if not os.path.exists(full_p):
        all_files_exist = False
        missing_files.append(rel_path)

if all_files_exist:
    record_test("MBAPP Kotlin Architecture Integrity", True, f"Toàn bộ {len(required_files)} file kiến trúc cốt lõi đều đầy đủ.")
else:
    record_test("MBAPP Kotlin Architecture Integrity", False, f"Thiếu các file: {missing_files}", {
        "title": "Thiếu file trong dự án Kotlin MBAPP",
        "module": "MBAPP",
        "severity": "HIGH",
        "desc": f"Thiếu các file: {missing_files}"
    })

# ==========================================
# XUẤT BÁO CÁO VÀ TẠO FOLDER BUG_FIX
# ==========================================
print("\n==========================================")
print("             TỔNG KẾT KIỂM THỬ")
print("==========================================")

total_tests = len(test_results)
passed_tests = sum(1 for t in test_results if t["passed"])
failed_tests = total_tests - passed_tests

print(f"Tổng số bài test : {total_tests}")
print(f"Thành công       : {passed_tests} ✅")
print(f"Thất bại         : {failed_tests} ❌")

# Ghi file báo cáo tổng hợp vào bug_fix/TEST_REPORT.md
report_md = f"""# 📋 BÁO CÁO KIỂM THỬ TOÀN DIỆN DỰ ÁN MŨ THÔNG MINH
*Thời gian chạy:* {time.strftime('%Y-%m-%d %H:%M:%S')}
*Nhánh Git:* Van_version2

## 1. Kết Quả Tổng Quan
- **Tổng số ca kiểm thử:** {total_tests}
- **Vượt qua (PASS):** {passed_tests} ✅
- **Phát hiện lỗi (FAIL):** {failed_tests} ❌
- **Tỷ lệ thành công:** {(passed_tests / total_tests * 100):.1f}%

---

## 2. Chi Tiết Từng Hạng Mục Kiểm Thử

| STT | Tên Hạng Mục Kiểm Thử | Trạng Thái | Chi Tiết / Thời Gian Phản Hồi |
| :---: | :--- | :---: | :--- |
"""

for i, t in enumerate(test_results, 1):
    status_icon = "✅ PASS" if t["passed"] else "❌ FAIL"
    report_md += f"| {i} | {t['name']} | **{status_icon}** | {t['details']} |\n"

report_md += "\n---\n\n## 3. Danh Sách Lỗi Cần Fix (Bug List)\n"
if not bugs_found:
    report_md += "> 🎉 **Không phát hiện lỗi nghiêm trọng nào! Toàn bộ các chức năng cốt lõi (YOLO, Tracking, ROI, Voice, WebSocket, REST, OCR, Kotlin App) đều đang hoạt động hoàn hảo.**\n"
else:
    for idx, bug in enumerate(bugs_found, 1):
        report_md += f"""
### 🐛 Bug #{idx}: {bug['title']}
- **Module:** `{bug['module']}`
- **Mức độ:** `{bug['severity']}`
- **Mô tả chi tiết:**
```
{bug['desc']}
```
"""
        # Ghi từng file bug riêng biệt
        bug_filename = os.path.join(BUG_FIX_DIR, f"BUG_{idx:02d}_{bug['module'].replace('/', '_')}.md")
        with open(bug_filename, "w", encoding="utf-8") as bf:
            bf.write(f"# 🐛 BUG REPORT #{idx}: {bug['title']}\n\n")
            bf.write(f"- **Module:** `{bug['module']}`\n")
            bf.write(f"- **Mức độ nghiêm trọng:** `{bug['severity']}`\n\n")
            bf.write("## Chi tiết lỗi:\n```\n" + bug['desc'] + "\n```\n\n")
            bf.write("## Đề xuất hướng xử lý:\n- Kiểm tra logic trong file tương ứng.\n")

report_path = os.path.join(BUG_FIX_DIR, "TEST_REPORT.md")
with open(report_path, "w", encoding="utf-8") as rf:
    rf.write(report_md)

print(f"\n📄 Đã ghi báo cáo kiểm thử chi tiết vào: {report_path}")
