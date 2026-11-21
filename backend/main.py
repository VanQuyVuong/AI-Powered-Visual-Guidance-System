"""
AI Visual Guidance System - Backend API Server (FastAPI)
Đầu não xử lý thị giác, giọng nói tiếng Việt và điều hướng thời gian thực.
"""

import os
import io
import sys
import json
import subprocess
import base64
import asyncio
from typing import Optional

# Tự động chuyển sang Python trong .venv nếu người dùng gõ `python main.py` từ Python hệ thống/Conda
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_python = os.path.join(_project_root, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(_venv_python)):
        sys.exit(subprocess.call([_venv_python, "-u"] + sys.argv))

# Thiết lập UTF-8 cho console Windows (dùng reconfigure an toàn, không đóng nhầm buffer của pytest)
if "PYTEST_CURRENT_TEST" not in os.environ:
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace", write_through=True)
            except Exception:
                pass

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn

# Schemas
from app.schemas.vision import (
    VisionAnalyzeResponse,
    Detection,
    Guidance,
    BoundingBox,
    build_light_client_payload,
)

# Services & Vision Modules
from app.vision.pipeline import analyze_frame_sync
from app.vision.distance_estimator import estimate_distance_m
from app.vision.ocr_service import extract_text
from app.vision.spatial_service import spatial_service
from app.services.blaze_service import blaze_service
from app.services.assistant_service import AssistantAskRequest, generate_assistant_answer

# Alias tương thích ngược cho các script test cũ nếu có
_analyze_frame_sync = analyze_frame_sync

app = FastAPI(
    title="AI Visual Guidance API",
    description="Hệ thống dẫn đường thị giác thời gian thực cho người khiếm thị",
    version="1.0.0"
)

# Cấu hình CORS cho phép Mobile App và Web HUD truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAM Service Router (MobileSAM on-demand segmentation)
from app.vision.sam_router import router as sam_router
app.include_router(sam_router)


# ==========================================
# 1. GIAO DIỆN WEB & STATIC DEMO
# ==========================================
@app.get("/")
async def serve_web_demo():
    """Phục vụ giao diện Trung tâm giám sát 4 Model AI & Video từ điện thoại."""
    phone_stream_file = os.path.join(os.path.dirname(__file__), "view_phone_stream.html")
    if os.path.exists(phone_stream_file):
        return FileResponse(phone_stream_file, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "AI Visual Guidance API is running. Web demo not found."}


@app.get("/api/v1/sample-3d-video")
async def get_sample_3d_video():
    """Phục vụ video mẫu 3D SpatialLM cho giao diện web."""
    video_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "Video_test_3d.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video_test_3d.mp4 not found")


@app.get("/download-apk")
async def download_apk():
    """Tải file APK Android mới nhất trực tiếp về điện thoại qua Wi-Fi mà không cần cắm cáp."""
    apk_path = os.path.join(os.path.dirname(__file__), "..", "MBAPP", "app", "build", "outputs", "apk", "debug", "app-debug.apk")
    if os.path.exists(apk_path):
        return FileResponse(apk_path, media_type="application/vnd.android.package-archive", filename="app-debug.apk")
    raise HTTPException(status_code=404, detail="Không tìm thấy file APK.")


# ==========================================
# 2. API THỊ GIÁC (VISION REST)
# ==========================================
@app.post("/api/v1/vision/analyze", response_model=VisionAnalyzeResponse)
async def analyze_image(image: UploadFile = File(...)):
    """API phân tích ảnh tĩnh: YOLO + Depth + Semantic Seg + Decision Engine."""
    if not image.content_type.startswith("image/"):
        return VisionAnalyzeResponse(
            success=False,
            detections=[],
            error={"code": "INVALID_IMAGE", "message": "File tải lên không phải là ảnh."}
        )

    try:
        contents = await image.read()
        result = await asyncio.to_thread(analyze_frame_sync, contents, False)
        final_detections = [
            Detection(
                class_name=obj["class_name"],
                class_name_vn=obj.get("class_name_vn"),
                confidence=obj["confidence"],
                position=obj["position"],
                distance_m=obj.get("distance_m"),
                in_roi=obj.get("in_roi"),
                is_danger=obj.get("is_danger"),
                bounding_box=BoundingBox(**obj["bounding_box"]),
            )
            for obj in result.get("objects", [])
        ]
        g_data = result.get("guidance")
        guidance_obj = Guidance(**g_data) if g_data else None

        return VisionAnalyzeResponse(
            success=True,
            detections=final_detections,
            guidance=guidance_obj
        )
    except ValueError as ve:
        return VisionAnalyzeResponse(
            success=False, detections=[], error={"code": "INVALID_IMAGE", "message": str(ve)}
        )
    except Exception as e:
        return VisionAnalyzeResponse(
            success=False, detections=[], error={"code": "INTERNAL_SERVER_ERROR", "message": str(e)}
        )


@app.post("/api/v1/vision/read_text")
async def read_text_api(image: UploadFile = File(...)):
    """API đọc chữ (OCR) trên biển báo, sách, nhãn hàng."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File không hợp lệ.")

    contents = await image.read()
    text = await asyncio.to_thread(extract_text, contents)
    return {"success": True, "text": text}


@app.get("/view-phone")
async def view_phone_stream_page():
    """Phục vụ giao diện xem video stream trực tiếp từ điện thoại."""
    html_path = os.path.join(os.path.dirname(__file__), "view_phone_stream.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})
    raise HTTPException(status_code=404, detail="view_phone_stream.html not found")


# ==========================================
# 3. WEBSOCKET STREAM THỜI GIAN THỰC (REAL-TIME HUD & REMOTE MONITOR)
# ==========================================
class ConnectionManager:
    """Quản lý các kết nối WebSocket cho cả Phone Stream và Web Remote Viewer."""
    def __init__(self):
        self.active_connections: list = []
        self.viewer_connections: set = set()
        self.latest_result: Optional[dict] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📱 Client kết nối. Tổng số kết nối: {len(self.active_connections)}")

    def mark_viewer(self, websocket: WebSocket):
        self.viewer_connections.add(websocket)

    def has_viewers(self) -> bool:
        return len(self.viewer_connections) > 0

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.viewer_connections.discard(websocket)
        print(f"📱 Client ngắt kết nối. Tổng số kết nối: {len(self.active_connections)}")

    async def broadcast(self, message: dict, sender_ws: Optional[WebSocket] = None):
        """
        Phát kết quả theo từng vai trò để chống nghẽn Wi-Fi & lag trình duyệt:
        - Phone/Sender nhận payload nhẹ (~1-2KB), không gửi ngược ảnh frame_b64/depth_b64.
        - Web Monitor (VIEW_ONLY) nhận full payload (kèm depth & Fusion HUD) đúng 1 lần mỗi frame mới.
        """
        self.latest_result = message
        light_msg = None
        for conn in list(self.active_connections):
            try:
                if conn in self.viewer_connections:
                    await conn.send_json(message)
                else:
                    if light_msg is None:
                        light_msg = build_light_client_payload(message)
                    await conn.send_json(light_msg)
            except Exception:
                pass

    async def broadcast_to_viewers(self, message: dict):
        """
        Chuyển tiếp telemetry & khung hình do điện thoại tự xử lý trên NPU/Edge sang Web Monitor
        trên Laptop (0% tải suy luận trên Laptop và không gửi dội ngược về điện thoại).
        """
        self.latest_result = message
        for conn in list(self.viewer_connections):
            try:
                await conn.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


@app.websocket("/api/v1/vision/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket Stream phân tích video thời gian thực từ Mobile App / Web:
    - Phone Server-Assist: Gửi frame base64 -> Laptop chạy AI & trả JSON gọn -> Broadcast Web Monitor.
    - Phone Edge/NPU Telemetry: Gửi JSON `type=EDGE_TELEMETRY` đã suy luận trên điện thoại -> Laptop chỉ mirror lên Web Monitor.
    - Web Viewer (VIEW_ONLY): Nhận push trực tiếp khi có frame mới.
    """
    await manager.connect(websocket)

    state = {"latest": None, "latest_telemetry": None, "closed": False, "is_viewer": False}
    new_frame = asyncio.Event()

    async def reader():
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "VIEW_ONLY":
                    state["is_viewer"] = True
                    manager.mark_viewer(websocket)
                    if manager.latest_result:
                        try:
                            await websocket.send_json(manager.latest_result)
                        except Exception:
                            pass
                    continue
                stripped = msg.lstrip()
                if stripped.startswith("{"):
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, dict) and (
                            parsed.get("type") == "EDGE_TELEMETRY"
                            or str(parsed.get("mode", "")).startswith("EDGE")
                        ):
                            state["latest_telemetry"] = parsed
                            new_frame.set()
                            continue
                    except Exception:
                        pass
                state["latest"] = msg
                new_frame.set()
        except Exception:
            pass
        finally:
            state["closed"] = True
            new_frame.set()

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            if state["is_viewer"]:
                # Viewer chỉ nhận push sự kiện thực từ manager.broadcast khi có frame mới -> không lặp dữ liệu
                await asyncio.sleep(0.25)
                if state["closed"]:
                    break
                continue

            await new_frame.wait()
            new_frame.clear()
            if state["closed"]:
                break

            tele = state["latest_telemetry"]
            if tele is not None:
                state["latest_telemetry"] = None
                tele.setdefault("success", True)
                tele.setdefault("mode", "EDGE_ON_DEVICE_NPU")
                await manager.broadcast_to_viewers(tele)

            data = state["latest"]
            state["latest"] = None
            if not data:
                continue
            try:
                if "," in data:
                    data = data.split(",", 1)[1]
                image_bytes = base64.b64decode(data)
                result = await asyncio.to_thread(
                    analyze_frame_sync, image_bytes, True, manager.has_viewers()
                )
                # Gửi bản gọn về client gửi và full HUD cho các Web Viewer
                await manager.broadcast(result, sender_ws=websocket)
            except WebSocketDisconnect:
                break
            except Exception as frame_error:
                try:
                    await websocket.send_json({"success": False, "error": str(frame_error)})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ Lỗi WebSocket: {e}")
    finally:
        reader_task.cancel()
        manager.disconnect(websocket)


# ==========================================
# 4. API GIỌNG NÓI & TRỢ LÝ AI (VOICE & ASSISTANT)
# ==========================================
class TtsRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = "HN-Nam-2-BL"
    audio_speed: Optional[str] = "1"


@app.post("/api/v1/voice/tts")
async def voice_tts(request: TtsRequest):
    """Chuyển văn bản thành giọng nói tiếng Việt tự nhiên (Blaze TTS)."""
    return await asyncio.to_thread(
        blaze_service.text_to_speech,
        text=request.text,
        speaker_id=request.speaker_id,
        audio_speed=request.audio_speed
    )


@app.get("/api/v1/voice/audio/{tts_id}")
async def get_audio_proxy(tts_id: str):
    """Stream trực tiếp file MP3 từ Blaze về client với Bearer Token."""
    audio_content = await asyncio.to_thread(blaze_service.get_audio_bytes, tts_id)
    if not audio_content:
        raise HTTPException(status_code=404, detail="Không tải được file âm thanh từ Blaze")
    return Response(content=audio_content, media_type="audio/mpeg")


@app.get("/api/v1/voice/sample")
async def get_sample_voice():
    """Phục vụ file âm thanh mẫu chuẩn tiếng Việt."""
    sample_path = os.path.join(os.path.dirname(__file__), "hn-viet-bach.mp3")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="File sample giọng đọc không tồn tại")


@app.post("/api/v1/voice/stt")
async def voice_stt(audio: UploadFile = File(...)):
    """Nhận diện giọng nói tiếng Việt từ file âm thanh người dùng (Blaze STT)."""
    audio_bytes = await audio.read()
    return await asyncio.to_thread(
        blaze_service.speech_to_text, audio_bytes, filename=audio.filename or "speech.wav"
    )


@app.post("/api/v1/assistant/ask")
async def assistant_ask(request: AssistantAskRequest):
    """Trợ lý đàm thoại AI thông minh hỗ trợ trả lời câu hỏi của người khiếm thị."""
    answer = generate_assistant_answer(
        question=request.question,
        current_step=request.current_step,
        detected_objects=request.detected_objects,
        is_safe=request.is_safe,
        closest_distance=request.closest_distance
    )
    tts_res = await asyncio.to_thread(blaze_service.text_to_speech, answer)
    return {
        "success": True,
        "question": request.question,
        "answer": answer,
        "audio_url": tts_res.get("audio_url")
    }


# ==========================================
# 5. KHẢO SÁT KHÔNG GIAN 3D (SPATIAL VISION)
# ==========================================
@app.post("/api/v1/spatial/inspect-room")
async def inspect_room_api(
    image: Optional[UploadFile] = File(None),
    prompt: Optional[str] = "Khảo sát không gian phòng và định vị vật thể"
):
    """Prototype khảo sát không gian phòng 3D (SpatialLM)."""
    img_bytes = None
    if image and image.content_type and image.content_type.startswith("image/"):
        img_bytes = await image.read()

    return spatial_service.inspect_room(image_bytes=img_bytes, custom_prompt=prompt)


def get_local_ip() -> str:
    """Tự động phát hiện địa chỉ IP Wi-Fi nội bộ của laptop."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


if __name__ == "__main__":
    import threading
    import webbrowser
    import time

    local_ip = get_local_ip()

    print("\n" + "=" * 70)
    print("🚀 AI VISUAL GUIDANCE SYSTEM - ĐẦU NÃO XỬ LÝ ĐÃ SẴN SÀNG!")
    print(f"🌐 Màn hình Web Monitor (Laptop):     http://localhost:8000/view-phone")
    print(f"📱 ĐỊA CHỈ IP CHO ĐIỆN THOẠI KẾT NỐI: {local_ip}")
    print(f"   (Nhập vào ô cài đặt trên App:      http://{local_ip}:8000)")
    print(f"📥 Link tải APK mới nhất qua Wi-Fi:   http://{local_ip}:8000/download-apk")
    print("=" * 70)
    print("🌐 Đang tự động mở màn hình Web Monitor trên trình duyệt...\n")

    def _open_browser():
        time.sleep(1.8)
        try:
            webbrowser.open("http://localhost:8000/view-phone")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
