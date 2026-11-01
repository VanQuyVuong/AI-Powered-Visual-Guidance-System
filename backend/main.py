"""
AI Visual Guidance System - Backend API Server (FastAPI)
Đầu não xử lý thị giác, giọng nói tiếng Việt và điều hướng thời gian thực.
"""

import os
import io
import sys
import base64
import asyncio
from typing import Optional

# Thiết lập UTF-8 cho console Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn

# Schemas
from app.schemas.vision import VisionAnalyzeResponse, Detection, Guidance, BoundingBox

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


# ==========================================
# 1. GIAO DIỆN WEB & STATIC DEMO
# ==========================================
@app.get("/")
async def serve_web_demo():
    """Phục vụ giao diện Web Demo AI Visual Guidance System & AR Navigation."""
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
        result = await asyncio.to_thread(analyze_frame_sync, contents)
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


# ==========================================
# 3. WEBSOCKET STREAM THỜI GIAN THỰC (REAL-TIME HUD)
# ==========================================
@app.websocket("/api/v1/vision/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket Stream phân tích video thời gian thực từ Mobile App / Web.
    Sử dụng kiến trúc Drop Frame (chỉ giữ frame mới nhất) để tránh trễ hình.
    """
    await websocket.accept()
    print("📱 Client đã kết nối tới Video Stream!")

    state = {"latest": None, "closed": False}
    new_frame = asyncio.Event()

    async def reader():
        try:
            while True:
                state["latest"] = await websocket.receive_text()
                new_frame.set()
        except Exception:
            pass
        finally:
            state["closed"] = True
            new_frame.set()

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            await new_frame.wait()
            new_frame.clear()
            if state["closed"]:
                break
            data = state["latest"]
            state["latest"] = None
            if not data:
                continue
            try:
                if "," in data:
                    data = data.split(",", 1)[1]
                image_bytes = base64.b64decode(data)
                result = await asyncio.to_thread(analyze_frame_sync, image_bytes)
                await websocket.send_json(result)
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
        print("📱 Client đã ngắt kết nối.")


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
