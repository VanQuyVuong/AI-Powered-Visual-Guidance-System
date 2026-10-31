from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import base64
import asyncio

BACKEND_DIR = Path(__file__).resolve().parent

# Import các thành phần chúng ta vừa tạo
from app.schemas.vision import VisionAnalyzeResponse, Detection, Guidance, BoundingBox
from app.vision.yolo_service import run_inference, run_tracking
from app.vision.decision_engine import (
    determine_position, generate_guidance, is_in_roi,
    estimate_distance, estimate_distance_from_depth, cluster_objects,
    set_mode, get_mode, generate_navigation_direction,
    update_user_position, get_user_movement_direction, answer_question
)
from app.vision import depth_service as _depth_svc

try:
    from app.voice.api_integration import queue_guidance_tts, pop_pending_tts
except Exception as _tts_import_err:
    print(f"[TTS] bo qua luc khoi dong: {_tts_import_err}")

    def queue_guidance_tts(text, priority="MEDIUM"):
        return None

    def pop_pending_tts():
        return "", "", "NONE"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Depth Anything V2 trên thread riêng, không chặn WebSocket/YOLO
    _depth_svc.start_background_load()
    yield


app = FastAPI(title="AI Visual Guidance API", version="0.1 (MVP)", lifespan=lifespan)

# Cho phép App trên điện thoại gọi API mà không bị chặn CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def demo_page():
    """Mở trang test camera tại http://localhost:8000/ (đừng mở file HTML trực tiếp)."""
    html = BACKEND_DIR / "test_websocket.html"
    if not html.is_file():
        raise HTTPException(404, "Khong tim thay test_websocket.html")
    return FileResponse(html, media_type="text/html")


@app.get("/view-phone")
def view_phone_stream():
    """Mở trang xem stream từ điện thoại tại http://localhost:8000/view-phone"""
    html = BACKEND_DIR / "view_phone_stream.html"
    if not html.is_file():
        raise HTTPException(404, "Khong tim thay view_phone_stream.html")
    return FileResponse(html, media_type="text/html")


@app.get("/demo/bus.jpg")
def demo_bus_image():
    img = BACKEND_DIR / "bus.jpg"
    if not img.is_file():
        raise HTTPException(404, "Thieu file bus.jpg (anh mau test YOLO)")
    return FileResponse(img, media_type="image/jpeg")


def _build_objects_payload(raw_detections, img_width, img_height, depth_map=None):
    from app.vision.decision_engine import is_in_roi, determine_position, estimate_distance, estimate_distance_from_depth
    from app.vision.depth_service import get_object_depth_value

    # Bước 1: Gom nhóm chồng lấp (loại phụ kiện nằm trong người, xe...)
    clustered = cluster_objects(raw_detections)

    objects_payload = []
    for det in clustered:
        box = det["bounding_box"]
        in_roi = is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
        pos = determine_position(box["x1"], box["x2"], img_width)

        # Ưu tiên dùng Depth AI nếu có depth map, fallback sang phối cảnh
        depth_val = get_object_depth_value(depth_map, box["x1"], box["y1"], box["x2"], box["y2"])
        if depth_map is not None and depth_val >= 0:
            dist_info = estimate_distance_from_depth(depth_val)
        else:
            dist_info = estimate_distance(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
            depth_val = -1.0

        objects_payload.append({
            "id": det.get("id"),
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "position": pos,
            "in_roi": in_roi,
            "distance": dist_info,
            "depth_value": round(float(depth_val), 3),
            "bounding_box": box,
        })

    # Lấy ROI objects (ĐẦY ĐỦ distance info) để generate guidance
    roi_objects = [obj for obj in objects_payload if obj["in_roi"]]
    guidance_data = generate_guidance(roi_objects, img_width, img_height, depth_map) or {"text": "", "priority": "NONE"}
    return objects_payload, guidance_data


@app.get("/api/v1/vision/depth/status")
def depth_status():
    return {"success": True, **_depth_svc.public_status()}


@app.post("/api/v1/vision/depth/test")
async def depth_test():
    """Chay depth tren anh mau bus.jpg de kiem tra model."""
    img = BACKEND_DIR / "bus.jpg"
    if not img.is_file():
        raise HTTPException(404, "Thieu backend/bus.jpg")
    result = await asyncio.to_thread(_depth_svc.get_depth, img.read_bytes())
    return {
        "success": True,
        "available": result.get("available"),
        "status": result.get("status") or _depth_svc.load_status(),
        "error": result.get("error"),
        "cached": result.get("cached", False),
        "colormap_b64": result.get("depth_colormap_b64"),
        "map_shape": list(result["depth_map"].shape) if result.get("depth_map") is not None else None,
    }


@app.get("/api/v1/vision/demo")
def demo_detect():
    """Test YOLO + guidance tren anh mau, khong can camera."""
    img = BACKEND_DIR / "bus.jpg"
    if not img.is_file():
        raise HTTPException(404, "Thieu backend/bus.jpg")
    dets, w, h = run_inference(img.read_bytes())
    objects_payload, guidance_data = _build_objects_payload(dets, w, h)
    if guidance_data.get("text"):
        queue_guidance_tts(guidance_data["text"], guidance_data.get("priority", "MEDIUM"))
        import time
        for _ in range(20):
            audio, queued_text, queued_priority = pop_pending_tts()
            if audio:
                guidance_data["audio_base64"] = audio
                break
            time.sleep(0.5)
    return {
        "success": True,
        "objects": objects_payload,
        "guidance": guidance_data,
        "frame_meta": {"width": w, "height": h},
    }

@app.post("/api/v1/vision/analyze", response_model=VisionAnalyzeResponse)
async def analyze_image(image: UploadFile = File(...)):
    """
    API nhận ảnh từ điện thoại, chạy qua YOLO và Decision Engine để trả về câu cảnh báo.
    """
    # 1. Kiểm tra file đầu vào
    if not image.content_type.startswith("image/"):
        return VisionAnalyzeResponse(
            success=False,
            detections=[],
            error={"code": "INVALID_IMAGE", "message": "File tải lên không phải là ảnh."}
        )
        
    try:
        # Đọc dữ liệu file
        contents = await image.read()
        
        # 2. Bước 1 của Pipeline: Chạy YOLO lấy Bounding Box
        raw_detections, img_width, img_height = run_inference(contents)
        
        # 3. Bước 2 của Pipeline: Decision Engine tính toán Position (Trái/Giữa/Phải)
        final_detections = []
        for det in raw_detections:
            pos = determine_position(det["bounding_box"]["x1"], det["bounding_box"]["x2"], img_width)
            
            det_obj = Detection(
                class_name=det["class_name"],
                confidence=det["confidence"],
                position=pos,
                bounding_box=BoundingBox(**det["bounding_box"])
            )
            final_detections.append(det_obj)

        # 4. Bước 3 của Pipeline: Sinh câu tiếng Việt
        guidance_data = generate_guidance(final_detections, img_width, img_height, None)
        guidance_obj = Guidance(**guidance_data) if guidance_data else None
        
        # 5. Trả kết quả JSON về cho App
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

# ==========================================
# API ĐỌC VĂN BẢN (OCR)
# ==========================================
from app.vision.ocr_service import extract_text

@app.post("/api/v1/vision/read_text")
async def read_text_api(image: UploadFile = File(...)):
    """
    API dành riêng cho tính năng đọc chữ.
    Người dùng giơ camera vào một bảng hiệu hoặc tờ giấy, AI sẽ đọc chữ trên đó.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File không hợp lệ.")

    contents = await image.read()
    text = extract_text(contents)

    return {
        "success": True,
        "text": text
    }


@app.post("/api/v1/vision/mode")
async def set_system_mode(mode_data: dict):
    """
    API để chuyển đổi chế độ hệ thống:
    - NAVIGATION: Chỉ hướng dẫn đường đi (mặc định)
    - CONVERSATION: Trả lời chi tiết về vật cản khi người dùng hỏi
    """
    mode = mode_data.get("mode", "NAVIGATION")
    if mode not in ["NAVIGATION", "CONVERSATION"]:
        return {"success": False, "error": "Invalid mode. Use NAVIGATION or CONVERSATION"}

    set_mode(mode)
    intro = (
        "Đã chuyển sang hỏi đáp. Bạn muốn hỏi gì về vật cản?"
        if mode == "CONVERSATION"
        else "Đã quay lại hướng dẫn đường."
    )
    queue_guidance_tts(intro, "MEDIUM")
    return {"success": True, "mode": mode, "message": intro}


@app.get("/api/v1/vision/mode")
async def get_system_mode():
    """API để lấy chế độ hiện tại"""
    return {"success": True, "mode": get_mode()}


@app.post("/api/v1/vision/qa")
async def answer_user_question(qa_data: dict):
    """
    API để trả lời câu hỏi của người dùng
    Body: {"question": "câu hỏi", "detections": [...]}
    """
    question = qa_data.get("question", "")
    detections = qa_data.get("detections", [])
    img_width = qa_data.get("img_width", 640)
    img_height = qa_data.get("img_height", 480)

    answer = answer_question(question, detections, img_width, img_height)
    text = answer.get("text") or ""
    priority = answer.get("priority", "LOW")
    if text:
        queue_guidance_tts(text, priority)

    return {
        "success": True,
        "question": question,
        "answer": text,
        "priority": priority,
    }

# ==========================================
# WEBSOCKET CHO REAL-TIME VIDEO TRACKING
# ==========================================

# Connection manager để hỗ trợ nhiều client
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.latest_result = None  # Lưu kết quả mới nhất

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📱 Client kết nối. Tổng: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"📱 Client ngắt kết nối. Tổng: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Gửi kết quả đến tất cả client"""
        self.latest_result = message
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Lỗi gửi đến client: {e}")

manager = ConnectionManager()

@app.websocket("/api/v1/vision/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Nếu có kết quả mới nhất, gửi ngay cho client mới
        if manager.latest_result:
            await websocket.send_json(manager.latest_result)

        frame_i = 0
        while True:
            try:
                # 1. Nhận frame ảnh từ App (dạng chuỗi base64)
                # Timeout 1s để có thể broadcast kết quả mới nhất cho viewer
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Timeout = không có data từ phone, chỉ broadcast kết quả mới nhất nếu có
                    if manager.latest_result:
                        await websocket.send_json(manager.latest_result)
                    continue

                # Nếu client chỉ muốn xem (không gửi video)
                if data == "VIEW_ONLY":
                    # Client này chỉ xem, gửi kết quả mới nhất liên tục
                    while True:
                        if manager.latest_result:
                            await websocket.send_json(manager.latest_result)
                        await asyncio.sleep(0.1)

                if data.startswith("data:") and "," in data:
                    data = data.split(",", 1)[1]
                image_bytes = base64.b64decode(data)

                raw_detections, img_width, img_height = run_inference(image_bytes)
                frame_i += 1
                names = [d["class_name"] for d in raw_detections[:8]]
                if frame_i <= 5 or frame_i % 20 == 1:
                    print(f"[YOLO] frame {frame_i}: bytes={len(image_bytes)} size={img_width}x{img_height} n={len(raw_detections)} {names}")

                # Depth Anything V2 – chạy thread pool để không block event loop
                depth_result = await asyncio.to_thread(_depth_svc.get_depth, image_bytes)
                depth_map    = depth_result.get("depth_map")       # np.ndarray hoặc None
                depth_b64    = depth_result.get("depth_colormap_b64")  # base64 JPEG
                depth_ok     = depth_result.get("available", False)

                from app.vision.tactile_paving_detector import detect_tactile_paving
                is_on_tactile = detect_tactile_paving(image_bytes)
                objects_payload, guidance_data = _build_objects_payload(
                    raw_detections, img_width, img_height, depth_map
                )

                # Navigation direction + guidance đều dùng chung roi_objects
                roi_objects = [d for d in objects_payload if d["in_roi"]]
                nav_direction = generate_navigation_direction(
                    roi_objects,
                    img_width, img_height
                )

                # Debug: log guidance decision
                if frame_i <= 5 or frame_i % 30 == 0:
                    roi_summary = [(o["class_name"], o["position"], o["distance"].get("level")) for o in roi_objects[:5]]
                    print(f"[NAV] frame {frame_i}: roi={len(roi_objects)} dir={nav_direction['direction']} guide='{guidance_data.get('text','')}' roi_detail={roi_summary}")

                # 5. TTS chạy nền — không chặn YOLO. Gắn audio khi job Blaze xong.
                if guidance_data.get("text"):
                    queue_guidance_tts(
                        guidance_data["text"],
                        guidance_data.get("priority", "MEDIUM"),
                    )
                audio_base64, queued_text, queued_priority = pop_pending_tts()
                if audio_base64:
                    guidance_data["audio_base64"] = audio_base64
                    if not guidance_data.get("text"):
                        guidance_data["text"] = queued_text
                        guidance_data["priority"] = queued_priority

                # 6. Bổ sung thông tin gạch dẫn đường (tactile paving)
                if is_on_tactile:
                    if guidance_data.get("text"):
                        guidance_data["text"] = "Đang đi đúng vạch dẫn đường. " + guidance_data["text"]
                    else:
                        guidance_data["text"] = "Đang đi đúng vạch dẫn đường."
                        guidance_data["priority"] = "LOW"

                # 7. Convert frame ảnh sang base64 để gửi về web
                import cv2
                import numpy as np
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                _, buffer = cv2.imencode('.jpg', img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # 8. Gửi kết quả về TẤT CẢ client (broadcast)
                result = {
                    "success":    True,
                    "mode":       get_mode(),
                    "objects":    objects_payload,
                    "guidance":   guidance_data,
                    "navigation": nav_direction,
                    "frame_meta": {"width": img_width, "height": img_height},
                    "frame_b64":  frame_b64,
                    "depth": {
                        "available":     depth_ok,
                        "colormap_b64":  depth_b64,
                        "cached":        depth_result.get("cached", False),
                        "error":         depth_result.get("error"),
                        "status":        depth_result.get("status"),
                    }
                }
                await manager.broadcast(result)
            except Exception as frame_error:
                print(f"[YOLO] frame error: {frame_error}")
                await websocket.send_json({
                    "success": False,
                    "error": str(frame_error)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        print(f"❌ Lỗi WebSocket: {e}")

# Dành cho việc chạy Backend
if __name__ == "__main__":
    # reload=True se restart process khi YOLO/HF ghi file -> depth load mai khong xong
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
