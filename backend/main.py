from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import base64
import os
import asyncio

# Import các thành phần chúng ta vừa tạo
from app.schemas.vision import VisionAnalyzeResponse, Detection, Guidance, BoundingBox
from app.vision.yolo_service import run_inference, run_tracking
from app.vision.decision_engine import determine_position, generate_guidance, is_in_roi
from app.services.blaze_service import blaze_service

app = FastAPI(title="AI Visual Guidance API", version="0.1 (MVP)")

# Cho phép App trên điện thoại gọi API mà không bị chặn CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_web_demo():
    """
    Phục vụ giao diện Web Demo AI Visual Guidance System & AR Navigation
    """
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "AI Visual Guidance API is running. Web demo not found."}

@app.get("/api/v1/sample-3d-video")
async def get_sample_3d_video():
    """
    Phục vụ video mẫu 3D SpatialLM (test_images/Video_test_3d.mp4) cho giao diện web
    """
    video_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "Video_test_3d.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video_test_3d.mp4 not found")

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
        raw_detections, img_width = await asyncio.to_thread(run_inference, contents)
        
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
        guidance_data = generate_guidance(final_detections)
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
    text = await asyncio.to_thread(extract_text, contents)
    
    return {
        "success": True,
        "text": text
    }

# ==========================================
# ==========================================
# API GIỌNG NÓI TIẾNG VIỆT (BLAZE.VN TTS PROXY)
# ==========================================
class TtsRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = "HN-Nam-2-BL"
    audio_speed: Optional[str] = "1"

@app.post("/api/v1/voice/tts")
async def voice_tts(request: TtsRequest):
    """
    Chuyển văn bản thành giọng nói tiếng Việt tự nhiên (Blaze TTS v1.5_pro)
    """
    result = await asyncio.to_thread(
        blaze_service.text_to_speech,
        text=request.text,
        speaker_id=request.speaker_id,
        audio_speed=request.audio_speed
    )
    return result

# ==========================================
# API KHẢO SÁT KHÔNG GIAN 3 CHIỀU (3D SPATIAL VISION / IOT)
# ==========================================
from app.vision.spatial_service import spatial_service

@app.post("/api/v1/spatial/inspect-room")
async def inspect_room_api(
    image: Optional[UploadFile] = File(None),
    prompt: Optional[str] = "Khảo sát không gian phòng và định vị vật thể"
):
    """
    Prototype khảo sát phòng 3D.
    Hiện trả stub (SpatialLM chưa inference). Web demo đọc trường vietnamese_description.
    """
    img_bytes = None
    if image and image.content_type and image.content_type.startswith("image/"):
        img_bytes = await image.read()
        
    result = spatial_service.inspect_room(
        image_bytes=img_bytes,
        custom_prompt=prompt
    )
    return result

@app.get("/api/v1/voice/audio/{tts_id}")
async def get_audio_proxy(tts_id: str):
    """
    Stream trực tiếp luồng MP3 từ Blaze.vn về trình duyệt kèm Bearer Token (tránh lỗi 401 Unauthorized)
    """
    audio_content = await asyncio.to_thread(blaze_service.get_audio_bytes, tts_id)
    if not audio_content:
        raise HTTPException(status_code=404, detail="Không tải được file âm thanh từ Blaze")
    return Response(content=audio_content, media_type="audio/mpeg")

@app.get("/api/v1/voice/sample")
async def get_sample_voice():
    """
    Phục vụ trực tiếp file âm thanh mẫu giọng đọc chuẩn Hà Nội (Việt Bách)
    """
    sample_path = os.path.join(os.path.dirname(__file__), "hn-viet-bach.mp3")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="File sample giọng đọc không tồn tại")

@app.post("/api/v1/voice/stt")
async def voice_stt(audio: UploadFile = File(...)):
    """
    Nhận diện giọng nói tiếng Việt từ file audio ghi âm của người dùng (Blaze STT v1.0)
    """
    audio_bytes = await audio.read()
    result = await asyncio.to_thread(
        blaze_service.speech_to_text, audio_bytes, filename=audio.filename or "speech.wav"
    )
    return result


# ==========================================
# API TRỢ LÝ HỘI THOẠI AI (VOICE ASSISTANT)
# ==========================================
# API TRỢ LÝ HỘI THOẠI AI (VOICE ASSISTANT)
# ==========================================
from app.vision.decision_engine import get_vietnamese_name

class AssistantAskRequest(BaseModel):
    question: str
    current_step: Optional[str] = "Chặng 1: Đi thẳng 25m"
    detected_objects: Optional[List[str]] = []
    is_safe: Optional[bool] = True
    closest_distance: Optional[float] = 2.5

@app.post("/api/v1/assistant/ask")
async def assistant_ask(request: AssistantAskRequest):
    """
    Trợ lý AI giao tiếp giọng nói trực tiếp với người khiếm thị.
    Thu gọn vào các nhu cầu thiết thực:
    1. Kiểm tra an toàn lối đi & vật cản
    2. Nhận diện người trước mặt
    3. Hỏi vị trí hiện tại
    4. Hướng dẫn bước đi tiếp theo
    5. Đọc chữ / biển báo
    6. Trợ giúp & hướng dẫn sử dụng
    """
    q = request.question.strip().lower()
    
    # Dịch toàn bộ tên vật cản sang tiếng Việt tự nhiên
    raw_objs = request.detected_objects or []
    vn_objs = [get_vietnamese_name(x) for x in raw_objs]
    
    # 1. HỎI VỀ AN TOÀN / VẬT CẢN PHÍA TRƯỚC
    if any(k in q for k in ["an toàn", "vật cản", "chướng ngại", "đi được không", "thông thoáng", "phía trước", "có gì", "nguy hiểm"]):
        if not request.is_safe or (request.closest_distance is not None and request.closest_distance < 1.8):
            obs_str = ", ".join(vn_objs[:2]) if vn_objs else "vật cản"
            dist_val = f"{request.closest_distance:.1f}" if request.closest_distance is not None else "1"
            answer = f"Cảnh báo! Lối đi phía trước có {obs_str}, cách khoảng {dist_val} mét. Bạn hãy đi chậm lại hoặc chú ý bước chân."
        else:
            answer = "Lối đi phía trước hiện rất thông thoáng và an toàn. Bạn có thể tự tin tiếp tục bước đi thẳng."

    # 2. HỎI VỀ CON NGƯỜI
    elif any(k in q for k in ["người", "ai đó", "có ai"]):
        has_person = any("person" in str(x).lower() or "người" in str(x).lower() for x in raw_objs)
        if has_person:
            dist_str = f"cách khoảng {request.closest_distance:.1f} mét" if request.closest_distance else "ở ngay trước mặt"
            answer = f"Camera phát hiện có người ở phía trước, {dist_str}. Bạn hãy giữ khoảng cách an toàn."
        else:
            answer = "Hiện tại không phát hiện người nào trong lối đi trước mặt bạn."

    # 3. HỎI VỊ TRÍ HIỆN TẠI
    elif any(k in q for k in ["ở đâu", "vị trí", "điểm đến", "đến đâu", "tôi đang ở đâu", "tọa độ"]):
        answer = f"Bạn đang ở {request.current_step}. Hãy chú ý quan sát xung quanh và tiếp tục lộ trình."

    # 4. HỎI HƯỚNG ĐI / BƯỚC TIẾP THEO
    elif any(k in q for k in ["làm gì", "tiếp theo", "hướng dẫn", "rẽ", "đi đâu", "hướng nào", "bước tiếp"]):
        answer = f"Bước tiếp theo: Bạn hãy {request.current_step}. Chú ý giữ thẳng hướng di chuyển."

    # 5. HỎI VỀ ĐỌC CHỮ / BIỂN BÁO
    elif any(k in q for k in ["đọc chữ", "biển báo", "chữ gì", "đọc biển", "đọc sách", "tài liệu", "chữ"]):
        answer = "Chức năng đọc chữ sẵn sàng. Bạn hãy hướng camera về phía biển báo hoặc văn bản, tôi sẽ đọc to nội dung cho bạn."

    # 6. YÊU CẦU TRỢ GIÚP / HƯỚNG DẪN CÂU HỎI
    elif any(k in q for k in ["giúp", "trợ giúp", "hướng dẫn tôi", "bạn làm được gì", "câu lệnh", "hỏi gì"]):
        answer = "Tôi có thể hỗ trợ bạn: Kiểm tra an toàn phía trước, phát hiện người, đọc biển báo, hoặc thông báo vị trí hiện tại."

    # 7. CHÀO HỎI / XÃ GIAO
    elif any(k in q for k in ["chào", "xin chào", "bạn là ai", "tên gì"]):
        answer = "Xin chào bạn! Tôi là trợ lý thị giác AI. Tôi luôn đồng hành quan sát và hỗ trợ bạn di chuyển an toàn."

    # 8. MẶC ĐỊNH SÚC TÍCH, DỄ HIỂU
    else:
        answer = "Tôi đang lắng nghe. Bạn có thể hỏi: Phía trước an toàn không, có người không, hoặc tôi đang ở đâu."

    tts_res = await asyncio.to_thread(blaze_service.text_to_speech, answer)
    audio_url = tts_res.get("audio_url")

    return {
        "success": True,
        "question": request.question,
        "answer": answer,
        "audio_url": audio_url
    }



# ==========================================
# WEBSOCKET CHO REAL-TIME VIDEO TRACKING
# ==========================================

# Danh mục các vật cản thực tế trên đường đi bộ (bỏ qua mèo, tranh, quạt trần, lọ hoa...)
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "table", "dining table", "bed", "bench",
    "backpack", "handbag", "suitcase", "cell phone", "bottle", "cup", "laptop",
    "dog", "car", "motorcycle", "bicycle", "bus", "truck",
    "door", "stairs", "stop sign", "traffic light"
}

import asyncio
import threading

# Các lớp đồ vật lớn (giường, sofa, bàn...) rất dễ bị nhận nhầm với tường/nền -> yêu cầu độ tin cậy cao hơn
LARGE_FALSE_POSITIVE_CLASSES = {"bed", "couch", "table", "dining table", "door", "bench"}

# Thông số camera giả định (điện thoại cầm ngang ngực, FOV dọc ~50°, FOV ngang ~60°)
CAMERA_HEIGHT_M = 1.4
PERSON_HEIGHT_M = 1.7
PERSON_SHOULDER_WIDTH_M = 0.45
MAX_DISTANCE_M = 15.0

# YOLO tracker giữ trạng thái chung -> chỉ cho 1 luồng suy luận chạy tại một thời điểm
_inference_lock = threading.Lock()


def estimate_distance_m(cname: str, box: dict, img_w: int, img_h: int) -> float:
    """
    Ước lượng khoảng cách (mét) bằng hình học camera lỗ kim (pinhole):
    - Người: d = f * H_thật / h_pixel (dùng chiều cao; nếu bị cắt khỏi khung thì dùng bề rộng vai)
    - Vật khác: điểm chạm đất (y2) so với đường chân trời (giữa ảnh) và chiều cao camera
    """
    box_h = max(1.0, float(box["y2"] - box["y1"]))
    box_w = max(1.0, float(box["x2"] - box["x1"]))
    f_y = (img_h / 2.0) / 0.4663   # tan(25°)
    f_x = (img_w / 2.0) / 0.5774   # tan(30°)

    touches_bottom = box["y2"] >= img_h * 0.97
    touches_top = box["y1"] <= img_h * 0.03

    if cname == "person":
        d_h = f_y * PERSON_HEIGHT_M / box_h
        d_w = f_x * PERSON_SHOULDER_WIDTH_M / box_w
        # Người bị cắt đầu/chân khỏi khung -> chiều cao bị ngắn lại -> d_h sai (quá gần). Dùng bề rộng.
        dist = d_w if (touches_bottom or touches_top) else d_h
    else:
        horizon_y = img_h / 2.0
        below = float(box["y2"]) - horizon_y
        if below <= 1.0:
            dist = MAX_DISTANCE_M
        else:
            dist = f_y * CAMERA_HEIGHT_M / below
        # Vật chạm đáy khung và chiếm phần lớn chiều cao -> ngay sát trước mặt
        if touches_bottom and box_h / img_h > 0.5:
            dist = min(dist, 1.0)

    return round(min(MAX_DISTANCE_M, max(0.3, dist)), 1)


def _analyze_frame_sync(image_bytes: bytes) -> dict:
    """Toàn bộ phần xử lý nặng (YOLO + quy tắc) - chạy trong thread riêng, không chặn event loop."""
    from app.vision.decision_engine import is_in_roi, determine_position, generate_guidance
    from app.schemas.vision import Detection, BoundingBox
    from app.vision.tactile_paving_detector import detect_tactile_paving

    with _inference_lock:
        raw_detections, img_width, img_height = run_tracking(image_bytes)

    is_on_tactile = detect_tactile_paving(image_bytes)

    all_objects = []
    roi_detections = []
    has_person_in_roi = False
    has_obstacle_in_roi = False

    for det in raw_detections:
        cname = det["class_name"].lower()
        conf = det["confidence"]
        box = det["bounding_box"]

        if cname not in OBSTACLE_CLASSES:
            continue
        if cname in ("cell phone", "bottle", "cup"):
            min_conf = 0.30
        elif cname == "person":
            min_conf = 0.50
        elif cname in LARGE_FALSE_POSITIVE_CLASSES:
            min_conf = 0.60
        else:
            min_conf = 0.45
        if conf < min_conf:
            continue

        in_corridor = is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
        if not in_corridor and cname in ("cell phone", "bottle", "laptop") and box["y2"] > img_height * 0.30:
            in_corridor = True
        pos = determine_position(box["x1"], box["x2"], img_width)
        is_person = (cname == "person")

        dist_m = estimate_distance_m(cname, box, img_width, img_height)

        # Dưới 1.8 mét mới báo nguy hiểm
        is_danger = in_corridor and dist_m <= 1.8

        if is_danger:
            if is_person:
                has_person_in_roi = True
            else:
                has_obstacle_in_roi = True
            roi_detections.append(Detection(
                class_name=det["class_name"],
                confidence=conf,
                position=pos,
                bounding_box=BoundingBox(**box)
            ))

        all_objects.append({
            "class_name": det["class_name"],
            "class_name_vn": get_vietnamese_name(det["class_name"]),
            "confidence": conf,
            "position": pos,
            "bounding_box": box,
            "in_roi": in_corridor,
            "distance_m": dist_m,
            "is_danger": is_danger
        })

    guidance_data = generate_guidance(roi_detections) or {"text": "", "priority": "LOW"}

    if is_on_tactile:
        if guidance_data["text"]:
            guidance_data["text"] = "Đang ở trên vạch dẫn đường. " + guidance_data["text"]
        else:
            guidance_data["text"] = "Đang đi đúng vạch dẫn đường."
            guidance_data["priority"] = "LOW"
    elif not guidance_data["text"]:
        guidance_data["text"] = "An toàn phía trước."

    return {
        "success": True,
        "objects": all_objects,
        "has_person_in_roi": has_person_in_roi,
        "has_obstacle_in_roi": has_obstacle_in_roi,
        "guidance": guidance_data,
        "img_size": {"width": img_width, "height": img_height}
    }


@app.websocket("/api/v1/vision/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Client đã kết nối tới Video Stream!")

    # Chỉ giữ LẠI FRAME MỚI NHẤT: nếu server xử lý chậm hơn tốc độ gửi thì bỏ frame cũ,
    # tránh hàng đợi dồn ứ làm kết quả trễ (nhìn thấy "người" của vài giây trước).
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
                result = await asyncio.to_thread(_analyze_frame_sync, image_bytes)
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


# Dành cho việc chạy Backend
if __name__ == "__main__":
    # Để chạy server, gõ: python backend/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
