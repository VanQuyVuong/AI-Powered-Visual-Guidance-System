from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import base64
import os

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
        raw_detections, img_width = run_inference(contents)
        
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
    text = extract_text(contents)
    
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
    result = blaze_service.text_to_speech(
        text=request.text,
        speaker_id=request.speaker_id,
        audio_speed=request.audio_speed
    )
    return result

@app.get("/api/v1/voice/audio/{tts_id}")
async def get_audio_proxy(tts_id: str):
    """
    Stream trực tiếp luồng MP3 từ Blaze.vn về trình duyệt kèm Bearer Token (tránh lỗi 401 Unauthorized)
    """
    audio_content = blaze_service.get_audio_bytes(tts_id)
    if not audio_content:
        raise HTTPException(status_code=404, detail="Không tải được file âm thanh từ Blaze")
    return Response(content=audio_content, media_type="audio/mpeg")

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

@app.websocket("/api/v1/vision/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Client đã kết nối tới Video Stream!")
    try:
        while True:
            try:
                # 1. Nhận frame ảnh từ App (dạng chuỗi base64)
                data = await websocket.receive_text()
                
                # Cắt bỏ phần header data:image/jpeg;base64, nếu có
                if "," in data:
                    data = data.split(",")[1]
                
                image_bytes = base64.b64decode(data)
                
                # 2. Chạy Tracking
                raw_detections, img_width, img_height = run_tracking(image_bytes)
                
                # 3. Decision Engine + ROI Filter + Đo khoảng cách
                from app.vision.decision_engine import is_in_roi, determine_position, generate_guidance
                from app.schemas.vision import Detection, BoundingBox
                from app.vision.tactile_paving_detector import detect_tactile_paving
                
                # Kiểm tra xem người dùng có đang đi trên gạch dẫn đường không
                is_on_tactile = detect_tactile_paving(image_bytes)
                
                all_objects = []
                roi_detections = []
                has_person_in_roi = False
                has_obstacle_in_roi = False
                
                frame_area = float(img_width * img_height)
                
                for det in raw_detections:
                    cname = det["class_name"].lower()
                    conf = det["confidence"]
                    box = det["bounding_box"]
                    
                    # Bỏ qua các vật thể không thuộc đối tượng cản đường hoặc độ tin cậy thấp
                    if cname not in OBSTACLE_CLASSES or conf < 0.38:
                        continue

                    in_corridor = is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
                    pos = determine_position(box["x1"], box["x2"], img_width)
                    is_person = (cname == "person")

                    # ƯỚC LƯỢNG KHOẢNG CÁCH THỰC TẾ (METERS)
                    box_h = box["y2"] - box["y1"]
                    box_w = box["x2"] - box["x1"]
                    box_area = box_w * box_h

                    if is_person:
                        # Người: tỷ lệ chiều cao cơ thể so với khung hình
                        h_ratio = box_h / float(img_height)
                        dist_m = round(max(0.5, 0.9 / max(0.12, h_ratio)), 1)
                    else:
                        # Vật cản: dựa trên điểm tiếp đất và độ lớn
                        top_y = img_height * 0.45
                        ground_progress = min(1.0, max(0.0, (box["y2"] - top_y) / (img_height - top_y)))
                        dist_m = round(max(0.3, 2.6 - (ground_progress * 2.2)), 1)

                    # ĐỐI TƯỢNG ĐƯA LẠI SIÊU GẦN CAMERA (CHOÁN TRÊN 12% KHUNG HÌNH) -> CỰC KỲ GẦN (< 0.5m)
                    if box_area / frame_area > 0.12 and abs((box["x1"] + box["x2"]) / 2 - img_width / 2) < img_width * 0.35:
                        dist_m = min(dist_m, 0.4)

                    # QUY ĐỊNH KHOẢNG CÁCH AN TOÀN: DƯỚI 1.8 MÉT MỚI BÁO NGUY HIỂM / STOP!
                    # Nếu vật cản ở xa (> 1.8m), lối đi vẫn an toàn
                    is_danger = (in_corridor and dist_m <= 1.8)

                    if is_danger:
                        if is_person:
                            has_person_in_roi = True
                        else:
                            has_obstacle_in_roi = True
                            
                        det_obj = Detection(
                            class_name=det["class_name"],
                            confidence=conf,
                            position=pos,
                            bounding_box=BoundingBox(**box)
                        )
                        roi_detections.append(det_obj)
                        
                    all_objects.append({
                        "class_name": det["class_name"],
                        "confidence": conf,
                        "position": pos,
                        "bounding_box": box,
                        "in_roi": in_corridor,
                        "distance_m": dist_m,
                        "is_danger": is_danger
                    })
                    
                # 4. Sinh câu tiếng Việt & Thêm cảnh báo
                guidance_data = generate_guidance(roi_detections) or {"text": "", "priority": "LOW"}
                
                if is_on_tactile:
                    if guidance_data["text"]:
                        guidance_data["text"] = "Đang ở trên vạch dẫn đường. " + guidance_data["text"]
                    else:
                        guidance_data["text"] = "Đang đi đúng vạch dẫn đường."
                        guidance_data["priority"] = "LOW"
                elif not guidance_data["text"]:
                    guidance_data["text"] = "An toàn phía trước."
                    
                # Gửi kết quả về cực nhanh với kích thước ảnh gốc để frontend căn chỉnh chuẩn xác
                await websocket.send_json({
                    "success": True,
                    "objects": all_objects,
                    "has_person_in_roi": has_person_in_roi,
                    "has_obstacle_in_roi": has_obstacle_in_roi,
                    "guidance": guidance_data,
                    "img_size": {"width": img_width, "height": img_height}
                })
            except Exception as frame_error:
                # Bắt lỗi từng ảnh riêng lẻ để không làm sập toàn bộ kết nối WebSocket
                await websocket.send_json({
                    "success": False,
                    "error": str(frame_error)
                })
                
    except WebSocketDisconnect:
        print("📱 Client đã ngắt kết nối.")
    except Exception as e:
        print(f"❌ Lỗi WebSocket: {e}")

# Dành cho việc chạy Backend
if __name__ == "__main__":
    # Để chạy server, gõ: python backend/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
