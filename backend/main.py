from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64

# Import các thành phần chúng ta vừa tạo
from app.schemas.vision import VisionAnalyzeResponse, Detection, Guidance, BoundingBox
from app.vision.yolo_service import run_inference, run_tracking
from app.vision.decision_engine import determine_position, generate_guidance, is_in_roi

app = FastAPI(title="AI Visual Guidance API", version="0.1 (MVP)")

# Cho phép App trên điện thoại gọi API mà không bị chặn CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# WEBSOCKET CHO REAL-TIME VIDEO TRACKING
# ==========================================

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
                
                # 3. Decision Engine + ROI Filter
                
                final_detections = []
                for det in raw_detections:
                    box = det["bounding_box"]
                    
                    # QUAN TRỌNG: Lọc bằng Vùng An Toàn (ROI)
                    if not is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height):
                        continue # Bỏ qua, vật thể này an toàn!
                        
                    pos = determine_position(box["x1"], box["x2"], img_width)
                    final_detections.append({
                        "id": det.get("id"),
                        "class": det["class_name"],
                        "position": pos,
                        "box": box
                    })
                    
                # Gửi kết quả về cực nhanh
                await websocket.send_json({
                    "success": True,
                    "objects": final_detections
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
