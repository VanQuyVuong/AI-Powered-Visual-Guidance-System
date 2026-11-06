"""
Router API phân đoạn SAM (độc lập, KHÔNG sửa main.py).

Cách gắn vào backend (thêm đúng 2 dòng trong backend/main.py, sau khi `app = FastAPI(...)`):

    from app.vision.sam_router import router as sam_router
    app.include_router(sam_router)

Endpoint:
    GET  /api/v1/vision/segment/status   -> trạng thái model SAM
    POST /api/v1/vision/segment          -> phân đoạn ảnh (multipart: file=<ảnh>)
         form (tùy chọn): x, y  -> phân đoạn vật tại điểm chạm (pixel của ảnh gửi lên)
         không có x,y           -> YOLO tìm vật, SAM vẽ mặt nạ ôm khít từng vật
"""

import asyncio
import base64
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.vision.decision_engine import get_vietnamese_name
from app.vision.sam_service import sam_service

router = APIRouter(prefix="/api/v1/vision", tags=["sam-segmentation"])

MAX_BOXES = 6  # giới hạn số box ưu tiên nhất để MobileSAM chạy nhanh trên CPU (~0.5-1s)


def _render_mask_overlay(frame: np.ndarray, objects: list) -> Optional[str]:
    """Tạo ảnh overlay vẽ mặt nạ phân đoạn màu sắc bán trong suốt dạng Base64 JPEG."""
    if frame is None:
        return None
    if not objects:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("utf-8")
    try:
        overlay = frame.copy()
        rng = np.random.default_rng(42)
        for obj in objects:
            poly = obj.get("polygon")
            if poly and len(poly) >= 3:
                color = tuple(int(c) for c in rng.integers(60, 255, 3))
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

                b = obj.get("bounding_box", {})
                label = obj.get("class_name_vn") or obj.get("class_name", "")
                if label and "x1" in b and "y1" in b:
                    cv2.putText(
                        frame,
                        label,
                        (b["x1"], max(18, b["y1"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        _, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None


def _segment_sync(
    image_bytes: bytes,
    x: Optional[float] = None,
    y: Optional[float] = None,
    existing_dets: Optional[list] = None,
) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) if nparr.size > 0 else None
    if frame is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")
    h, w = frame.shape[:2]
    t0 = time.time()

    # Chế độ 1: phân đoạn theo điểm chạm (Promptable Point)
    if x is not None and y is not None:
        segs = sam_service.segment_points(frame, [[x, y]])
        objects = [{"class_name": "object", "class_name_vn": "vật thể", "confidence": None, **s} for s in segs]
    else:
        # Chế độ 2: Dùng ngay các hộp YOLO sẵn có từ WebSocket (KHÔNG khóa _inference_lock của YOLO!)
        if existing_dets is not None:
            dets = [
                d for d in existing_dets
                if isinstance(d, dict) and "bounding_box" in d and all(k in d["bounding_box"] for k in ("x1", "y1", "x2", "y2"))
            ]
        else:
            from app.vision.yolo_service import run_inference
            dets, _ = run_inference(image_bytes)

        dets = sorted(dets, key=lambda d: d.get("confidence") or 0.5, reverse=True)[:MAX_BOXES]
        if dets:
            boxes = [[d["bounding_box"][k] for k in ("x1", "y1", "x2", "y2")] for d in dets]
            segs = sam_service.segment_boxes(frame, boxes)
            objects = []
            for d, s in zip(dets, segs):
                objects.append({
                    "class_name": d.get("class_name", "object"),
                    "class_name_vn": d.get("class_name_vn") or get_vietnamese_name(d.get("class_name", "object")),
                    "confidence": d.get("confidence"),
                    **s,
                })
        else:
            # Nếu khung hình không có hộp YOLO nào (VD: đang nhìn vào bức tường hoặc bậc thềm), phân đoạn tâm ảnh
            segs = sam_service.segment_points(frame, [[w * 0.5, h * 0.55]])
            objects = [{"class_name": "surface", "class_name_vn": "bề mặt phía trước", "confidence": None, **s} for s in segs]

    if not objects and sam_service.load_error:
        raise RuntimeError(f"SAM không khả dụng: {sam_service.load_error}")

    overlay_b64 = _render_mask_overlay(frame, objects)

    return {
        "success": True,
        "model": sam_service.model_name,
        "img_size": {"width": w, "height": h},
        "count": len(objects),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "overlay_b64": overlay_b64,
        "objects": objects,
    }


class SamBase64Request(BaseModel):
    image_b64: str
    x: Optional[float] = None
    y: Optional[float] = None
    detections: Optional[list] = None


@router.get("/segment/status")
async def sam_status():
    return {
        "model": sam_service.model_name,
        "loaded": sam_service.is_loaded,
        "error": sam_service.load_error,
    }


@router.post("/segment")
async def sam_segment(
    file: UploadFile = File(...),
    x: Optional[float] = Form(None),
    y: Optional[float] = Form(None),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Ảnh rỗng (INVALID_IMAGE)")
    try:
        return await asyncio.to_thread(_segment_sync, image_bytes, x, y, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/segment_base64")
async def sam_segment_base64(req: SamBase64Request):
    """API phân đoạn SAM nhận trực tiếp frame Base64 + hộp YOLO có sẵn từ WebSocket stream."""
    try:
        data = req.image_b64
        if "," in data:
            data = data.split(",", 1)[1]
        image_bytes = base64.b64decode(data)
        return await asyncio.to_thread(_segment_sync, image_bytes, req.x, req.y, req.detections)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi giải mã Base64: {exc}") from exc

