from pydantic import BaseModel
from typing import List, Optional

# Cấu trúc tọa độ
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

# Cấu trúc của 1 vật thể tìm thấy
class Detection(BaseModel):
    model_config = {"extra": "allow"}

    class_name: str
    class_name_vn: Optional[str] = None
    confidence: float
    position: Optional[str] = None
    distance_m: Optional[float] = None
    in_roi: Optional[bool] = None
    is_danger: Optional[bool] = None
    bounding_box: BoundingBox

# Cấu trúc lời cảnh báo
class Guidance(BaseModel):
    model_config = {"extra": "allow"}

    text: str
    priority: Optional[str] = None
    has_person: Optional[bool] = False

# Cấu trúc API Response trả về cho Flutter
class VisionAnalyzeResponse(BaseModel):
    success: bool
    detections: List[Detection]
    guidance: Optional[Guidance] = None
    error: Optional[dict] = None


def build_light_client_payload(result: dict) -> dict:
    """
    Rút gọn payload gửi về Mobile/Edge client (~1-2KB, bỏ frame_b64/depth_b64 và telemetry nặng)
    để giảm tối đa băng thông và độ trễ qua Wi-Fi.
    """
    nav = result.get("navigation")
    light_nav = None
    if isinstance(nav, dict):
        light_nav = {
            "direction": nav.get("direction", "straight"),
            "confidence": nav.get("confidence", 0.9),
            "text": nav.get("text"),
        }
    return {
        "success": result.get("success", True),
        "mode": result.get("mode"),
        "processing_ms": result.get("processing_ms"),
        "objects": result.get("objects"),
        "has_person_in_roi": result.get("has_person_in_roi"),
        "has_obstacle_in_roi": result.get("has_obstacle_in_roi"),
        "guidance": result.get("guidance"),
        "navigation": light_nav,
        "img_size": result.get("img_size"),
    }
