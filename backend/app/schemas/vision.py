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
