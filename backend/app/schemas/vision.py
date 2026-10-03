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
    class_name: str
    confidence: float
    position: Optional[str] = None
    bounding_box: BoundingBox

# Cấu trúc lời cảnh báo
class Guidance(BaseModel):
    text: str
    priority: Optional[str] = None
    audio_url: Optional[str] = None

# Cấu trúc API Response trả về cho Flutter
class VisionAnalyzeResponse(BaseModel):
    success: bool
    detections: List[Detection]
    guidance: Optional[Guidance] = None
    error: Optional[dict] = None
