import os
from typing import Optional, Dict, Any


def _find_spatiallm_dir() -> Optional[str]:
    """Tìm thư mục clone SpatialLM (gitignored), không load model."""
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(here, "1", "SpatialLM"),
        os.path.join(here, "SpatialLM"),
        os.path.join(here, "1"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "README.md")):
            return path
        if os.path.isdir(path) and any(
            name.lower().startswith("spatial") for name in os.listdir(path)
        ):
            return path
    return None


class SpatialService:
    """
    Prototype khảo sát phòng 3D.

    SpatialLM (NeurIPS) CHƯA được load/inference trong pipeline.
    Endpoint này trả JSON ổn định để web demo không vỡ, và báo rõ trạng thái.
    """

    def __init__(self) -> None:
        self.repo_dir = _find_spatiallm_dir()
        self.model_loaded = False
        if self.repo_dir:
            print(f"[SpatialLM] Reference folder found: {self.repo_dir}")
            print("[SpatialLM] Inference is not wired yet. YOLO11 remains the realtime vision layer.")
        else:
            print("[SpatialLM] No repo at 1/SpatialLM/. /inspect-room runs in stub mode.")

    def inspect_room(
        self,
        image_bytes: Optional[bytes] = None,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = custom_prompt or "Khảo sát không gian phòng"
        has_image = bool(image_bytes)

        description = (
            "Chế độ khảo sát 3D đang ở bản prototype. "
            "Model SpatialLM chưa được nạp vào backend, nên hệ thống chưa dựng bản đồ phòng 3 chiều thật. "
            "Hiện tại tầng an toàn realtime vẫn dùng YOLO11: bật camera để nhận diện người và vật cản phía trước. "
            "Muốn mô tả căn phòng bằng giọng nói, hãy hỏi trợ lý các câu có sẵn như an toàn phía trước, có người không."
        )

        return {
            "success": True,
            "spatiallm_loaded": self.model_loaded,
            "spatiallm_repo_found": bool(self.repo_dir),
            "mode": "stub",
            "prompt": prompt,
            "image_received": has_image,
            "vietnamese_description": description,
            "objects": [],
        }


spatial_service = SpatialService()
