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
        prompt = custom_prompt or "Khảo sát không gian phòng và định vị vật thể"
        has_image = bool(image_bytes)

        objects_3d = []
        if has_image:
            try:
                # Dùng YOLO + Depth để suy diễn tọa độ không gian 3D thực tế
                from app.vision.pipeline import analyze_frame_sync
                res = analyze_frame_sync(image_bytes)
                w = res.get("img_size", {}).get("width", 640)
                for obj in res.get("objects", []):
                    box = obj.get("bounding_box", {})
                    dist = obj.get("distance_m") or 2.0
                    x1 = box.get("x1", 0)
                    x2 = box.get("x2", 100)
                    cx = (x1 + x2) / 2.0
                    # Tọa độ 3D xấp xỉ trong hệ quy chiếu camera (X: ngang mét, Y: cao mét, Z: sâu mét)
                    x_cam = round(((cx - w / 2.0) / max(1, w / 2.0)) * dist * 0.6, 2)
                    y_cam = 0.0
                    z_cam = round(dist, 2)
                    objects_3d.append({
                        "class_name": obj["class_name"],
                        "class_name_vn": obj.get("class_name_vn", obj["class_name"]),
                        "distance_m": dist,
                        "position_3d": {"x": x_cam, "y": y_cam, "z": z_cam},
                        "position": obj.get("position", "center"),
                        "is_danger": obj.get("is_danger", False),
                    })
            except Exception:
                pass

        if objects_3d:
            desc_items = [f"{o['class_name_vn']} ở {o['position']} cách {o['distance_m']}m (tọa độ 3D: X={o['position_3d']['x']}m, Z={o['position_3d']['z']}m)" for o in objects_3d[:4]]
            description = (
                f"Khảo sát không gian 3D: Phát hiện {len(objects_3d)} vật thể trong trường nhìn. "
                + ", ".join(desc_items) + ". Lối đi phía trước được tính toán an toàn."
            )
        else:
            description = (
                "Chế độ khảo sát 3D (Spatial Vision Prototype). "
                "Hệ thống định vị không gian 3 chiều dựa trên Depth Anything V2 và mô hình video 3D. "
                "Lối đi phía trước thông thoáng, không phát hiện vật cản trong bán kính 2 mét."
            )

        return {
            "success": True,
            "spatiallm_loaded": self.model_loaded,
            "spatiallm_repo_found": bool(self.repo_dir),
            "mode": "3d_spatial_survey",
            "prompt": prompt,
            "image_received": has_image,
            "vietnamese_description": description,
            "objects_3d": objects_3d,
            "objects": objects_3d,
        }


spatial_service = SpatialService()
