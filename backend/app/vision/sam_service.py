"""
SAM Service - Phân đoạn ảnh theo từng pixel (Segment Anything) cho hệ thống dẫn đường.

Thiết kế:
- Mặc định dùng MobileSAM (~39MB, chạy được trên CPU laptop) qua Ultralytics.
- Tải LƯỜI (lazy): chỉ nạp model khi gọi lần đầu -> không làm chậm lúc khởi động server.
- KHÔNG nằm trong luồng realtime 30 FPS. Chỉ chạy "theo yêu cầu" (on-demand):
    + segment_boxes(): lấy mặt nạ ôm khít các vật mà YOLO đã phát hiện.
    + segment_points(): phân đoạn theo điểm chạm.
- Tùy chọn SAM 3 (cần file sam3.pt do Meta cấp quyền, đặt tên trong biến môi trường SAM_MODEL).

Biến môi trường:
    SAM_MODEL   : tên/đường dẫn trọng số. Mặc định "mobile_sam.pt".
"""

import os
import threading
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

DEFAULT_SAM_MODEL = "mobile_sam.pt"

# Thư mục lưu trọng số (nằm trong .gitignore qua *.pt)
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SamService:
    """Bọc Ultralytics SAM, an toàn đa luồng, nạp model lười."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("SAM_MODEL", DEFAULT_SAM_MODEL)
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from ultralytics import SAM

                local_path = os.path.join(_MODEL_DIR, self.model_name)
                weights = local_path if os.path.exists(local_path) else self.model_name
                t0 = time.time()
                print(f"[SamService] Đang nạp {weights} ...")
                self._model = SAM(weights)
                print(f"[SamService] Đã nạp xong ({time.time() - t0:.1f}s)")
                return True
            except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi tải model
                self._load_error = f"{type(exc).__name__}: {exc}"
                print(f"[SamService] Không nạp được SAM: {self._load_error}")
                return False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, max_points: int = 60) -> List[List[int]]:
        """Chuyển mặt nạ nhị phân thành đa giác nén gọn (gửi nhẹ qua mạng)."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        # Giới hạn số điểm
        while len(approx) > max_points:
            eps *= 1.5
            approx = cv2.approxPolyDP(largest, eps, True)
        return [[int(p[0][0]), int(p[0][1])] for p in approx]

    def _run(self, frame_bgr: np.ndarray, **prompts) -> List[dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if not self._ensure_loaded():
            return []

        h, w = frame_bgr.shape[:2]
        with self._lock:  # SAM không an toàn khi gọi song song
            results = self._model(frame_bgr, verbose=False, **prompts)

        out: List[dict] = []
        for res in results:
            if res.masks is None:
                continue
            masks = res.masks.data.cpu().numpy()
            for m in masks:
                mask = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                area = int(mask.sum())
                if area == 0:
                    continue
                ys, xs = np.where(mask > 0)
                out.append({
                    "area_px": area,
                    "area_ratio": round(area / float(w * h), 4),
                    "bounding_box": {
                        "x1": int(xs.min()), "y1": int(ys.min()),
                        "x2": int(xs.max()), "y2": int(ys.max()),
                    },
                    "polygon": self._mask_to_polygon(mask),
                })
        return out

    # ------------------------------------------------------------------ #
    def segment_boxes(self, frame_bgr: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[dict]:
        """Phân đoạn từng vật theo hộp YOLO. boxes: [[x1,y1,x2,y2], ...]."""
        if not boxes:
            return []
        return self._run(frame_bgr, bboxes=[list(map(float, b)) for b in boxes])

    def segment_points(self, frame_bgr: np.ndarray, points: Sequence[Sequence[float]]) -> List[dict]:
        """Phân đoạn theo điểm chạm. points: [[x,y], ...] (mỗi điểm = một vật)."""
        if not points:
            return []
        pts = [list(map(float, p)) for p in points]
        return self._run(frame_bgr, points=pts, labels=[1] * len(pts))

    def segment_everything(self, frame_bgr: np.ndarray) -> List[dict]:
        """Phân đoạn mọi thứ trong ảnh (chậm, chỉ dùng để khảo sát)."""
        return self._run(frame_bgr)


# Singleton dùng chung toàn backend
sam_service = SamService()
