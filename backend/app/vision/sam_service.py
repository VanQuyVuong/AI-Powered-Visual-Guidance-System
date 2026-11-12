"""
SAM Service - Phân đoạn ảnh theo từng pixel (Segment Anything) cho hệ thống dẫn đường.

Thiết kế:
- Ưu tiên dùng EdgeSAM (`edge_sam_3x.pth` ~38MB, RepViT backbone siêu nhanh trên CPU/Edge) từ `backend/third_party/EdgeSAM`.
- Tự động dự phòng (fallback) sang MobileSAM (`mobile_sam.pt`) qua Ultralytics nếu thiếu EdgeSAM.
- Tải LƯỜI (lazy): chỉ nạp model khi gọi lần đầu -> không làm chậm lúc khởi động server.
- Tách riêng hoàn toàn khỏi luồng điều hướng/phát hiện vật cản chính để không ảnh hưởng logic hiện có:
    + segment_boxes(): lấy mặt nạ ôm khít các vật mà YOLO đã phát hiện.
    + segment_points(): phân đoạn theo điểm chạm.

Biến môi trường:
    SAM_MODEL   : tên/đường dẫn trọng số. Mặc định "edge_sam_3x.pth".
"""

import os
import sys
import threading
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

DEFAULT_SAM_MODEL = "edge_sam_3x.pth"
FALLBACK_SAM_MODEL = "mobile_sam.pt"

# Thư mục gốc dự án (nơi chứa edge_sam_3x.pth / mobile_sam.pt)
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_EDGESAM_REPO_DIR = os.path.join(_MODEL_DIR, "backend", "third_party", "EdgeSAM")


class SamService:
    """Bọc EdgeSAM (hoặc Ultralytics MobileSAM), an toàn đa luồng, nạp model lười & hỗ trợ nhận diện vật cản."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("SAM_MODEL", DEFAULT_SAM_MODEL)
        self._model = None
        self._predictor = None
        self._backend_type: Optional[str] = None  # "edgesam" | "ultralytics"
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        # Bộ nhớ đệm mặt nạ EdgeSAM mới nhất để bổ trợ trực tiếp cho Pipeline phát hiện vật cản (0ms latency)
        self._latest_masks: List[dict] = []
        self._latest_masks_time: float = 0.0
        self._latest_img_size: tuple = (640, 480)
        self._bg_running: bool = False
        self._last_bg_trigger: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._predictor is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        if self._load_error is not None:
            return False
        with self._lock:
            if self.is_loaded:
                return True

            # 1. Ưu tiên nạp EdgeSAM (edge_sam_3x.pth) nếu có repo và file trọng số
            edgesam_ckpt = os.path.join(_MODEL_DIR, self.model_name)
            if not os.path.exists(edgesam_ckpt):
                edgesam_ckpt = os.path.join(_MODEL_DIR, DEFAULT_SAM_MODEL)

            if os.path.exists(edgesam_ckpt) and os.path.isdir(_EDGESAM_REPO_DIR):
                try:
                    if _EDGESAM_REPO_DIR not in sys.path:
                        sys.path.insert(0, _EDGESAM_REPO_DIR)
                    import torch
                    from edge_sam import SamPredictor, build_sam

                    t0 = time.time()
                    print(f"[SamService] Loading EdgeSAM ({edgesam_ckpt}) ...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    sam_model = build_sam(checkpoint=edgesam_ckpt)
                    sam_model.to(device)
                    sam_model.eval()
                    self._predictor = SamPredictor(sam_model)
                    self._model = sam_model
                    self._backend_type = "edgesam"
                    self.model_name = os.path.basename(edgesam_ckpt)
                    print(f"[SamService] Loaded EdgeSAM on {device.upper()} ({time.time() - t0:.1f}s)")
                    return True
                except Exception as exc:
                    print(f"[SamService] EdgeSAM load failed ({exc}), falling back to MobileSAM...")

            # 2. Fallback sang Ultralytics SAM (mobile_sam.pt)
            try:
                from ultralytics import SAM

                fallback_name = FALLBACK_SAM_MODEL if self.model_name.endswith(".pth") else self.model_name
                local_path = os.path.join(_MODEL_DIR, fallback_name)
                weights = local_path if os.path.exists(local_path) else fallback_name
                t0 = time.time()
                print(f"[SamService] Loading Ultralytics SAM ({weights}) ...")
                self._model = SAM(weights)
                self._backend_type = "ultralytics"
                self.model_name = os.path.basename(weights)
                print(f"[SamService] Loaded Ultralytics SAM ({time.time() - t0:.1f}s)")
                return True
            except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi tải model
                self._load_error = f"{type(exc).__name__}: {exc}"
                print(f"[SamService] Cannot load SAM: {self._load_error}")
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

    @classmethod
    def _binary_mask_to_dict(cls, mask: np.ndarray, w: int, h: int) -> Optional[dict]:
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(mask_u8.sum())
        if area == 0:
            return None
        ys, xs = np.where(mask_u8 > 0)
        return {
            "area_px": area,
            "area_ratio": round(area / float(w * h), 4),
            "bounding_box": {
                "x1": int(xs.min()), "y1": int(ys.min()),
                "x2": int(xs.max()), "y2": int(ys.max()),
            },
            "polygon": cls._mask_to_polygon(mask_u8),
        }

    @staticmethod
    def _prepare_fast_sam_frame(
        frame_bgr: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        max_side: int = 320,
    ) -> tuple:
        """
        Thu nhỏ khung hình về tối đa `max_side=320px` trước khi đưa qua bộ mã hóa ảnh (Image Encoder)
        của EdgeSAM/MobileSAM, đồng thời nội suy tỷ lệ hộp bao `bboxes` và tọa độ `points`.
        Giúp EdgeSAM chạy nhanh gấp ~4 lần trên CPU/Edge mà vẫn giữ nguyên độ chính xác mặt nạ
        khi ánh xạ ngược về kích thước gốc `(w, h)`.
        """
        h, w = frame_bgr.shape[:2]
        longest = max(h, w)
        if longest <= max_side or max_side <= 0:
            return frame_bgr, bboxes, points, 1.0, 1.0

        scale = float(max_side) / float(longest)
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        sx = float(new_w) / float(w)
        sy = float(new_h) / float(h)

        small_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled_boxes = (
            [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bboxes]
            if bboxes else None
        )
        scaled_points = (
            [[p[0] * sx, p[1] * sy] for p in points]
            if points else None
        )
        return small_frame, scaled_boxes, scaled_points, sx, sy

    def _run_edgesam(
        self,
        frame_bgr: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
    ) -> List[dict]:
        h, w = frame_bgr.shape[:2]
        small_frame, scaled_boxes, scaled_points, _, _ = self._prepare_fast_sam_frame(
            frame_bgr, bboxes=bboxes, points=points, max_side=320
        )
        sh, sw = small_frame.shape[:2]
        out: List[dict] = []
        with self._lock:
            self._predictor.set_image(small_frame, image_format="BGR")
            if scaled_boxes:
                for b in scaled_boxes:
                    box_np = np.array(b, dtype=np.float32)
                    masks, scores, _ = self._predictor.predict(
                        box=box_np,
                        num_multimask_outputs=1,
                    )
                    if masks is not None and len(masks) > 0:
                        best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
                        item = self._binary_mask_to_dict(masks[best_idx], w, h)
                        if item is not None:
                            out.append(item)
            elif scaled_points:
                for pt in scaled_points:
                    pt_np = np.array([pt], dtype=np.float32)
                    lbl_np = np.array([1], dtype=np.int32)
                    masks, scores, _ = self._predictor.predict(
                        point_coords=pt_np,
                        point_labels=lbl_np,
                        num_multimask_outputs=2,
                    )
                    if masks is not None and len(masks) > 0:
                        best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
                        item = self._binary_mask_to_dict(masks[best_idx], w, h)
                        if item is not None:
                            out.append(item)
            else:
                # Nếu không có hộp hay điểm, lấy điểm chính giữa khung hình
                pt_np = np.array([[sw / 2.0, sh / 2.0]], dtype=np.float32)
                lbl_np = np.array([1], dtype=np.int32)
                masks, scores, _ = self._predictor.predict(
                    point_coords=pt_np,
                    point_labels=lbl_np,
                    num_multimask_outputs=2,
                )
                if masks is not None and len(masks) > 0:
                    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
                    item = self._binary_mask_to_dict(masks[best_idx], w, h)
                    if item is not None:
                        out.append(item)
        return out

    def _run(self, frame_bgr: np.ndarray, **prompts) -> List[dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if not self._ensure_loaded():
            return []

        if self._backend_type == "edgesam":
            out = self._run_edgesam(
                frame_bgr,
                bboxes=prompts.get("bboxes"),
                points=prompts.get("points"),
            )
            self._latest_masks = out
            self._latest_masks_time = time.time()
            self._latest_img_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            return out

        h, w = frame_bgr.shape[:2]
        small_frame, scaled_boxes, scaled_points, _, _ = self._prepare_fast_sam_frame(
            frame_bgr,
            bboxes=prompts.get("bboxes"),
            points=prompts.get("points"),
            max_side=320,
        )
        fast_prompts = dict(prompts)
        if scaled_boxes is not None:
            fast_prompts["bboxes"] = scaled_boxes
        if scaled_points is not None:
            fast_prompts["points"] = scaled_points
        with self._lock:  # SAM không an toàn khi gọi song song
            results = self._model(small_frame, imgsz=320, verbose=False, **fast_prompts)

        out: List[dict] = []
        for res in results:
            if res.masks is None:
                continue
            masks = res.masks.data.cpu().numpy()
            for m in masks:
                item = self._binary_mask_to_dict(m, w, h)
                if item is not None:
                    out.append(item)
        self._latest_masks = out
        self._latest_masks_time = time.time()
        self._latest_img_size = (w, h)
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

    # ------------------------------------------------------------------ #
    # CÁC HÀM BỔ TRỢ NHẬN BIẾT VẬT CẢN CHO PIPELINE ĐIỀU HƯỚNG (0ms BLOCKING)
    # ------------------------------------------------------------------ #
    def trigger_async_support(self, frame_bgr: np.ndarray, raw_detections: list, min_interval_s: float = 1.8) -> None:
        """
        Chạy ngầm EdgeSAM trong background thread (nếu chưa có tiến trình SAM nào đang chạy)
        để liên tục cập nhật mặt nạ pixel hỗ trợ YOLO + Depth phát hiện vật cản.
        Nhờ `_prepare_fast_sam_frame(max_side=320)` và chọn lọc tối đa 2 hộp bao quan trọng nhất
        ở lối đi trung tâm, EdgeSAM hoạt động liên tục cùng 3 mô hình còn lại mà không gây nghẽn CPU.
        """
        now = time.time()
        if self._bg_running or (now - self._last_bg_trigger) < min_interval_s:
            return
        if (now - self._latest_masks_time) < 1.4:
            # Đã có mặt nạ mới vừa được cập nhật -> dùng luôn, không cần chạy trùng
            return
        if frame_bgr is None or frame_bgr.size == 0:
            return

        self._bg_running = True
        self._last_bg_trigger = now
        frame_copy = frame_bgr.copy()
        h, w = frame_copy.shape[:2]

        # Ưu tiên tối đa 2 hộp bao lớn nhất / gần trung tâm nhất để giảm số lần gọi decoder
        prioritized_dets = []
        if raw_detections:
            def _box_priority(d: dict) -> float:
                b = d.get("bounding_box", {})
                bx1, by1, bx2, by2 = float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
                area = max(1.0, (bx2 - bx1) * (by2 - by1))
                cx = 0.5 * (bx1 + bx2)
                center_bonus = 1.6 if (0.28 * w <= cx <= 0.72 * w) else 1.0
                return area * center_bonus

            valid_dets = [d for d in raw_detections if isinstance(d, dict) and "bounding_box" in d]
            prioritized_dets = sorted(valid_dets, key=_box_priority, reverse=True)[:2]

        def _worker():
            try:
                if prioritized_dets:
                    boxes = [
                        [d["bounding_box"]["x1"], d["bounding_box"]["y1"], d["bounding_box"]["x2"], d["bounding_box"]["y2"]]
                        for d in prioritized_dets
                    ]
                    if boxes:
                        self.segment_boxes(frame_copy, boxes)
                        return
                # Nếu không có hộp YOLO nào, phân đoạn vùng trung tâm lối đi trước mặt (bắt tấm bạt/vật cản lạ)
                self.segment_points(frame_copy, [[w * 0.5, h * 0.55]])
            except Exception:
                pass
            finally:
                self._bg_running = False

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _box_iou(b1: dict, b2: dict) -> float:
        ix1 = max(float(b1.get("x1", 0)), float(b2.get("x1", 0)))
        iy1 = max(float(b1.get("y1", 0)), float(b2.get("y1", 0)))
        ix2 = min(float(b1.get("x2", 0)), float(b2.get("x2", 0)))
        iy2 = min(float(b1.get("y2", 0)), float(b2.get("y2", 0)))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        a1 = max(1.0, (float(b1.get("x2", 0)) - float(b1.get("x1", 0))) * (float(b1.get("y2", 0)) - float(b1.get("y1", 0))))
        a2 = max(1.0, (float(b2.get("x2", 0)) - float(b2.get("x1", 0))) * (float(b2.get("y2", 0)) - float(b2.get("y1", 0))))
        return inter / (a1 + a2 - inter)

    def get_mask_refined_depth(self, box: dict, dmap: Optional[np.ndarray]) -> Optional[float]:
        """
        Dùng mặt nạ đa giác của EdgeSAM để lấy đúng các pixel thuộc về vật cản (loại bỏ nền tường phía sau hộp YOLO),
        giúp đo khoảng cách mét của vật cản chính xác hơn.
        """
        if dmap is None or not self._latest_masks or (time.time() - self._latest_masks_time) > 4.0:
            return None
        best_mask = None
        best_iou = 0.35
        for m in self._latest_masks:
            mb = m.get("bounding_box")
            if not mb:
                continue
            iou = self._box_iou(box, mb)
            if iou > best_iou:
                best_iou = iou
                best_mask = m
        if not best_mask or not best_mask.get("polygon") or len(best_mask["polygon"]) < 3:
            return None

        dh, dw = dmap.shape[:2]
        mw, mh = self._latest_img_size
        if mw <= 0 or mh <= 0:
            return None

        pts = np.array(
            [[int(p[0] * dw / mw), int(p[1] * dh / mh)] for p in best_mask["polygon"]],
            dtype=np.int32,
        )
        mask_img = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillPoly(mask_img, [pts], 1)
        vals = dmap[mask_img > 0]
        if vals.size < 10:
            return None
        median_d = float(np.median(vals))
        return round(median_d, 2) if 0.2 <= median_d <= 15.0 else None

    def get_unlabeled_mask_obstacles(
        self,
        existing_objects: list,
        dmap: Optional[np.ndarray],
        img_width: int,
        img_height: int,
    ) -> List[dict]:
        """
        Nếu trước mặt có một mặt nạ vật thể lớn do EdgeSAM tách ra (chiếm >= 18% diện tích khung hình ở lối đi giữa)
        và bản đồ độ sâu bên trong mặt nạ đó báo rất gần (<= 1.75m), nhưng YOLO hay bộ dò tường chưa bắt được,
        EdgeSAM sẽ bổ sung cảnh báo vật cản phía trước.
        """
        if existing_objects or dmap is None or not self._latest_masks:
            return []
        if (time.time() - self._latest_masks_time) > 4.0:
            return []

        extra_obstacles: List[dict] = []
        for m in self._latest_masks:
            area_ratio = float(m.get("area_ratio", 0.0))
            mb = m.get("bounding_box") or {}
            if area_ratio < 0.18 or not mb:
                continue
            cx = 0.5 * (float(mb.get("x1", 0)) + float(mb.get("x2", 0)))
            if not (img_width * 0.25 <= cx <= img_width * 0.75):
                continue
            mask_depth = self.get_mask_refined_depth(mb, dmap)
            if mask_depth is not None and mask_depth <= 1.75:
                extra_obstacles.append({
                    "class_name": "obstacle",
                    "class_name_vn": "vật cản",
                    "confidence": 0.78,
                    "position": "center",
                    "bounding_box": mb,
                    "in_roi": True,
                    "distance_m": mask_depth,
                    "is_danger": True,
                })
                break
        return extra_obstacles


# Singleton dùng chung toàn backend
sam_service = SamService()


