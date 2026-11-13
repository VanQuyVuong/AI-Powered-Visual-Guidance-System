import os
import time
import threading
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, List, Dict, Any

# Tắt cảnh báo symlink trên Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ánh xạ các lớp bề mặt/vật cản của SegFormer ADE20K sang nhãn chuẩn của hệ thống
SEG_OBSTACLE_MAP = {
    "wall": ("wall", "bức tường"),
    "door": ("door", "cửa ra vào"),
    "double door": ("door", "cửa ra vào"),
    "screen door": ("door", "cửa ra vào"),
    "stairway": ("stairs", "cầu thang"),
    "stairs": ("stairs", "cầu thang"),
    "step": ("stairs", "bậc thang"),
    "escalator": ("stairs", "thang cuốn"),
    "column": ("column", "cột nhà"),
    "pillar": ("column", "cột nhà"),
    "cabinet": ("cabinet", "tủ"),
    "wardrobe": ("cabinet", "tủ quần áo"),
    "bookcase": ("cabinet", "tủ sách"),
    "chest of drawers": ("cabinet", "tủ ngăn kéo"),
    "shelf": ("cabinet", "kệ tủ"),
    "refrigerator": ("refrigerator", "tủ lạnh"),
    "table": ("table", "bàn"),
    "desk": ("table", "bàn"),
    "coffee table": ("table", "bàn"),
    "chair": ("chair", "ghế"),
    "armchair": ("chair", "ghế"),
    "swivel chair": ("chair", "ghế xoay"),
    "sofa": ("couch", "ghế sofa"),
    "bed": ("bed", "giường ngủ"),
    "fence": ("obstacle", "hàng rào"),
    "railing": ("obstacle", "lan can"),
    "bannister": ("obstacle", "tay vịn cầu thang"),
    "person": ("person", "người"),
    "individual": ("person", "người"),
}

# Các lớp mặt phẳng đi lại an toàn (không coi là vật cản đứng chắn trước mặt)
SAFE_GROUND_LABELS = {
    "floor", "flooring", "road", "route", "sidewalk", "pavement",
    "path", "earth", "ground", "rug", "carpet", "mat", "grass",
    "field", "sand", "ceiling", "sky",
}


class DepthService:
    """
    Dịch vụ tích hợp 3 mô hình thị giác bổ trợ cho YOLO:
    1. Depth-Anything-V2-Metric-Indoor-Small-hf: Đo khoảng cách thực tế theo mét (m).
    2. SegFormer (nvidia/segformer-b0-finetuned-ade-512-512): Nhận diện BỨC TƯỜNG (wall),
       cửa, cầu thang, cột nhà, bàn ghế khi tiến lại gần hoặc khi YOLO bỏ sót.
    3. Pothole YOLO (pothole.pt): Nhận diện ổ gà / mặt đường sụt lún kèm khoảng cách mét.
    """

    def __init__(self, model_id: str = DEPTH_MODEL_ID, seg_model_id: str = SEG_MODEL_ID):
        self.model_id = model_id
        self.seg_model_id = seg_model_id
        self.device = DEVICE

        self.processor = None
        self.model = None
        self.seg_processor = None
        self.seg_model = None
        self.pothole_model = None

        self.is_loaded = False
        self.seg_loaded = False
        self._lock = threading.Lock()

        # Bộ đệm depth map, segmentation map và ổ gà mới nhất
        self._latest_depth_map: Optional[np.ndarray] = None
        self._latest_seg_map: Optional[np.ndarray] = None
        self._latest_potholes: List[Dict[str, Any]] = []
        self._last_computed_time: float = 0.0
        self._is_busy = False
        self._cycle_count: int = 0

        # Khởi động nạp model trong luồng riêng để server bật tức thì
        threading.Thread(target=self._load_model, daemon=True).start()

    def _should_run_auxiliary_models(self, cycle_idx: int) -> tuple[bool, bool]:
        """
        Điều phối nhịp chạy xen kẽ (Interleaved Multi-Model Scheduling):
        - Depth Anything V2 luôn chạy ở MỌI chu kỳ (100%) vì khoảng cách 3D thay đổi theo từng bước chân.
        - SegFormer chạy ở chu kỳ đầu tiên (khi chưa có `_latest_seg_map`) và mỗi 2 chu kỳ một lần (`cycle_idx % 2 == 0`),
          vì cấu trúc tường/cửa/cột nhà không biến đổi đột ngột từng mili-giây.
        - Pothole YOLO chạy ở chu kỳ đầu tiên và mỗi 3 chu kỳ một lần (`cycle_idx % 3 == 0`).
        Giúp tốc độ cập nhật bản đồ độ sâu 3D tăng gấp ~3 lần mà cả 4 mô hình vẫn hoạt động song song.
        """
        run_seg = (self._latest_seg_map is None) or (cycle_idx % 2 == 0)
        run_pothole = (cycle_idx == 1) or (cycle_idx % 3 == 0)
        return run_seg, run_pothole

    def _load_model(self):
        t0 = time.time()
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
                AutoModelForSemanticSegmentation,
            )
            print(f"[DepthService] Đang nạp Depth Anything V2 ({self.model_id}) trên {self.device.upper()}...")
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
                self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id, local_files_only=True).to(self.device).eval()
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device).eval()
            self.is_loaded = True
            print(f"[DepthService] ✅ Đã nạp xong Depth Anything V2 ({time.time() - t0:.2f}s)")
        except Exception as e:
            print(f"[DepthService] ⚠️ Không thể nạp Depth Anything V2: {e}")
            self.is_loaded = False

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            t1 = time.time()
            print(f"[DepthService] Đang nạp SegFormer ({self.seg_model_id}) nhận diện tường/bề mặt...")
            try:
                self.seg_processor = AutoImageProcessor.from_pretrained(self.seg_model_id, local_files_only=True)
                self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(self.seg_model_id, local_files_only=True).to(self.device).eval()
            except Exception:
                self.seg_processor = AutoImageProcessor.from_pretrained(self.seg_model_id)
                self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(self.seg_model_id).to(self.device).eval()
            self.seg_loaded = True
            print(f"[DepthService] ✅ Đã nạp xong SegFormer ({time.time() - t1:.2f}s)")
        except Exception as e:
            print(f"[DepthService] ⚠️ Không thể nạp SegFormer: {e}")
            self.seg_loaded = False

        try:
            from ultralytics import YOLO
            here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            candidates = [
                os.path.join(here, "pothole.pt"),
                "pothole.pt",
            ]
            for p in candidates:
                if os.path.exists(p):
                    self.pothole_model = YOLO(p)
                    print(f"[DepthService] ✅ Đã nạp mô hình phát hiện ổ gà: {p}")
                    break
        except Exception as e:
            print(f"[DepthService] ⚠️ Không nạp được pothole.pt: {e}")

    def compute_depth_map(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Tính toán bản đồ độ sâu (depth map) theo mét VÀ bản đồ phân đoạn bề mặt (SegFormer)
        theo nhịp xen kẽ thông minh cho ảnh BGR đầu vào.
        """
        if not self.is_loaded or self.model is None:
            return None

        try:
            self._cycle_count += 1
            cycle_idx = self._cycle_count
            run_seg, run_pothole = self._should_run_auxiliary_models(cycle_idx)

            h, w = frame_bgr.shape[:2]
            small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (224, 168), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(small_rgb)

            # 1. Tính Metric Depth Map ở mọi chu kỳ (ép size 252x252 là bội số của patch 14 để tránh processor phóng ngược lên 518x518)
            inputs = self.processor(
                images=pil_img,
                size={"height": 252, "width": 252},
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs).predicted_depth

            depth_map = torch.nn.functional.interpolate(
                out.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

            depth_map = np.clip(depth_map, 0.2, 15.0)

            # 2. Tính Semantic Segmentation Map (ép size 256x256 để giảm 4 lần RAM và tăng tốc 4 lần)
            seg_map_resized = None
            if run_seg and self.seg_loaded and self.seg_model is not None and self.seg_processor is not None:
                try:
                    s_inputs = self.seg_processor(
                        images=pil_img,
                        size={"height": 256, "width": 256},
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        s_logits = self.seg_model(**s_inputs).logits
                    seg_ids = s_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
                    seg_map_resized = cv2.resize(seg_ids, (w, h), interpolation=cv2.INTER_NEAREST)
                except Exception as se:
                    print(f"[DepthService] Lỗi SegFormer: {se}")

            # 3. Chạy phát hiện ổ gà (Pothole) theo nhịp xen kẽ
            pothole_list = None
            if run_pothole and self.pothole_model is not None:
                pothole_list = []
                try:
                    p_res = self.pothole_model(frame_bgr, verbose=False, conf=0.35, imgsz=288)
                    for r in p_res:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            dist = self.get_box_median_depth(x1, y1, x2, y2, depth_map) or 1.8
                            pothole_list.append({
                                "class_name": "pothole",
                                "class_name_vn": "ổ gà",
                                "confidence": round(float(box.conf[0].item()), 2),
                                "position": "center",
                                "distance_m": dist,
                                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "in_roi": True,
                                "is_danger": dist <= 3.5,
                            })
                except Exception:
                    pass

            with self._lock:
                self._latest_depth_map = depth_map
                if seg_map_resized is not None:
                    self._latest_seg_map = seg_map_resized
                if pothole_list is not None:
                    self._latest_potholes = pothole_list
                self._last_computed_time = time.time()

            return depth_map
        except Exception as e:
            print(f"[DepthService] Lỗi tính depth map: {e}")
            return None

    def trigger_async_compute(self, frame_bgr: np.ndarray):
        """
        Kích hoạt tính depth + segmentation hoàn toàn bất đồng bộ trong background thread
        (trừ khi chưa có bản đồ độ sâu đầu tiên). Nhờ `_should_run_auxiliary_models` và kích thước `224x168`,
        luồng cập nhật độ sâu nhanh gấp ~3 lần.
        """
        if not self.is_loaded:
            return

        with self._lock:
            has_first_map = self._latest_depth_map is not None

        if not has_first_map:
            if not self._is_busy:
                self._is_busy = True
                try:
                    self.compute_depth_map(frame_bgr)
                finally:
                    self._is_busy = False
            return

        if self._is_busy:
            return

        def _worker(img_copy):
            self._is_busy = True
            try:
                self.compute_depth_map(img_copy)
            finally:
                self._is_busy = False

        threading.Thread(target=_worker, args=(frame_bgr.copy(),), daemon=True).start()

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_depth_map

    def get_latest_potholes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._latest_potholes)

    def get_box_median_depth(
        self, x1: int, y1: int, x2: int, y2: int, depth_map: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Lấy khoảng cách trung vị (median distance tính bằng mét) của vùng lõi Bounding Box.
        """
        dmap = depth_map if depth_map is not None else self.get_latest_depth_map()
        if dmap is None:
            return None

        h, w = dmap.shape[:2]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        cx1 = max(0, min(w - 1, int(x1 + 0.2 * bw)))
        cy1 = max(0, min(h - 1, int(y1 + 0.2 * bh)))
        cx2 = max(cx1 + 1, min(w, int(x2 - 0.2 * bw)))
        cy2 = max(cy1 + 1, min(h, int(y2 - 0.2 * bh)))

        roi = dmap[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            roi = dmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if roi.size > 0:
            val = float(np.median(roi))
            return round(val, 1)
        return None

    def _classify_front_region(self, x1: int, y1: int, x2: int, y2: int) -> tuple[str, str]:
        """
        Dùng SegFormer để xác định chính xác vật cản trước mặt là 'bức tường' (wall),
        'cửa ra vào' (door), 'cầu thang' (stairs), 'bàn' (table), 'ghế' (chair) hay 'vật cản'.
        """
        with self._lock:
            smap = self._latest_seg_map

        if smap is None or not self.seg_loaded or self.seg_model is None:
            return "obstacle", "vật cản"

        sh, sw = smap.shape[:2]
        rx1, ry1 = max(0, min(sw - 1, x1)), max(0, min(sh - 1, y1))
        rx2, ry2 = max(rx1 + 1, min(sw, x2)), max(ry1 + 1, min(sh, y2))
        patch = smap[ry1:ry2, rx1:rx2]
        if patch.size == 0:
            return "obstacle", "vật cản"

        vals, counts = np.unique(patch, return_counts=True)
        order = np.argsort(-counts)
        total = float(patch.size)

        id2label = getattr(self.seg_model.config, "id2label", {})
        for idx in order:
            cls_id = int(vals[idx])
            ratio = counts[idx] / total
            if ratio < 0.15:
                continue
            raw_label = id2label.get(cls_id, "").lower()
            parts = [p.strip() for p in raw_label.replace(";", ",").split(",")]
            # Bỏ qua nếu là mặt sàn/đường đi
            if any(p in SAFE_GROUND_LABELS for p in parts):
                continue
            for p in parts:
                if p in SEG_OBSTACLE_MAP:
                    return SEG_OBSTACLE_MAP[p]

        return "obstacle", "vật cản"

    def get_or_create_display_depth_map(self, frame_bgr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Trả về bản đồ độ sâu từ Depth Anything V2.
        Nếu trong vài giây đầu model đang nạp lên RAM mà frame đã gửi tới,
        tạo bản đồ độ sâu hình học/quang học tức thì để Màn hình 2 luôn hiển thị mượt mà từ giây đầu tiên.
        """
        dmap = self.get_latest_depth_map()
        if dmap is not None:
            return dmap
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            h, w = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # Gradient phối cảnh từ chân ảnh (gần ~0.8m) lên đỉnh ảnh (xa ~5.0m) kết hợp độ tương phản vật thể
            y_grad = np.linspace(4.5, 0.8, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
            edges = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (15, 15), 0)
            synth_depth = np.clip(y_grad - edges * 1.2 + (0.5 - gray) * 0.6, 0.4, 8.0)
            return synth_depth
        except Exception:
            return None

    def detect_unknown_obstacles(
        self,
        frame_bgr: np.ndarray,
        known_boxes: List[Dict[str, int]],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        HỢP NHẤT ĐA MÔ HÌNH (YOLO + DEPTH ANYTHING V2 + SEGFORMER + GEOMETRIC + POTHOLE):
        1. Ưu tiên số 1: Phát hiện Cầu thang / Bậc thềm (`detect_stairs_and_curbs`) và Hố sâu (`detect_ground_dropoff`)
           TRƯỚC TIÊN để bức tường chiếu nghỉ phía trước cầu thang (1.5m - 1.8m) không bao giờ che mất cảnh báo cầu thang!
        2. Phát hiện Cây cột (`detect_pillar`), Bức tường (`detect_wall`) và Vật cản lạ chắn ngang (`detect_generic_front_obstacle`).
        """
        dmap = depth_map if depth_map is not None else self.get_latest_depth_map()
        if frame_bgr is not None and frame_bgr.size > 0:
            h, w = frame_bgr.shape[:2]
        elif dmap is not None and dmap.size > 0:
            h, w = dmap.shape[:2]
        else:
            return []

        results: List[Dict[str, Any]] = []
        known_boxes = list(known_boxes) if known_boxes else []

        def _is_covered_by_yolo(bx1: int, by1: int, bx2: int, by2: int, thresh: float = 0.35) -> bool:
            area = float(max(1, (bx2 - bx1) * (by2 - by1)))
            for box in known_boxes:
                ix1 = max(bx1, box.get("x1", 0))
                iy1 = max(by1, box.get("y1", 0))
                ix2 = min(bx2, box.get("x2", 0))
                iy2 = min(by2, box.get("y2", 0))
                if ix2 > ix1 and iy2 > iy1:
                    if float((ix2 - ix1) * (iy2 - iy1)) / area > thresh:
                        return True
            return False

        # 1. Quét các vật thể/kiến trúc quan trọng từ SegFormer (cầu thang, cột nhà, cửa) mà YOLO bỏ sót
        with self._lock:
            smap = self._latest_seg_map

        wall_found_by_seg = False
        if smap is not None and self.seg_loaded and self.seg_model is not None:
            id2label = getattr(self.seg_model.config, "id2label", {})
            vals, counts = np.unique(smap, return_counts=True)
            total_pixels = float(h * w)

            for cls_id, cnt in zip(vals, counts):
                if cnt / total_pixels < 0.05:  # Chiếm ít nhất 5% khung hình
                    continue
                raw_label = id2label.get(int(cls_id), "").lower()
                parts = [p.strip() for p in raw_label.replace(";", ",").split(",")]

                matched = None
                for p in parts:
                    # Dùng SegFormer cho cầu thang / cột nhà / cửa / tủ kệ (YOLO COCO không có các lớp này)
                    if p in (
                        "stairway", "stairs", "step", "escalator",
                        "column", "pillar", "door", "double door",
                        "cabinet", "wardrobe", "bookcase", "chest of drawers", "shelf",
                    ):
                        matched = SEG_OBSTACLE_MAP[p]
                        break
                if not matched:
                    continue

                cls_en, cls_vn = matched
                mask = (smap == cls_id).astype(np.uint8)
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    continue
                bx1, by1, bx2, by2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
                if _is_covered_by_yolo(bx1, by1, bx2, by2, thresh=0.25):
                    continue

                if dmap is not None and dmap.size > 0:
                    obj_depths = dmap[mask > 0]
                    raw_d = float(np.median(obj_depths)) if obj_depths.size > 0 else 1.8
                    dist_m = round(min(raw_d, 1.6), 1) if cls_en == "stairs" else round(raw_d, 1)
                else:
                    dist_m = 1.6 if cls_en == "stairs" else 1.8

                max_valid_d = 2.4 if cls_en == "stairs" else 2.2
                if dist_m <= max_valid_d:
                    cx = (bx1 + bx2) / 2.0
                    pos = "center" if cls_en == "stairs" else ("left" if cx < w / 3.0 else ("right" if cx > 2.0 * w / 3.0 else "center"))
                    results.append({
                        "class_name": cls_en,
                        "class_name_vn": cls_vn,
                        "confidence": 0.85 if cls_en == "stairs" else 0.82,
                        "position": pos,
                        "distance_m": dist_m,
                        "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
                        "in_roi": True,
                        "is_danger": dist_m <= (2.4 if cls_en == "stairs" else 2.0),
                        "is_unknown": True,
                    })
                    known_boxes.append({"x1": bx1, "y1": by1, "x2": bx2, "y2": by2})

        # 2. PHÁT HIỆN HÌNH HỌC & QUANG HỌC: ƯU TIÊN CẦU THANG & HỐ SÂU TRƯỚC, SAU ĐÓ MỚI TỚI CỘT / TƯỜNG / VẬT CẢN
        wall_or_pillar_detected = wall_found_by_seg
        try:
            from app.vision.geometric_hazard_detector import geometric_hazard_detector

            has_center_known_obstacle = any(
                (k.get("x1", 0) < 0.65 * w and k.get("x2", 0) > 0.35 * w)
                for k in known_boxes
            )

            # A1. ƯU TIÊN SỐ 1: Kiểm tra Bậc thềm / Cầu thang dưới chân TRƯỚC TIÊN
            # (để khi đi lên/xuống cầu thang có tường chiếu nghỉ ở 1.5m - 1.8m phía trước, cầu thang không bao giờ bị tường chặn mất!)
            if (
                os.getenv("ENABLE_GEOMETRIC_STAIRS", "1").lower() in ("1", "true", "yes")
                and not has_center_known_obstacle
                and frame_bgr is not None
                and frame_bgr.size > 0
            ):
                stairs_hazard = geometric_hazard_detector.detect_stairs_and_curbs(
                    frame_bgr, dmap, known_boxes=known_boxes
                )
                if stairs_hazard:
                    results.append(stairs_hazard)
                    known_boxes.append(stairs_hazard["bounding_box"])
                    has_center_known_obstacle = True

            # A2. Kiểm tra Hố sâu / Sụt lún / Bậc hụt chân dưới đất (Ground Drop-off):
            if not has_center_known_obstacle:
                dropoff_hazard = geometric_hazard_detector.detect_ground_dropoff(
                    dmap, known_boxes=known_boxes
                )
                if dropoff_hazard:
                    results.append(dropoff_hazard)
                    known_boxes.append(dropoff_hazard["bounding_box"])
                    has_center_known_obstacle = True

            # A3. Kiểm tra Cây cột / Trụ đứng (Pillar / Column) chắn trước mặt (cách <= 1.75m):
            if not wall_or_pillar_detected and not has_center_known_obstacle:
                pillar_hazard = geometric_hazard_detector.detect_pillar(
                    dmap, known_boxes=known_boxes, frame_bgr=frame_bgr
                )
                if pillar_hazard:
                    results.append(pillar_hazard)
                    known_boxes.append(pillar_hazard["bounding_box"])
                    wall_or_pillar_detected = True
                    has_center_known_obstacle = True

            # A4. Kiểm tra Bức tường chắn trước mặt:
            if not wall_or_pillar_detected and not has_center_known_obstacle:
                wall_hazard = geometric_hazard_detector.detect_wall(
                    dmap, known_boxes=known_boxes, frame_bgr=frame_bgr
                )
                if wall_hazard:
                    results.append(wall_hazard)
                    known_boxes.append(wall_hazard["bounding_box"])
                    wall_or_pillar_detected = True
                    has_center_known_obstacle = True

            # A5. Kiểm tra Vật cản lạ chắn ngang phía trước (như tấm bạt căng ngang, rèm chắn <= 1.50m):
            if not wall_or_pillar_detected and not has_center_known_obstacle:
                generic_obs = geometric_hazard_detector.detect_generic_front_obstacle(
                    dmap, known_boxes=known_boxes, frame_bgr=frame_bgr
                )
                if generic_obs:
                    results.append(generic_obs)
                    known_boxes.append(generic_obs["bounding_box"])
                    has_center_known_obstacle = True
        except Exception:
            pass

        # 3. QUY TẮC ĐẢO NGƯỢC AN TOÀN P0 (SAFETY INVERSION RULE TỪ DEPTH MAP):
        # Chỉ kích hoạt khi vật cản áp sát <= 0.98m (dưới 1 bước chân) để không báo nhầm khi tường còn cách 2 bước chân!
        if dmap is not None and dmap.size > 0 and not any(
            (k.get("x1", 0) < 0.65 * w and k.get("x2", 0) > 0.35 * w) for k in known_boxes
        ):
            dh, dw = dmap.shape[:2]
            cy1, cy2 = int(dh * 0.20), int(dh * 0.78)
            cx1, cx2 = int(dw * 0.33), int(dw * 0.67)
            center_lane = dmap[cy1:cy2, cx1:cx2]
            if center_lane.size > 0:
                close_ratio = float(np.mean(center_lane < 0.98))
                median_center_z = float(np.median(center_lane))
                if close_ratio >= 0.42 or median_center_z <= 0.95:
                    cls_en, cls_vn = self._classify_front_region(cx1, cy1, cx2, cy2)
                    results.append({
                        "class_name": cls_en,
                        "class_name_vn": cls_vn,
                        "confidence": 0.88,
                        "position": "center",
                        "distance_m": round(max(0.4, min(median_center_z, 0.95)), 1),
                        "bounding_box": {"x1": int(cx1 * w / dw), "y1": int(cy1 * h / dh), "x2": int(cx2 * w / dw), "y2": int(cy2 * h / dh)},
                        "in_roi": True,
                        "is_danger": True,
                        "hazard_type": "DEPTH_BLOCKED_CORRIDOR",
                    })

        return results

    def analyze_three_lanes_depth(self, depth_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Phân tích 3 làn (Trái / Giữa / Phải) trực tiếp từ bản đồ độ sâu (Depth Map):
        Trả về khoảng cách trung vị (m) và tỷ lệ bị chắn (< 1.00m, < 2.0m) của từng làn.
        Chỉ đánh dấu `is_blocked=True` khi khoảng cách <= 1.00m để người dùng còn đủ 2 bước chân đi tiếp khi cách tường 1.2m - 1.8m.
        """
        dmap = depth_map if depth_map is not None else self.get_latest_depth_map()
        if dmap is None or dmap.size == 0:
            return {"has_depth": False, "lanes": {}}

        dh, dw = dmap.shape[:2]
        y1, y2 = int(dh * 0.22), int(dh * 0.82)
        x_left = int(dw * 0.33)
        x_right = int(dw * 0.67)

        slices = {
            "left": dmap[y1:y2, 0:x_left],
            "center": dmap[y1:y2, x_left:x_right],
            "right": dmap[y1:y2, x_right:dw],
        }
        lanes_info = {}
        for name, patch in slices.items():
            if patch.size == 0:
                lanes_info[name] = {"median_m": 4.0, "close_ratio": 0.0, "mid_ratio": 0.0, "is_blocked": False}
                continue
            med = round(float(np.median(patch)), 2)
            close_r = round(float(np.mean(patch < 1.00)), 2)
            mid_r = round(float(np.mean(patch < 2.0)), 2)
            lanes_info[name] = {
                "median_m": med,
                "close_ratio": close_r,
                "mid_ratio": mid_r,
                "is_blocked": (close_r >= 0.42 or med <= 1.00),
            }

        return {"has_depth": True, "lanes": lanes_info}


depth_service = DepthService()

