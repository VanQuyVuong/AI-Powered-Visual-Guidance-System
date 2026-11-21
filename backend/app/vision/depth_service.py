import os
import time
import threading
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, List, Dict, Any
from app.vision.model_scheduler import aux_coordinator, configure_cpu_threads

# Tắt cảnh báo symlink trên Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SEG_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Hệ số hiệu chuẩn độ sâu cho camera thực tế (có thể chỉnh trong .env bằng script calibrate)
DEPTH_SCALE_FACTOR = float(os.getenv("DEPTH_SCALE_FACTOR", "1.0"))

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
    "grandstand": ("stairs", "cầu thang"),
    "mirror": ("mirror", "tấm gương"),
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

# Nhãn nền đất/mặt đường nơi ổ gà / vũng nước thực tế có thể xuất hiện
POTHOLE_GROUND_LABELS = {
    "earth", "road", "route", "sidewalk", "pavement", "path",
    "ground", "water", "sand", "grass", "field",
}

# Nhãn kiến trúc/vật thể không phải mặt đường (dùng lọc báo giả pothole trên cầu thang, cây cối, tường)
POTHOLE_NON_GROUND_LABELS = {
    "stairway", "stairs", "step", "escalator", "grandstand",
    "tree", "plant", "wall", "building", "house", "sky", "person",
    "sofa", "chair", "armchair", "table", "coffee table", "desk",
    "cabinet", "door", "fence", "railing", "bannister", "mirror",
}

STAIR_SEG_LABELS = {"stairway", "stairs", "step", "escalator", "grandstand"}
FLAT_SURFACE_SEG_LABELS = {"floor", "flooring", "sidewalk", "pavement", "path", "ground", "earth", "road"}

AUX_HAZARD_CLASS_MAP = {
    "pothole": ("pothole", "ổ gà"),
    "open_manhole": ("open_manhole", "hố ga hở"),
    "hole": ("dropoff", "hố sâu"),
    "bump": ("step", "gờ bậc"),
    "pole": ("column", "cột trụ"),
    "fence": ("obstacle", "hàng rào"),
    "waste_container": ("obstacle", "vật cản"),
    "animal": ("dog", "động vật"),
}
AUX_GROUND_DEFECT_CLASSES = {"pothole", "open_manhole", "dropoff", "step"}

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
        self.load_complete = False
        self._lock = threading.Lock()

        # Bộ đệm depth map, segmentation map và ổ gà mới nhất
        self._latest_depth_map: Optional[np.ndarray] = None
        self._latest_seg_map: Optional[np.ndarray] = None
        self._latest_seg_hazards_norm: List[Dict[str, Any]] = []
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
        if self.device == "cpu":
            configure_cpu_threads()
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
            backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            repo_root = os.path.abspath(os.path.join(backend_root, ".."))
            candidates = [
                os.path.join(repo_root, "obstacle_yolo11n.pt"),
                os.path.join(backend_root, "obstacle_yolo11n.pt"),
                "obstacle_yolo11n.pt",
                os.path.join(repo_root, "pothole.pt"),
                os.path.join(backend_root, "pothole.pt"),
                "pothole.pt",
            ]
            for p in candidates:
                if os.path.exists(p):
                    self.pothole_model = YOLO(p)
                    print(f"[DepthService] ✅ Đã nạp mô hình vật cản/ổ gà chuyên biệt: {p}")
                    break
        except Exception as e:
            print(f"[DepthService] ⚠️ Không nạp được mô hình vật cản phụ trợ: {e}")
        finally:
            self.load_complete = True

    def compute_depth_map(self, frame_bgr: np.ndarray, force_aux: bool = False) -> Optional[np.ndarray]:
        """
        Tính toán bản đồ độ sâu (depth map) theo mét VÀ bản đồ phân đoạn bề mặt (SegFormer)
        theo nhịp xen kẽ thông minh (hoặc ép chạy đủ mọi nhánh khi `force_aux=True` cho ảnh tĩnh).
        """
        if not self.is_loaded or self.model is None:
            return None

        try:
            self._cycle_count += 1
            cycle_idx = self._cycle_count
            if force_aux:
                run_seg, run_pothole = True, True
            else:
                run_seg, run_pothole = self._should_run_auxiliary_models(cycle_idx)

            h, w = frame_bgr.shape[:2]
            # Giữ nguyên tỷ lệ khung hình dọc/ngang (bội số patch 14 cho ViT) để không bóp méo chiều sâu vật thể
            if h > w:
                d_w, d_h = 210, 280
                seg_w, seg_h = 192, 256
            else:
                d_w, d_h = 280, 210
                seg_w, seg_h = 256, 192

            small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (d_w, d_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(small_rgb)

            # 1. Tính Metric Depth Map ở mọi chu kỳ (giữ nguyên tỷ lệ khung hình, bội số patch 14)
            inputs = self.processor(
                images=pil_img,
                size={"height": d_h, "width": d_w},
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self.model(**inputs).predicted_depth

            raw_depth_np = out.squeeze().cpu().numpy().astype(np.float32, copy=False)
            if DEPTH_SCALE_FACTOR != 1.0:
                raw_depth_np = raw_depth_np * DEPTH_SCALE_FACTOR
            depth_map = cv2.resize(raw_depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            np.clip(depth_map, 0.2, 15.0, out=depth_map)

            # 2. Tính Semantic Segmentation Map giữ tỷ lệ khung hình -> rút trích trực tiếp trên logits nhỏ
            seg_ids_small = None
            seg_hazards_norm = None
            if run_seg and self.seg_loaded and self.seg_model is not None and self.seg_processor is not None:
                try:
                    s_inputs = self.seg_processor(
                        images=pil_img,
                        size={"height": seg_h, "width": seg_w},
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.inference_mode():
                        s_logits = self.seg_model(**s_inputs).logits
                    seg_ids_small = s_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
                    seg_hazards_norm = self._extract_seg_hazards_normalized(seg_ids_small, raw_depth_np)
                except Exception as se:
                    print(f"[DepthService] Lỗi SegFormer: {se}")

            # 3. Chạy phát hiện ổ gà, hố ga, gờ, cột trụ từ Obstacle/Pothole YOLO + lọc ngữ cảnh SegFormer
            pothole_list = None
            active_seg = seg_ids_small if seg_ids_small is not None else self._latest_seg_map
            if run_pothole and self.pothole_model is not None:
                pothole_list = []
                try:
                    with torch.inference_mode():
                        p_res = self.pothole_model(frame_bgr, verbose=False, conf=0.32, imgsz=416)
                    for r in p_res:
                        names = getattr(r, "names", None) or getattr(self.pothole_model, "names", {})
                        for box in r.boxes:
                            cls_idx = int(box.cls[0].item())
                            raw_cls = str(names.get(cls_idx, "pothole")).lower().strip()
                            cls_en, cls_vn = AUX_HAZARD_CLASS_MAP.get(raw_cls, ("pothole", "ổ gà"))
                            conf_val = round(float(box.conf[0].item()), 2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = 0.5 * (x1 + x2)
                            pos = "left" if cx < w / 3.0 else ("right" if cx > 2.0 * w / 3.0 else "center")

                            if cls_en in AUX_GROUND_DEFECT_CLASSES:
                                if not self._is_valid_pothole_box(x1, y1, x2, y2, w, h, active_seg):
                                    continue
                                dist = self._estimate_pothole_distance(x1, y1, x2, y2, depth_map, w, h)
                                if dist > 3.2:
                                    continue
                                danger_d = 2.8 if cls_en in ("open_manhole", "dropoff") else 2.4
                            else:
                                if conf_val < 0.42:
                                    continue
                                dist = self.get_box_median_depth(x1, y1, x2, y2, depth_map) or 2.5
                                if dist > 2.4:
                                    continue
                                danger_d = 1.85

                            pothole_list.append({
                                "class_name": cls_en,
                                "class_name_vn": cls_vn,
                                "confidence": conf_val,
                                "position": pos,
                                "distance_m": dist,
                                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "in_roi": True,
                                "is_danger": dist <= danger_d,
                                "source_model": "ObstacleYOLO11+DepthV2",
                            })
                    has_ground_defect = any(p["class_name"] in AUX_GROUND_DEFECT_CLASSES for p in pothole_list)
                    if not has_ground_defect and active_seg is not None:
                        puddle = self._detect_water_pothole_from_seg(active_seg, depth_map, w, h)
                        if puddle is not None:
                            pothole_list.append(puddle)
                except Exception:
                    pass

            with self._lock:
                self._latest_depth_map = depth_map
                if seg_ids_small is not None:
                    self._latest_seg_map = seg_ids_small
                if seg_hazards_norm is not None:
                    self._latest_seg_hazards_norm = seg_hazards_norm
                if pothole_list is not None:
                    self._latest_potholes = pothole_list
                self._last_computed_time = time.time()

            return depth_map
        except Exception as e:
            print(f"[DepthService] Lỗi tính depth map: {e}")
            return None

    def _extract_seg_hazards_normalized(
        self,
        seg_small: np.ndarray,
        depth_small: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Rút trích sẵn vật thể cấu trúc SegFormer (cầu thang, cột nhà, cửa, tủ...) ngay trên lưới nhỏ (64x64)
        để luồng chính Fusion không phải chạy np.unique() trên hàng trăm nghìn pixel ở mỗi frame (< 0.1ms).
        """
        if seg_small is None or seg_small.size == 0 or not self.seg_loaded or self.seg_model is None:
            return []
        sh, sw = seg_small.shape[:2]
        d_small = None
        if depth_small is not None and depth_small.size > 0:
            d_small = cv2.resize(np.clip(depth_small, 0.2, 15.0), (sw, sh), interpolation=cv2.INTER_NEAREST)

        id2label = getattr(self.seg_model.config, "id2label", {})
        vals, counts = np.unique(seg_small, return_counts=True)
        total_pixels = float(sh * sw)
        extracted: List[Dict[str, Any]] = []

        for cls_id, cnt in zip(vals, counts):
            pix_ratio = cnt / total_pixels
            if pix_ratio < 0.018:
                continue
            raw_label = id2label.get(int(cls_id), "").lower()
            parts = [p.strip() for p in raw_label.replace(";", ",").split(",")]
            matched = None
            is_mirror_cls = False
            is_grandstand_cls = False
            for p in parts:
                if p == "mirror":
                    matched = SEG_OBSTACLE_MAP[p]
                    is_mirror_cls = True
                    break
                if p == "grandstand":
                    matched = SEG_OBSTACLE_MAP[p]
                    is_grandstand_cls = True
                    break
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
            mask = (seg_small == cls_id)
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x1n = float(np.min(xs)) / float(sw)
            y1n = float(np.min(ys)) / float(sh)
            x2n = float(np.max(xs) + 1) / float(sw)
            y2n = float(np.max(ys) + 1) / float(sh)
            box_w_n = max(0.05, x2n - x1n)
            box_h_n = max(0.05, y2n - y1n)
            min_ratio = (0.018 if (box_h_n >= 0.55 and box_w_n >= 0.40) else 0.028) if is_mirror_cls else (0.08 if is_grandstand_cls else 0.05)
            if pix_ratio < min_ratio:
                continue

            if is_mirror_cls:
                # Bên trong gương phản chiếu chiều sâu phòng nên phải đo theo viền khung gương / chiều cao gương
                ring_d = 2.2
                if d_small is not None:
                    mask_u8 = mask.astype(np.uint8)
                    ring = (cv2.dilate(mask_u8, np.ones((7, 7), np.uint8)) > 0) & (~mask)
                    if np.any(ring):
                        ring_d = float(np.percentile(d_small[ring], 25))
                if pix_ratio >= 0.15 or box_h_n >= 0.45:
                    dist_m = round(min(max(0.7, ring_d), 1.7), 1)
                else:
                    dist_m = round(ring_d, 1)
            elif d_small is not None:
                obj_d = d_small[mask]
                raw_d = float(np.median(obj_d)) if obj_d.size > 0 else 1.8
                dist_m = round(min(raw_d, 1.6), 1) if cls_en == "stairs" else round(raw_d, 1)
            else:
                dist_m = 1.6 if cls_en == "stairs" else 1.8

            max_valid_d = 2.5 if cls_en == "mirror" else (2.4 if cls_en == "stairs" else 2.2)
            if dist_m <= max_valid_d:
                cx_n = 0.5 * (x1n + x2n)
                pos = "center" if cls_en in ("stairs", "mirror") else ("left" if cx_n < 1.0 / 3.0 else ("right" if cx_n > 2.0 / 3.0 else "center"))
                extracted.append({
                    "class_name": cls_en,
                    "class_name_vn": cls_vn,
                    "confidence": 0.85 if cls_en in ("stairs", "mirror") else 0.82,
                    "position": pos,
                    "distance_m": dist_m,
                    "norm_box": (x1n, y1n, min(1.0, x2n), min(1.0, y2n)),
                    "in_roi": True,
                    "is_danger": dist_m <= (2.4 if cls_en == "stairs" else (1.85 if cls_en == "mirror" else 2.0)),
                    "is_unknown": True,
                })
        return extracted

    def _get_seg_box_label_ratios(
        self,
        seg_map: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_w: int,
        img_h: int,
    ) -> Dict[str, float]:
        """Tính tỷ lệ các nhóm nhãn SegFormer bên trong một bounding box ảnh gốc."""
        if seg_map is None or seg_map.size == 0 or self.seg_model is None:
            return {}
        sh, sw = seg_map.shape[:2]
        sx1 = max(0, min(sw - 1, int(x1 * sw / max(1, img_w))))
        sy1 = max(0, min(sh - 1, int(y1 * sh / max(1, img_h))))
        sx2 = max(sx1 + 1, min(sw, int(np.ceil(x2 * sw / max(1, img_w)))))
        sy2 = max(sy1 + 1, min(sh, int(np.ceil(y2 * sh / max(1, img_h)))))
        patch = seg_map[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            return {}

        id2label = getattr(self.seg_model.config, "id2label", {})
        vals, counts = np.unique(patch, return_counts=True)
        total = float(patch.size)
        ratios = {"stair": 0.0, "flat": 0.0, "wall": 0.0, "ground": 0.0, "non_ground": 0.0, "water": 0.0}
        for cid, cnt in zip(vals, counts):
            r = float(cnt) / total
            raw = id2label.get(int(cid), "").lower()
            parts = [p.strip() for p in raw.replace(";", ",").split(",")]
            if any(p in STAIR_SEG_LABELS for p in parts):
                ratios["stair"] += r
            if any(p in FLAT_SURFACE_SEG_LABELS for p in parts):
                ratios["flat"] += r
            if any(p == "wall" for p in parts):
                ratios["wall"] += r
            if any(p in POTHOLE_GROUND_LABELS for p in parts):
                ratios["ground"] += r
            if any(p in POTHOLE_NON_GROUND_LABELS for p in parts):
                ratios["non_ground"] += r
            if any(p in ("water", "river", "lake") for p in parts):
                ratios["water"] += r
        return ratios

    def estimate_surface_ring_depth(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        depth_map: Optional[np.ndarray],
        w: int,
        h: int,
    ) -> float:
        """
        Đo khoảng cách tới bề mặt gương / cửa kính từ dải viền khung bao hai bên
        thay vì đo vào lòng kính (nơi phản chiếu chiều sâu phòng xa 3m - 8m).
        """
        bh_n = float(max(1, y2 - y1)) / float(max(1, h))
        bw = max(1, x2 - x1)
        pad_x = max(4, int(bw * 0.08))
        samples: List[float] = []
        if depth_map is not None and depth_map.size > 0:
            dh, dw = depth_map.shape[:2]
            sy1 = max(0, min(dh - 1, int(y1 * dh / max(1, h))))
            sy2 = max(sy1 + 1, min(dh, int(y2 * dh / max(1, h))))
            sx1 = max(0, min(dw - 1, int(x1 * dw / max(1, w))))
            sx2 = max(sx1 + 1, min(dw, int(x2 * dw / max(1, w))))
            left_band = depth_map[sy1:sy2, max(0, sx1 - pad_x):min(dw, sx1 + pad_x)]
            right_band = depth_map[sy1:sy2, max(0, sx2 - pad_x):min(dw, sx2 + pad_x)]
            if left_band.size > 0:
                samples.append(float(np.percentile(left_band, 25)))
            if right_band.size > 0:
                samples.append(float(np.percentile(right_band, 25)))
        ring_d = float(np.median(samples)) if samples else 1.8
        if bh_n >= 0.45:
            ring_d = min(max(0.7, ring_d), 1.7)
        return round(max(0.5, ring_d), 1)

    def _estimate_pothole_distance(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        depth_map: np.ndarray,
        w: int,
        h: int,
    ) -> float:
        """Đo khoảng cách ổ gà ở phần mép gần dưới chân (không lấy nền đường xa ở nửa trên hộp)."""
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px1 = max(0, min(w - 1, int(x1 + bw * 0.15)))
        px2 = max(px1 + 1, min(w, int(x2 - bw * 0.15)))
        py1 = max(0, min(h - 1, int(y1 + bh * 0.55)))
        py2 = max(py1 + 1, min(h, int(y1 + bh * 0.92)))
        patch = depth_map[py1:py2, px1:px2] if depth_map is not None else np.array([])
        dist = float(np.percentile(patch, 25)) if patch.size > 0 else 1.8
        if y2 >= int(h * 0.80):
            dist = min(dist, 2.0)
        return round(max(0.4, dist), 1)

    def _is_valid_pothole_box(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
        seg_map: Optional[np.ndarray],
    ) -> bool:
        """Lọc bỏ hộp ổ gà giả nằm ở nửa trên ảnh, sát mép hẹp hoặc trên cầu thang/tường/nền gạch trong nhà."""
        area_ratio = float(max(1, (x2 - x1) * (y2 - y1))) / float(max(1, w * h))
        if y2 < int(h * 0.42) or area_ratio > 0.90:
            return False
        if x2 < int(w * 0.18) or x1 > int(w * 0.82):
            return False
        if seg_map is None or not self.seg_loaded:
            return True
        ratios = self._get_seg_box_label_ratios(seg_map, x1, y1, x2, y2, w, h)
        if not ratios:
            return True
        if ratios["non_ground"] >= 0.35:
            return False
        if ratios["non_ground"] >= 0.15 and ratios["ground"] < 0.20:
            return False
        return True

    def _detect_water_pothole_from_seg(
        self,
        seg_map: np.ndarray,
        depth_map: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Dict[str, Any]]:
        """Phát hiện hố ngập nước trên đường đất/nhựa khi Pothole YOLO bỏ sót nhưng SegFormer thấy vũng nước lớn dưới chân."""
        if seg_map is None or seg_map.size == 0 or self.seg_model is None:
            return None
        sh, sw = seg_map.shape[:2]
        y1s, y2s = int(sh * 0.45), int(sh * 0.95)
        x1s, x2s = int(sw * 0.10), int(sw * 0.90)
        corridor = seg_map[y1s:y2s, x1s:x2s]
        if corridor.size == 0:
            return None

        id2label = getattr(self.seg_model.config, "id2label", {})
        water_mask = np.zeros_like(corridor, dtype=bool)
        earth_road_ratio = 0.0
        vals, counts = np.unique(corridor, return_counts=True)
        for cid, cnt in zip(vals, counts):
            parts = [p.strip() for p in id2label.get(int(cid), "").lower().replace(";", ",").split(",")]
            if any(p == "water" for p in parts):
                water_mask |= (corridor == cid)
            if any(p in ("earth", "road", "route", "sidewalk", "pavement", "ground", "sand") for p in parts):
                earth_road_ratio += float(cnt) / float(corridor.size)

        water_ratio = float(np.mean(water_mask))
        if water_ratio < 0.20 or earth_road_ratio < 0.08:
            return None

        ys, xs = np.where(water_mask)
        if ys.size == 0:
            return None
        bx1 = int((x1s + int(np.min(xs))) * w / max(1, sw))
        by1 = int((y1s + int(np.min(ys))) * h / max(1, sh))
        bx2 = int((x1s + int(np.max(xs)) + 1) * w / max(1, sw))
        by2 = int((y1s + int(np.max(ys)) + 1) * h / max(1, sh))
        dist = self.get_box_median_depth(bx1, by1, bx2, by2, depth_map) or 1.6
        return {
            "class_name": "pothole",
            "class_name_vn": "ổ gà",
            "confidence": 0.80,
            "position": "center",
            "distance_m": round(min(dist, 2.2), 1),
            "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
            "in_roi": True,
            "is_danger": True,
            "source_model": "SegFormer+WaterPuddle",
        }

    def _is_stairs_suppressed_by_floor_seg(
        self,
        stairs_hazard: Dict[str, Any],
        w: int,
        h: int,
    ) -> bool:
        """Đối chiếu lại bằng SegFormer: loại bỏ báo giả cầu thang khi bề mặt trong hộp rõ ràng là nền gạch phẳng/tường."""
        with self._lock:
            seg_map = self._latest_seg_map
        if seg_map is None or not self.seg_loaded:
            return False
        bb = stairs_hazard.get("bounding_box") or {}
        ratios = self._get_seg_box_label_ratios(
            seg_map, bb.get("x1", 0), bb.get("y1", 0), bb.get("x2", w), bb.get("y2", h), w, h
        )
        if not ratios:
            return False
        if ratios["stair"] >= 0.03:
            return False
        if ratios["flat"] >= 0.48:
            return True
        if ratios["wall"] >= 0.45 and ratios["flat"] >= 0.15:
            return True
        return False

    def trigger_async_compute(self, frame_bgr: np.ndarray):
        """
        Kích hoạt tính depth + segmentation hoàn toàn bất đồng bộ trong background thread
        (trừ khi chưa có bản đồ độ sâu đầu tiên). Điều phối lệch pha với EdgeSAM để không nghẽn CPU.
        """
        if not self.is_loaded:
            return

        with self._lock:
            has_first_map = self._latest_depth_map is not None

        if not has_first_map:
            if not self._is_busy and aux_coordinator.try_acquire("depth"):
                self._is_busy = True
                try:
                    self.compute_depth_map(frame_bgr)
                finally:
                    self._is_busy = False
                    aux_coordinator.release("depth")
            return

        if self._is_busy or not aux_coordinator.try_acquire("depth"):
            return

        def _worker(img_copy):
            self._is_busy = True
            try:
                self.compute_depth_map(img_copy)
            finally:
                self._is_busy = False
                aux_coordinator.release("depth")

        try:
            threading.Thread(target=_worker, args=(frame_bgr.copy(),), daemon=True).start()
        except Exception:
            self._is_busy = False
            aux_coordinator.release("depth")

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
        # Lấy vùng lõi thân trên-giữa (y: 18%..68%, x: 22%..78%) để tránh dính pixel sàn nhà phía trước chân ghế/bàn
        cx1 = max(0, min(w - 1, int(x1 + 0.22 * bw)))
        cy1 = max(0, min(h - 1, int(y1 + 0.18 * bh)))
        cx2 = max(cx1 + 1, min(w, int(x2 - 0.22 * bw)))
        cy2 = max(cy1 + 1, min(h, int(y1 + 0.68 * bh)))

        roi = dmap[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            roi = dmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if roi.size > 0:
            if roi.shape[0] > 40 and roi.shape[1] > 40:
                roi = roi[::2, ::2]
            val = float(np.median(roi))
            return round(val, 1)
        return None

    def _classify_front_region(self, x1: int, y1: int, x2: int, y2: int, ref_w: int = 0, ref_h: int = 0) -> tuple[str, str]:
        """
        Dùng SegFormer để xác định chính xác vật cản trước mặt là 'bức tường' (wall),
        'cửa ra vào' (door), 'cầu thang' (stairs), 'bàn' (table), 'ghế' (chair) hay 'vật cản'.
        """
        with self._lock:
            smap = self._latest_seg_map

        if smap is None or not self.seg_loaded or self.seg_model is None:
            return "obstacle", "vật cản"

        sh, sw = smap.shape[:2]
        if ref_w > 0 and ref_h > 0 and (ref_w != sw or ref_h != sh):
            rx1 = max(0, min(sw - 1, int(x1 * sw / float(ref_w))))
            ry1 = max(0, min(sh - 1, int(y1 * sh / float(ref_h))))
            rx2 = max(rx1 + 1, min(sw, int(x2 * sw / float(ref_w))))
            ry2 = max(ry1 + 1, min(sh, int(y2 * sh / float(ref_h))))
        else:
            rx1, ry1 = max(0, min(sw - 1, x1)), max(0, min(sh - 1, y1))
            rx2 = max(rx1 + 1, min(sw, x2))
            ry2 = max(ry1 + 1, min(sh, y2))
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

        # 1. Quét các vật thể/kiến trúc quan trọng từ SegFormer (dùng danh sách đã rút trích sẵn trên lưới 64x64 -> < 0.05ms)
        with self._lock:
            seg_norm_list = list(self._latest_seg_hazards_norm)

        wall_found_by_seg = False
        for seg_item in seg_norm_list:
            x1n, y1n, x2n, y2n = seg_item["norm_box"]
            bx1 = int(x1n * w)
            by1 = int(y1n * h)
            bx2 = max(bx1 + 1, int(x2n * w))
            by2 = max(by1 + 1, int(y2n * h))
            if _is_covered_by_yolo(bx1, by1, bx2, by2, thresh=0.25):
                continue
            item_copy = {k: v for k, v in seg_item.items() if k != "norm_box"}
            item_copy["bounding_box"] = {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2}
            results.append(item_copy)
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
                if stairs_hazard and not self._is_stairs_suppressed_by_floor_seg(stairs_hazard, w, h):
                    results.append(stairs_hazard)
                    known_boxes.append(stairs_hazard["bounding_box"])
                    has_center_known_obstacle = True

            # A1b. Hợp nhất kết quả Pothole YOLO / Vũng nước ngập sâu từ luồng Pothole
            with self._lock:
                potholes_copy = [dict(p) for p in self._latest_potholes]
            for ph in potholes_copy:
                pb = ph.get("bounding_box") or {}
                px1 = max(0, min(w - 1, int(pb.get("x1", 0))))
                py1 = max(0, min(h - 1, int(pb.get("y1", 0))))
                px2 = max(px1 + 1, min(w, int(pb.get("x2", w))))
                py2 = max(py1 + 1, min(h, int(pb.get("y2", h))))
                if _is_covered_by_yolo(px1, py1, px2, py2, thresh=0.35):
                    continue
                results.append(ph)
                known_boxes.append({"x1": px1, "y1": py1, "x2": px2, "y2": py2})
                if ph.get("position", "center") == "center":
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
                    cls_en, cls_vn = self._classify_front_region(cx1, cy1, cx2, cy2, ref_w=dw, ref_h=dh)
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

        # Downsample bằng stride 3 khi độ phân giải lớn để giảm ~9x pixel quét median (~0.3ms thay vì 2ms)
        sample_map = dmap[::3, ::3] if (dmap.shape[0] >= 180 and dmap.shape[1] >= 180) else dmap
        dh, dw = sample_map.shape[:2]
        y1, y2 = int(dh * 0.22), int(dh * 0.82)
        x_left = int(dw * 0.33)
        x_right = int(dw * 0.67)

        slices = {
            "left": sample_map[y1:y2, 0:x_left],
            "center": sample_map[y1:y2, x_left:x_right],
            "right": sample_map[y1:y2, x_right:dw],
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

