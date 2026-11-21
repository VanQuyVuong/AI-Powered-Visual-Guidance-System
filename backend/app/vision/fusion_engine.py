"""
Multi-Model Spatial Fusion Engine (Bộ Tổng Hợp Đa Mô Hình & Điều Hướng Tức Thì).

Giải quyết bài toán thầy hướng dẫn góp ý:
- Khi chạy đồng thời nhiều mô hình (YOLO11 + Depth Anything V2 + EdgeSAM + SegFormer/Geometric 3D),
  dữ liệu thô trả về từ các nhánh có thể chồng lấn, xung đột nhãn (ví dụ: vừa báo ghế vừa báo tường,
  hoặc nhảy khoảng cách khi lia camera) gây hỗn loạn thông tin.
- `MultiModelFusionEngine` đóng vai trò là một "Central Fusion Tool" (Bộ hợp nhất trung tâm):
  1. Đối chiếu chéo khoảng cách Quang học (Pinhole) <-> Độ sâu Mặt nạ (EdgeSAM + Depth V2).
  2. Lọc trùng lặp không gian đa mô hình (Cross-Model Spatial NMS / IoU Deduplication) theo thứ tự ưu tiên ngữ nghĩa:
     [Bậc thềm / Cầu thang] > [Người / Vật thể cụ thể: Ghế, Bàn, Cột nhà, Cửa, Tủ] > [Bề mặt chung: Bức tường, Vật cản lạ].
  3. Làm mượt khoảng cách theo thời gian (Temporal EMA Filtering) chống nhiễu frame đơn lẻ.
  4. Phân tích ma trận thông thoáng 3 làn (Trái - Giữa - Phải) và lập tức tổng hợp quyết định dẫn đường.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Tuple


# Phân cấp độ ưu tiên ngữ nghĩa khi 2 mô hình cùng phát hiện 1 vùng không gian (IoU >= 0.35)
SEMANTIC_PRIORITY: Dict[str, int] = {
    "stairs": 100,
    "step": 95,
    "open_manhole": 98,
    "dropoff": 96,
    "pothole": 95,
    "person": 90,
    "column": 85,
    "pillar": 85,
    "mirror": 86,
    "glass_door": 86,
    "door": 82,
    "chair": 80,
    "table": 80,
    "dining table": 80,
    "desk": 80,
    "cabinet": 78,
    "couch": 78,
    "bed": 78,
    "refrigerator": 78,
    "motorcycle": 85,
    "bicycle": 85,
    "car": 88,
    "wall": 40,       # Bề mặt tường chỉ giữ lại khi vùng đó không phải là vật thể cụ thể
    "obstacle": 30,   # Vật cản chung từ EdgeSAM/Depth có độ ưu tiên thấp hơn nhãn cụ thể của YOLO/SegFormer
}


def compute_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Tính độ giao cắt IoU và tỷ lệ bao phủ giữa 2 bounding box."""
    ax1, ay1, ax2, ay2 = float(box_a.get("x1", 0)), float(box_a.get("y1", 0)), float(box_a.get("x2", 0)), float(box_a.get("y2", 0))
    bx1, by1, bx2, by2 = float(box_b.get("x1", 0)), float(box_b.get("y1", 0)), float(box_b.get("x2", 0)), float(box_b.get("y2", 0))

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    iou = inter / max(1.0, union)
    # Xét cả tỷ lệ hộp nhỏ nằm gọn trong hộp lớn (containment ratio)
    contain = inter / min(area_a, area_b)
    return max(iou, contain * 0.65)


class MultiModelFusionEngine:
    """
    Công cụ hợp nhất thông tin từ 4 nguồn thị giác:
      - YOLO11 (Semantic 2D Objects)
      - Depth Anything V2 (3D Metric Depth & 3-Lane Free Space)
      - EdgeSAM (Instance Mask Refined Depth & Unlabeled Obstacles)
      - SegFormer + Geometric 3D Detector (Pillar, Wall, Stairs, Curb)
    """

    def __init__(self, ema_alpha: float = 0.68, memory_ttl_s: float = 1.6):
        self.ema_alpha = ema_alpha
        self.memory_ttl_s = memory_ttl_s
        # Lưu lịch sử khoảng cách theo khóa (class_name, position) -> (timestamp, smoothed_dist_m)
        self._temporal_tracks: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def reset(self) -> None:
        self._temporal_tracks.clear()

    def _smooth_distance(self, class_name: str, position: str, raw_dist: Optional[float], now: float) -> Optional[float]:
        """Làm mượt khoảng cách theo thời gian (EMA) nhưng phản ứng tức thì khi vật tiến lại gần đột ngột."""
        if raw_dist is None:
            return None
        key = (str(class_name).lower(), str(position).lower())
        prev = self._temporal_tracks.get(key)
        if prev is not None:
            prev_t, prev_d = prev
            if (now - prev_t) <= self.memory_ttl_s:
                # Nếu vật tiến sát nhanh (raw_dist nhỏ hơn trước), ưu tiên cập nhật nhanh (alpha=0.82) để đảm bảo an toàn
                alpha = 0.82 if raw_dist < prev_d else self.ema_alpha
                smoothed = round(alpha * raw_dist + (1.0 - alpha) * prev_d, 1)
                self._temporal_tracks[key] = (now, smoothed)
                return smoothed

        smoothed = round(float(raw_dist), 1)
        self._temporal_tracks[key] = (now, smoothed)
        return smoothed

    def fuse_detections(
        self,
        yolo_objects: List[Dict[str, Any]],
        structural_hazards: List[Dict[str, Any]],
        sam_obstacles: List[Dict[str, Any]],
        depth_lanes: Optional[Dict[str, Any]],
        img_width: int,
        img_height: int,
        now: Optional[float] = None,
        sam_mask_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Hợp nhất toàn bộ kết quả từ các model, khử xung đột và trả về:
          - fused_objects: Danh sách vật cản đã làm sạch và đồng bộ khoảng cách
          - fusion_summary: Báo cáo tổng hợp 3 làn và quyết định điều hướng tức thì
        """
        t_fusion0 = time.perf_counter()
        ts = now if now is not None else time.time()
        # Dọn các track đã quá hạn trực tiếp (tránh tạo dict thừa nếu chưa có track hết hạn)
        if self._temporal_tracks:
            expired = [k for k, v in self._temporal_tracks.items() if (ts - v[0]) > self.memory_ttl_s]
            for k in expired:
                del self._temporal_tracks[k]

        # 1. Gom & tiền chỉ mục tọa độ (tuple số thực phẳng) để loại trùng O(N^2) siêu nhanh (< 0.3ms)
        indexed_candidates: List[Tuple[int, float, float, float, float, float, str, int, Dict[str, Any]]] = []
        # Tuple: (-prio, dist, x1, y1, x2, y2, area, cname, prio, item_dict) -> giữ đơn giản 9 phần tử:
        # (sort_neg_prio, sort_dist, x1, y1, x2, y2, area, cname, item)
        def _pack(src_list, default_source: str):
            for raw in (src_list or ()):
                item = dict(raw)
                item.setdefault("source_model", default_source)
                cname = str(item.get("class_name", "")).lower()
                prio = SEMANTIC_PRIORITY.get(cname, 60)
                d_val = item.get("distance_m")
                dist = float(d_val) if d_val is not None else 9.9
                b = item.get("bounding_box") or {}
                x1 = float(b.get("x1", 0))
                y1 = float(b.get("y1", 0))
                x2 = float(b.get("x2", 0))
                y2 = float(b.get("y2", 0))
                area = max(1.0, (x2 - x1) * (y2 - y1)) if (x2 > x1 and y2 > y1) else 1.0
                indexed_candidates.append((-prio, dist, x1, y1, x2, y2, area, cname, item))

        _pack(yolo_objects, "YOLO11+DepthV2+EdgeSAM")
        _pack(structural_hazards, "SegFormer+Geometric3D")
        _pack(sam_obstacles, "EdgeSAM+DepthV2")

        indexed_candidates.sort(key=lambda t: (t[0], t[1]))

        # 2. Khử trùng lặp không gian đa mô hình (Cross-Model Spatial Deduplication với AABB Early-Skip)
        fused: List[Dict[str, Any]] = []
        kept_meta: List[Tuple[float, float, float, float, float, int]] = []  # (x1, y1, x2, y2, area, prio)
        has_specific_center_hazard = False

        for neg_prio, _, ax1, ay1, ax2, ay2, area_a, c_name, cand in indexed_candidates:
            c_prio = -neg_prio

            if c_name in ("wall", "obstacle") and has_specific_center_hazard:
                continue

            is_duplicate = False
            if ax2 > ax1 and ay2 > ay1:
                for bx1, by1, bx2, by2, area_b, k_prio in kept_meta:
                    if k_prio < c_prio:
                        continue
                    # Fast AABB rejection: bỏ qua ngay nếu 2 hộp không chạm nhau theo trục X hoặc Y
                    if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                        continue
                    inter = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))
                    union = area_a + area_b - inter
                    overlap = max(inter / max(1.0, union), (inter / min(area_a, area_b)) * 0.65)
                    if overlap >= 0.35:
                        is_duplicate = True
                        break

            if is_duplicate:
                continue

            # 3. Làm mượt khoảng cách theo thời gian (EMA)
            smoothed_d = self._smooth_distance(
                c_name,
                str(cand.get("position", "center")),
                cand.get("distance_m"),
                ts,
            )
            if smoothed_d is not None:
                cand["distance_m"] = smoothed_d

            dist_val = float(cand.get("distance_m") if cand.get("distance_m") is not None else 9.9)
            pos_val = str(cand.get("position", "center"))
            if c_name == "wall":
                cand["is_danger"] = dist_val <= 1.00
            elif c_name in ("stairs", "step", "dropoff", "pothole", "open_manhole"):
                cand["is_danger"] = dist_val <= 2.40
            elif c_name in ("column", "pillar", "mirror", "glass_door"):
                cand["is_danger"] = dist_val <= 1.85
            elif c_name == "obstacle":
                cand["is_danger"] = dist_val <= 1.50
            else:
                in_path = bool(cand.get("in_roi")) or (pos_val == "center")
                cand["is_danger"] = (in_path and dist_val <= 2.25) or (dist_val <= 1.65)

            if c_prio >= 75 and pos_val == "center" and cand.get("is_danger"):
                has_specific_center_hazard = True

            fused.append(cand)
            if ax2 > ax1 and ay2 > ay1:
                kept_meta.append((ax1, ay1, ax2, ay2, area_a, c_prio))

        raw_candidate_count = len(indexed_candidates)

        # 4. Tổng hợp trạng thái 3 làn (Trái - Giữa - Phải) & Quyết định điều hướng tức thì
        lanes_raw = (depth_lanes or {}).get("lanes", {}) if depth_lanes else {}
        lane_clearance: Dict[str, Dict[str, Any]] = {}

        for lane_name in ("left", "center", "right"):
            d_info = lanes_raw.get(lane_name, {})
            med_m = float(d_info.get("median_m", 3.5) or 3.5)
            lane_objs = [
                o for o in fused
                if o.get("position") == lane_name and o.get("is_danger")
            ]
            nearest_obj = min(lane_objs, key=lambda x: float(x.get("distance_m") or 9.9)) if lane_objs else None
            nearest_dist = float(nearest_obj.get("distance_m")) if (nearest_obj and nearest_obj.get("distance_m") is not None) else med_m
            eff_clear_m = round(min(med_m, nearest_dist), 2)
            is_blocked = bool(lane_objs) or bool(d_info.get("is_blocked", False) and med_m <= 1.00)

            lane_clearance[lane_name] = {
                "clear_distance_m": eff_clear_m,
                "is_blocked": is_blocked,
                "primary_hazard": (nearest_obj.get("class_name_vn") or nearest_obj.get("class_name")) if nearest_obj else None,
            }

        # Chọn hành động tối ưu tức thì
        if not lane_clearance["center"]["is_blocked"]:
            instant_action = "GO_STRAIGHT"
        elif not lane_clearance["left"]["is_blocked"] and not lane_clearance["right"]["is_blocked"]:
            instant_action = (
                "TURN_LEFT"
                if lane_clearance["left"]["clear_distance_m"] >= lane_clearance["right"]["clear_distance_m"]
                else "TURN_RIGHT"
            )
        elif not lane_clearance["left"]["is_blocked"]:
            instant_action = "TURN_LEFT"
        elif not lane_clearance["right"]["is_blocked"]:
            instant_action = "TURN_RIGHT"
        else:
            instant_action = "TURN_AROUND"

        model_branches = self._build_model_branch_breakdown(
            yolo_objects=yolo_objects or [],
            structural_hazards=structural_hazards or [],
            sam_obstacles=sam_obstacles or [],
            depth_lanes=depth_lanes,
            fused=fused,
            sam_mask_count=sam_mask_count,
        )
        danger_count = sum(1 for o in fused if o.get("is_danger"))
        fusion_latency_ms = round((time.perf_counter() - t_fusion0) * 1000.0, 2)

        return {
            "fused_objects": fused,
            "fusion_summary": {
                "active_models": ["YOLO11", "DepthAnythingV2", "EdgeSAM", "SegFormer+Geometric3D"],
                "raw_candidate_count": raw_candidate_count,
                "fused_obstacle_count": len(fused),
                "danger_obstacle_count": danger_count,
                "dedup_removed_count": max(0, raw_candidate_count - len(fused)),
                "ema_active_tracks": len(self._temporal_tracks),
                "fusion_latency_ms": fusion_latency_ms,
                "model_branches": model_branches,
                "lane_clearance": lane_clearance,
                "instant_action": instant_action,
            },
        }

    def _build_model_branch_breakdown(
        self,
        yolo_objects: List[Dict[str, Any]],
        structural_hazards: List[Dict[str, Any]],
        sam_obstacles: List[Dict[str, Any]],
        depth_lanes: Optional[Dict[str, Any]],
        fused: List[Dict[str, Any]],
        sam_mask_count: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Thống kê chi tiết thông tin thu thập được từ từng nhánh trong 4 mô hình AI
        để hiển thị trực quan trên bảng điều khiển báo cáo hội đồng.
        """
        yolo_names = [
            (o.get("class_name_vn") or o.get("class_name") or "vật thể")
            for o in yolo_objects
        ]
        struct_names = [
            (h.get("class_name_vn") or h.get("class_name") or "cấu trúc")
            for h in structural_hazards
        ]
        sam_names = [
            (s.get("class_name_vn") or s.get("class_name") or "mặt nạ pixel")
            for s in sam_obstacles
        ]

        lanes_raw = (depth_lanes or {}).get("lanes", {}) if depth_lanes else {}
        c_med = float(lanes_raw.get("center", {}).get("median_m", 3.5) or 3.5)
        l_med = float(lanes_raw.get("left", {}).get("median_m", 3.5) or 3.5)
        r_med = float(lanes_raw.get("right", {}).get("median_m", 3.5) or 3.5)

        refined_cnt = sum(1 for o in yolo_objects if o.get("sam_mask_refined"))
        active_masks = int(sam_mask_count) if sam_mask_count is not None else max(refined_cnt, len(sam_obstacles))
        if sam_obstacles or refined_cnt > 0 or active_masks > 0:
            sam_detail = f"{active_masks} mặt nạ • lọc Z {refined_cnt} box" + (
                f" • +{len(sam_obstacles)} vật lạ" if sam_obstacles else ""
            )
        else:
            sam_detail = "Đang quét mặt nạ pixel ngầm (320px)"

        return {
            "yolo11": {
                "model_name": "YOLO11n + ByteTrack",
                "role": "Nhận diện & bám vết vật thể định danh",
                "count": len(yolo_objects),
                "items": yolo_names[:3],
                "detail": ", ".join(yolo_names[:3]) if yolo_names else "Không có vật thể định danh",
            },
            "depth_v2": {
                "model_name": "Depth Anything V2",
                "role": "Đo khoảng cách Metric 3D (Z mét)",
                "has_depth": bool(lanes_raw),
                "center_m": round(c_med, 2),
                "left_m": round(l_med, 2),
                "right_m": round(r_med, 2),
                "detail": f"T:{l_med:.1f}m | G:{c_med:.1f}m | P:{r_med:.1f}m",
            },
            "edgesam": {
                "model_name": "EdgeSAM (RepViT)",
                "role": "Lọc độ sâu theo Pixel Mask & bắt vật lạ",
                "count": len(sam_obstacles),
                "active_masks": active_masks,
                "refined_boxes": refined_cnt,
                "items": sam_names[:2],
                "detail": sam_detail,
            },
            "segformer_geo": {
                "model_name": "SegFormer + Geometric 3D",
                "role": "Bắt cầu thang, hố sâu, tường, cột, tấm bạt",
                "count": len(structural_hazards),
                "items": struct_names[:3],
                "detail": ", ".join(struct_names[:3]) if struct_names else "Mặt sàn & kiến trúc thông thoáng",
            },
        }


# Singleton dùng chung trong toàn bộ pipeline
fusion_engine = MultiModelFusionEngine()

