"""
Pipeline thị giác hợp nhất (Vision Processing Pipeline):
Kết hợp YOLOv8/11 + ByteTrack + Depth Anything V2 + SegFormer + Tactile Paving + Decision Engine
để tạo dữ liệu hướng dẫn thời gian thực cho người khiếm thị.
"""

import time
import threading
from typing import Optional
import cv2
import numpy as np

from app.schemas.vision import Detection, BoundingBox
from app.vision.yolo_service import run_tracking
from app.vision.decision_engine import (
    is_in_roi,
    determine_position,
    generate_guidance,
    build_navigation_speech,
    generate_navigation_direction,
    get_mode,
    get_vietnamese_name,
)
from app.vision.tactile_paving_detector import detect_tactile_paving
from app.vision.depth_service import depth_service
from app.vision.distance_estimator import estimate_distance_m
from app.vision.sam_service import sam_service

# Danh mục các vật cản & đồ vật thực tế trong nhà và ngoài đường
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "table", "dining table", "bed", "bench",
    "backpack", "handbag", "suitcase", "cell phone", "bottle", "cup",
    "wine glass", "bowl", "book", "laptop", "tv", "keyboard", "mouse",
    "potted plant", "refrigerator", "sink", "clock", "vase",
    "dog", "cat", "car", "motorcycle", "bicycle", "bus", "truck",
    "door", "stairs", "stair", "step", "curb", "doorstep", "threshold",
    "stop sign", "traffic light", "fire hydrant",
    "wall", "column", "pillar", "cabinet", "obstacle", "multi_obstacle", "pothole"
}

# Các lớp đồ vật lớn
LARGE_FALSE_POSITIVE_CLASSES = {"bed", "couch"}

# YOLO tracker giữ trạng thái chung -> chỉ cho 1 luồng suy luận chạy tại một thời điểm
_inference_lock = threading.Lock()

# Bộ điều khiển giọng nói dẫn đường (bước sang bên / quét có hướng dẫn khi bị chắn kín)
from app.vision.guided_scan import (  # noqa: E402
    voice_state as _voice_state,
    reset_voice_state,
    update_spatial_scan_and_voice,
)


def analyze_frame_sync(image_bytes: bytes, is_stream: bool = False) -> dict:
    """
    Toàn bộ phần xử lý nặng (YOLO + Depth Anything V2 + SegFormer + Geometric Hazard + Tactile Paving).
    Hàm đồng bộ chạy trong threadpool (asyncio.to_thread).
    """
    if not image_bytes:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")

    # 1. Giải mã ảnh và kích hoạt tính Depth + SegFormer
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) if nparr.size > 0 else None
    if frame_bgr is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")

    depth_service.trigger_async_compute(frame_bgr)

    with _inference_lock:
        raw_detections, img_width, img_height = run_tracking(image_bytes)

    # Kích hoạt EdgeSAM chạy ngầm bổ trợ mặt nạ pixel (0ms latency cho luồng chính)
    sam_service.trigger_async_support(frame_bgr, raw_detections)

    is_on_tactile = detect_tactile_paving(image_bytes)
    dmap = depth_service.get_latest_depth_map()

    all_objects = []
    roi_detections = []
    has_person_in_roi = False
    has_obstacle_in_roi = False

    for det in raw_detections:
        cname = det["class_name"].lower()
        conf = det["confidence"]
        box = det["bounding_box"]

        if cname not in OBSTACLE_CLASSES:
            continue
        if cname in (
            "cell phone", "bottle", "cup", "wine glass", "bowl", "book",
            "backpack", "handbag", "suitcase", "chair", "table", "dining table", "laptop"
        ):
            min_conf = 0.25
        elif cname == "person":
            min_conf = 0.35
        elif cname in LARGE_FALSE_POSITIVE_CLASSES:
            min_conf = 0.45
        else:
            min_conf = 0.30
        if conf < min_conf:
            continue

        in_corridor = is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
        pos = determine_position(box["x1"], box["x2"], img_width)
        is_person = (cname == "person")

        # 2. Ưu tiên lấy khoảng cách mét tinh chỉnh theo Pixel Mask của EdgeSAM + Depth Anything V2,
        # ĐỒNG THỜI đối chiếu chéo với khoảng cách hình học Pinhole để chống lỗi lệch frame (stale depth map)
        # khi lia camera nhanh từ vật gần sang chiếc ghế ở xa (ví dụ ghế cách 5.5m nhưng depth map cũ còn lưu 1.3m).
        pinhole_dist = estimate_distance_m(cname, box, img_width, img_height)
        sam_mask_dist = sam_service.get_mask_refined_depth(box, dmap)
        metric_dist = sam_mask_dist or depth_service.get_box_median_depth(box["x1"], box["y1"], box["x2"], box["y2"], dmap)
        box_h_ratio = (box["y2"] - box["y1"]) / float(max(1, img_height))

        if metric_dist is not None and 0.2 <= metric_dist <= 15.0:
            # Nếu hộp bao rất nhỏ (vật ở xa >= 3.0m theo kích thước quang học) nhưng depth map lại báo quá gần (< 62% pinhole_dist)
            # do bản đồ độ sâu bất đồng bộ chưa kịp cập nhật khi xoay camera -> ưu tiên khoảng cách quang học thực tế!
            if pinhole_dist >= 3.0 and box_h_ratio < 0.22 and metric_dist < pinhole_dist * 0.62:
                dist_m = round(pinhole_dist, 1)
            else:
                dist_m = round(metric_dist, 1)
        else:
            dist_m = round(pinhole_dist, 1)

        overlaps_center = (box["x1"] < 0.65 * img_width and box["x2"] > 0.35 * img_width)
        # 3. Tiêu chí nguy hiểm chuẩn trong nhà: Chỉ cảnh báo khi vật cản tiến vào phạm vi <= 2.25m ở làn giữa (hoặc <= 1.65m ở hai bên).
        # Tuyệt đối không cảnh báo nguy hiểm đối với ghế/bàn ở xa 3m - 5.5m!
        in_path = in_corridor or (pos == "center") or overlaps_center or (box_h_ratio >= 0.28 and dist_m <= 2.25)
        is_danger = (in_path and dist_m <= 2.25) or (dist_m <= 1.65)

        # Chuẩn hóa confidence >= 0.40 cho các đồ vật nhỏ đã qua lọc
        effective_conf = max(conf, 0.42)

        det_item = Detection(
            class_name=det["class_name"],
            class_name_vn=get_vietnamese_name(det["class_name"]),
            confidence=effective_conf,
            position=pos,
            distance_m=dist_m,
            in_roi=in_corridor or in_path,
            is_danger=is_danger,
            bounding_box=BoundingBox(**box)
        )

        if is_danger:
            if is_person:
                has_person_in_roi = True
            else:
                has_obstacle_in_roi = True
            roi_detections.append(det_item)

        all_objects.append({
            "class_name": det["class_name"],
            "class_name_vn": get_vietnamese_name(det["class_name"]),
            "confidence": conf,
            "position": pos,
            "bounding_box": box,
            "in_roi": in_corridor or in_path,
            "distance_m": dist_m,
            "is_danger": is_danger
        })

    # 4. LUÔN CHẠY PHÁT HIỆN CỘT NHÀ, BỨC TƯỜNG, BẬC THỀM, CẦU THANG & VẬT CẢN LẠ (KẾT HỢP DEPTH + GEOMETRIC + EDGESAM)
    # Chỉ truyền các hộp bao của vật thể GẦN (<= 2.25m) vào known_boxes để bàn ghế ở xa (4m - 5.5m) trong lớp học
    # KHÔNG BAO GIỜ vô hiệu hóa việc phát hiện cây cột 1.2m hoặc bức tường ngay trước mặt!
    if frame_bgr is not None:
        near_known_boxes = [
            o["bounding_box"] for o in all_objects if (o.get("distance_m") or 9.9) <= 2.25
        ]
        unknown_obstacles = depth_service.detect_unknown_obstacles(frame_bgr, near_known_boxes, dmap)
        # Bổ sung vật cản lớn từ mặt nạ EdgeSAM nếu ở làn giữa chưa có vật cản gần nào được bắt
        has_near_center_obj = any(
            o.get("position") == "center" and (o.get("distance_m") or 9.9) <= 2.25
            for o in all_objects
        )
        if not unknown_obstacles and not has_near_center_obj:
            near_objs_only = [o for o in all_objects if (o.get("distance_m") or 9.9) <= 2.25]
            unknown_obstacles = sam_service.get_unlabeled_mask_obstacles(near_objs_only, dmap, img_width, img_height)
        for unk in unknown_obstacles:
            all_objects.append(unk)
            if unk.get("is_danger", True):
                has_obstacle_in_roi = True
                unk_det = Detection(
                    class_name=unk["class_name"],
                    class_name_vn=unk.get("class_name_vn"),
                    confidence=max(float(unk["confidence"]), 0.45),
                    position=unk["position"],
                    distance_m=unk.get("distance_m"),
                    in_roi=True,
                    is_danger=True,
                    bounding_box=BoundingBox(**unk["bounding_box"])
                )
                roi_detections.append(unk_det)

    # 5. Chỉ đưa thêm các vật ở gần vào roi_detections nếu roi_detections đang trống
    # (Tường chỉ đưa vào khi <= 1.35m, các vật cản khác <= 2.25m)
    if not roi_detections and all_objects:
        near_objs = [
            o for o in all_objects
            if (o.get("distance_m") or 9.9) <= (1.35 if o.get("class_name") == "wall" else 2.25)
        ]
        nearest = sorted(near_objs, key=lambda o: o.get("distance_m") or 9.9)[:4]
        for obj in nearest:
            obj["is_danger"] = True
            roi_detections.append(
                Detection(
                    class_name=obj["class_name"],
                    class_name_vn=obj.get("class_name_vn"),
                    confidence=max(float(obj["confidence"]), 0.42),
                    position=obj["position"],
                    distance_m=obj.get("distance_m"),
                    in_roi=True,
                    is_danger=True,
                    bounding_box=BoundingBox(**obj["bounding_box"])
                )
            )

    # 5b. BỘ NHỚ CHỐNG RUNG TAY 2.0 GIÂY CHO VẬT CẢN Ở GIỮA:
    now = time.time()
    center_objs_now = [o for o in all_objects if o.get("position") == "center"]
    _voice_state["history"] = [(t, cnt) for (t, cnt) in _voice_state["history"] if (now - t) <= 2.0]
    _voice_state["history"].append((now, len(center_objs_now)))
    max_recent_center = max((cnt for _, cnt in _voice_state["history"]), default=0)

    effective_objects_for_speech = list(all_objects)
    if len(center_objs_now) == 1 and len(all_objects) == 1 and max_recent_center >= 2:
        # Giữ nguyên trạng thái nhiều vật cản ở giữa khi tay rung nhẹ làm rớt bớt box
        effective_objects_for_speech.append(all_objects[0])

    # 6. Thuật toán điều hướng 3 làn (Navigation: straight/left/right/stop + Depth 3-Lane Free-Space)
    nav_data = generate_navigation_direction(roi_detections, img_width, img_height)
    depth_lanes = depth_service.analyze_three_lanes_depth(dmap)
    if depth_lanes.get("has_depth"):
        nav_data["depth_lanes"] = depth_lanes["lanes"]

    # 7. Sinh câu hướng dẫn gom nhóm + chỉ đường (Đi thẳng / Rẽ trái / Rẽ phải / Nhìn trái phải)
    base_guidance = generate_guidance(roi_detections)
    guidance_data = build_navigation_speech(
        base_guidance, effective_objects_for_speech, nav_data, img_width, depth_lanes=depth_lanes
    )

    if is_on_tactile:
        if base_guidance and base_guidance.get("text"):
            guidance_data["text"] = "Đang ở trên vạch dẫn đường. " + guidance_data["text"]
        else:
            guidance_data["text"] = "Đang đi đúng vạch dẫn đường, tiếp tục đi thẳng."
            guidance_data["priority"] = "LOW"

    # 8. BỘ NHỚ QUÉT XUNG QUANH & DẪN ĐƯỜNG 3 GIAI ĐOẠN (Báo cản -> Nhìn trái/phải ghi nhớ -> Chỉ hướng né -> Đi tiếp)
    cur_text, cur_prio, should_speak_now = update_spatial_scan_and_voice(
        guidance_data.get("text", ""),
        guidance_data.get("priority", "LOW"),
        all_objects,
        depth_lanes,
        img_width,
        now,
    )
    guidance_data["priority"] = cur_prio

    if should_speak_now:
        guidance_data["display_text"] = cur_text
        guidance_data["text"] = cur_text
    else:
        guidance_data["display_text"] = _voice_state["last_text"] or cur_text
        if is_stream:
            guidance_data["text"] = ""
        else:
            guidance_data["text"] = cur_text

    # 9. Mã hóa frame_b64 và depth_b64 phục vụ xem trực tiếp trên web monitor (/view-phone)
    import base64
    frame_b64 = None
    if frame_bgr is not None:
        try:
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception:
            pass

    depth_b64 = None
    cur_dmap = dmap if dmap is not None else depth_service.get_or_create_display_depth_map(frame_bgr)
    if cur_dmap is not None:
        try:
            d_norm = cv2.normalize(cur_dmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            d_colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            _, d_buf = cv2.imencode('.jpg', d_colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
            depth_b64 = base64.b64encode(d_buf).decode('utf-8')
        except Exception:
            pass

    return {
        "success": True,
        "mode": get_mode(),
        "objects": all_objects,
        "has_person_in_roi": has_person_in_roi,
        "has_obstacle_in_roi": has_obstacle_in_roi,
        "guidance": guidance_data,
        "navigation": nav_data,
        "frame_b64": frame_b64,
        "depth_b64": depth_b64,
        "depth": {"colormap_b64": depth_b64} if depth_b64 else None,
        "img_size": {"width": img_width, "height": img_height}
    }

